import torch
import torch.nn as nn
from functools import partial

from timm.models import create_model
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

from ..neurons.lif import LIF
from ..tokenizer.spk_tokenizer import SpikingTokenizer
from ..attention.spk_self_attention import SpikingSelfAttention
from ..mlp.mlp import MLP

class SpikingTransformer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
    ):
        super().__init__()
        # Retained for architectural/state-dict continuity with upstream. The
        # original Spikingformer forward path does not invoke norm1/norm2 or
        # drop_path either.
        self.norm1 = norm_layer(dim)
        self.attn = SpikingSelfAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spike-driven residual learning, unchanged from upstream.
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x



class ViTSNN(nn.Module):
    def __init__(
            self,
            img_size_h=128,
            img_size_w=128,
            patch_size=16,
            in_channels=2,
            num_classes=11,
            embed_dims=256,
            num_heads=4,
            mlp_ratios=4,
            qkv_bias=False,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=nn.LayerNorm,
            depths=6,
            sr_ratios=1,
            T=4,
            pretrained_cfg=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T = int(T)
        if self.T<1:
            raise ValueError(f"Invalid value for T: {self.T}. T must be a positive integer.")

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        self.patch_embed = SpikingTokenizer(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims
        )

        self.block = nn.ModuleList([
            SpikingTransformer(
                dim=embed_dims,
                num_heads=num_heads,
                mlp_ratio=mlp_ratios,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios
            ) for i in range(depths)])

        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def reset_state(self) -> None:
        """Reset every LIF state owned by this model at an episode boundary.

        Do this recursively and per instance instead of calling
        ``snntorch.utils.reset(self)``.  In snnTorch 1.0.0 that helper only
        inspects the modules directly registered under ``self`` before invoking
        a class-level reset.  Spikingformer's LIF neurons are nested inside the
        tokenizer and Transformer blocks, so relying on that helper can leave
        their membrane tensors attached to the previous minibatch graph.
        ``reset_mem`` replaces each membrane tensor with a fresh zero tensor,
        severing the old autograd graph before the next sequence starts.
        """
        for module in self.modules():
            if isinstance(module, LIF):
                module.reset_mem()

    def clear_state(self) -> None:
        """Release membrane tensors (set to empty); the next step re-initialises them to zeros."""
        for module in self.modules():
            if isinstance(module, LIF):
                module.mem = module.mem.new_zeros(0)

    def forward_sequence(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Run a [T, B, C, H, W] input sequence and return per-step features [T, B, D].

        Layers are evaluated layer-by-layer over the whole sequence (each LIF
        loops over T internally), so every Conv/BatchNorm sees all T*B samples
        at once.  This is the upstream Spikingformer data flow and is what
        keeps BatchNorm train/eval statistics consistent.

        Neuron state is *not* reset here.
        """
        x, _ = self.patch_embed(x_seq)
        for blk in self.block:
            x = blk(x)
        return x.flatten(3).mean(3)

    def forward_step(self, x: torch.Tensor) -> torch.Tensor:
        """Run exactly one simulation timestep ([B, C, H, W]) and return pooled features.

        Neuron state is *not* reset here; callers control episode boundaries
        with ``reset_state()``.  In eval mode chaining T calls is numerically
        identical to ``forward_features``.  Do not use this for training:
        BatchNorm would only see one timestep per call.
        """
        return self.forward_sequence(x.unsqueeze(0))[0]

    def forward_features(self, x: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        """Run the native T-step simulation from a static image batch."""
        self.reset_state()
        x_seq = x.unsqueeze(0).expand(self.T, *x.shape)
        features = self.forward_sequence(x_seq)  # [T, B, D]
        # Drop membrane references once the episode is done.  The autograd
        # graph keeps what backward needs; holding the tensors on the module
        # only pins memory, gets broadcast by DDP (mem is a buffer) and breaks
        # deepcopy (e.g. ModelEmaV2) of a model that just ran a train step.
        self.clear_state()
        if return_sequence:
            return features
        return features.mean(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


@register_model    
def Spikingformer(pretrained=False, **kwargs):
    # timm injects these (version dependent); the model does not use them.
    for k in ("pretrained_cfg", "pretrained_cfg_overlay", "drop_block_rate"):
        kwargs.pop(k, None)
    model = ViTSNN(**kwargs)
    model.default_cfg = _cfg()
    return model


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 3, 32, 32, device=device)
    model = create_model(
        "Spikingformer",
        pretrained=False,
        drop_rate=0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        img_size_h=32,
        img_size_w=32,
        patch_size=4,
        embed_dims=384,
        num_heads=12,
        mlp_ratios=4,
        in_channels=3,
        num_classes=10,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=4,
        sr_ratios=1,
        T=4,
    ).to(device)
    model.eval()
    with torch.no_grad():
        y = model(x)
    print(y.shape)
    print("Native snnTorch Spikingformer smoke test passed")
