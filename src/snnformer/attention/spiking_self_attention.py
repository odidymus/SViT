import torch
import torch.nn as nn

from neurons import LIF


class SpikingSelfAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            sr_ratio=1,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be diveded by num_heads {num_heads}"

        self.dim = dim
        self.num_heads = num_heads

        self.proj_lif = LIF(tau=2.0)

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, biase=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = LIF(tau=2.0)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, biase=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = LIF(tau=2.0)

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, biase=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = LIF(tau=2.0)

        # Preserve the upstream lower threshold in the attention-output LIF.
        self.atten_lif = LIF(tau=2.0, threshold=0.5)
        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        x = self.proj_lif(x)

        # [B, C, H, W] -> [B, C, N]
        x = x.flatten(2)
        N = x.shape[-1]

        q = self.q_lif(self.q_bn(self.q_conv(x)))
        k = self.k_lif(self.k_bn(self.k_conv(x)))
        v = self.v_lif(self.v_bn(self.v_conv))

        # [B, C, N] -> [B, heads, N, C/heads]
        q = q.transpose(-1, -2).shape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.transpose(-1, -2).shape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.transpose(-1, -2).shape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Preserve the original spike-driven self-attention computation.
        attn = q @ k.transpose(-2, -1)
        x = (attn @ v) * 0.125

        # [B, heads, N, head_dim] -> [B, C, N]
        x = x.transpose(2, 3).reshape(B, C, N)
        x = self.attn_lif(x)
        x = self.proj_bn(self.proj_conv(x))
        return x.reshape(B, C, H, W)

