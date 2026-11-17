import torch
import torch.nn as nn

from ..neurons.lif import MultiStepLIF, seq_apply


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
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125

        self.proj_lif = MultiStepLIF(tau=2.0)

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIF(tau=2.0)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIF(tau=2.0)

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIF(tau=2.0)

        # Preserve the upstream lower threshold in the attention-output LIF.
        self.attn_lif = MultiStepLIF(tau=2.0, threshold=0.5)
        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)

    def _heads(self, t: torch.Tensor) -> torch.Tensor:
        # [T, B, C, N] -> [T, B, heads, N, C/heads]
        T, B, C, N = t.shape
        return t.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, B, C, H, W]
        T, B, C, H, W = x.shape
        x = self.proj_lif(x)
        x = x.flatten(3)  # [T, B, C, N]
        N = x.shape[-1]

        q = self.q_lif(seq_apply(lambda z: self.q_bn(self.q_conv(z)), x))
        k = self.k_lif(seq_apply(lambda z: self.k_bn(self.k_conv(z)), x))
        v = self.v_lif(seq_apply(lambda z: self.v_bn(self.v_conv(z)), x))

        q, k, v = self._heads(q), self._heads(k), self._heads(v)

        # Spike-driven self-attention (no softmax), same association order as
        # upstream: Q @ (K^T @ V) is O(N*d^2) instead of O(N^2*d).
        x = (q @ (k.transpose(-2, -1) @ v)) * self.scale

        # [T, B, heads, N, head_dim] -> [T, B, C, N]
        x = x.transpose(3, 4).reshape(T, B, C, N)
        x = self.attn_lif(x)
        x = seq_apply(lambda z: self.proj_bn(self.proj_conv(z)), x)
        return x.reshape(T, B, C, H, W)
