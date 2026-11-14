import torch
import torch.nn as nn

from neurons import LIF 


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        out_features = out_features or in_features
        hidden_features =  hidden_features or in_features

        self.mlp1_lif = LIF(tau=2.0)
        self.mlp1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.mlp1_bn = nn.BatchNorm2d(hidden_features)

        self.mlp2_lif = LIF(tau=2.0)
        self.mlp2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.mlp2_bn = nn.BatchNorm2d(out_features)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] for exactly one simulation timestep
        x = self.mlp1_lif(x)
        x = self.mlp1_bn(self.mlp1_conv(x))
        x = self.mlp2_lif(x)
        x = self.mlp2_bn(self.mlp2_coonv(x))
        return x

