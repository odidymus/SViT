import torch
import torch.nn as nn

from timm.models.layers import to_2tuple

from ..neurons.lif import MultiStepLIF, seq_apply


class SpikingTokenizer(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        self.patch_size = to_2tuple(patch_size)
        self.C = in_channels
        self.H = self.image_size[0] // self.patch_size[0]
        self.W = self.image_size[1] // self.patch_size[1]
        self.num_patches = self.H * self.W

        self.block0_conv = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.block0_bn = nn.BatchNorm2d(embed_dims // 8)

        self.block1_lif = MultiStepLIF(tau=2.0)
        self.block1_conv = nn.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.block1_bn = nn.BatchNorm2d(embed_dims // 4)

        self.block2_lif = MultiStepLIF(tau=2.0)
        self.block2_conv = nn.Conv2d(embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.block2_bn = nn.BatchNorm2d(embed_dims // 2)

        self.block3_lif = MultiStepLIF(tau=2.0)
        self.block3_mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.block3_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.block3_bn = nn.BatchNorm2d(embed_dims)

        self.block4_lif = MultiStepLIF(tau=2.0)
        self.block4_mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.block4_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.block4_bn = nn.BatchNorm2d(embed_dims)

    def forward(self, x: torch.Tensor):
        # x: [T, B, C, H, W]
        x = seq_apply(lambda z: self.block0_bn(self.block0_conv(z)), x)

        x = self.block1_lif(x)
        x = seq_apply(lambda z: self.block1_bn(self.block1_conv(z)), x)

        x = self.block2_lif(x)
        x = seq_apply(lambda z: self.block2_bn(self.block2_conv(z)), x)

        x = self.block3_lif(x)
        x = seq_apply(lambda z: self.block3_bn(self.block3_conv(self.block3_mp(z))), x)

        x = self.block4_lif(x)
        x = seq_apply(lambda z: self.block4_bn(self.block4_conv(self.block4_mp(z))), x)

        return x, (x.shape[-2], x.shape[-1])  # [T, B, C, H', W'], (H', W')
