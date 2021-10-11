import torch.nn as nn
import torch
import torch.nn.functional as F
from ..util import conv
from timm.models.layers import trunc_normal_
import math

class ConvFPN(nn.Module):

    def __init__(self, in_dim, num_blocks=6, pre_conv=None, with_pos=False, sr_ratios=None, **kwargs):
        super().__init__()
        if pre_conv is not None:
            self.bottleneck = nn.Conv2d(in_dim, pre_conv, kernel_size=1)
            in_dim = pre_conv
        self.num_blocks = num_blocks
        self.rpn = nn.ModuleList([conv(in_dim, in_dim) for i in range(num_blocks)])
        self.with_pos = with_pos
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, feat):
        """
        tfeat_l3, tfeat_l4: B, C, 8, 8  B, C2, 4, 4
        sfeat_l3, sfeat_l4: B, C, 16, 16  B, C2, 8, 8
        """
        feat = feat[0]
        if self.bottleneck is not None:
            feat = self.bottleneck(feat)
        if self.num_blocks >0:
            for i in range(self.num_blocks):
                feat = self.rpn[i](feat)

        feat = feat
        return feat