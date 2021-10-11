# Modified from
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# https://github.com/facebookresearch/LeViT/blob/main/levit.py
# Copyright 2020 Daitao Xing, Apache-2.0 License

import torch.nn as nn
import torch
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import math


class Linear(nn.Module):
    def __init__(self, in_dim, hid_dim=None, act_layer=nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hid_dim, in_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim_q, dim_kv, num_heads=4, qkv_bias=False, stride=1):
        super().__init__()

        self.dim = dim_q
        self.num_heads = num_heads
        head_dim = dim_q // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim_q, dim_q, bias=qkv_bias)
        self.kv = nn.Linear(dim_kv, dim_q * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim_q, dim_q)

        self.stride = stride
        if stride > 1:
            self.pool = nn.AvgPool2d(stride, stride=stride)
            self.sr = nn.Conv2d(dim_kv, dim_kv, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim_kv)
            self.act = nn.GELU()
    def forward(self, x, y):
        B, N, C = x.shape
        B, L, C2 = y.shape
        H = W = int(math.sqrt(L))
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.stride > 1:
            y_ = y.permute(0, 2, 1).contiguous().reshape(B, C2, H, W)
            y_ = self.sr(self.pool(y_)).reshape(B, C2, -1).permute(0, 2, 1)
            y_ = self.norm(y_)
            y_ = self.act(y_)
            kv = self.kv(y_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        else:
            kv = self.kv(y).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class Block(nn.Module):

    def __init__(self, dim_q, dim_kv, cross=True, num_heads=4, mlp_ratio=2., qkv_bias=False,
                act_layer=nn.GELU, norm_layer=nn.LayerNorm, stride=1):
        super().__init__()
        self.norm1 = norm_layer(dim_q)
        self.attn = Attention(
            dim_q, dim_kv,
            num_heads=num_heads, qkv_bias=qkv_bias, stride=stride)
        self.norm2 = norm_layer(dim_q)
        mlp_hidden_dim = int(dim_q * mlp_ratio)
        self.mlp = Linear(in_dim=dim_q, hid_dim=mlp_hidden_dim, act_layer=act_layer)
        if cross:
            self.norm3 = norm_layer(dim_kv)
        self.cross = cross

    def forward(self, x, y):
        if self.cross:
            return self.forward_cross(x, y)
        else:
            return self.forward_self(x)

    def forward_self(self, x):
        norm_x = self.norm1(x)
        x = x + self.attn(norm_x, norm_x)
        x = x + self.mlp(self.norm2(x))

        return x

    def forward_cross(self, x, y):
        x = x + self.attn(self.norm1(x), self.norm3(y))
        x = x + self.mlp(self.norm2(x))

        return x


class ReFPNBlock(nn.Module):

    def __init__(self, dim, **kwargs):
        super().__init__()
        self.block_l4_1 = Block(dim, dim, cross=False, stride=2, **kwargs)
        self.block_l4_2 = Block(dim, dim, cross=False, stride=2, **kwargs)
        self.block_l4_3 = Block(dim, dim, cross=False, stride=2, **kwargs)
        self.block_l5_l4 = Block(dim, dim, stride=1, **kwargs)
        self.block_l3_l4 = Block(dim, dim, stride=4, **kwargs)

    def forward(self, feat_l3, feat_l4, feat_l5):
        
        feat = self.block_l4_1(feat_l4, feat_l4) + \
                self.block_l5_l4(feat_l4, feat_l5) + \
                    self.block_l3_l4(feat_l4, feat_l3)
        
        feat = self.block_l4_1(feat, feat)
        feat = self.block_l4_1(feat, feat)

        return feat


class TReFPN(nn.Module):

    def __init__(self, in_dim,  num_blocks=2, pre_conv=None, **kwargs):
        super().__init__()
        assert pre_conv is not None
        self.num_blocks = num_blocks
        num_l = len(in_dim)
        self.num_layers = num_l
        if pre_conv is not None:
            self.bottleneck = nn.ModuleList([nn.Conv2d(in_dim[i], pre_conv[i], kernel_size=1) for i in range(num_l)])
        hidden_dim = pre_conv[0]
        self.rpn = nn.ModuleList([ReFPNBlock(hidden_dim, **kwargs) for i in range(num_blocks)])

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
        feat = [self.bottleneck[i](feat[i]) for i in range(self.num_layers)]
        feat_l3, feat_l4, feat_l5 = feat    
        B, C, H4, W4 = feat_l4.shape

        feat_l3 = feat_l3.flatten(2).permute(0,2,1)
        feat_l4 = feat_l4.flatten(2).permute(0,2,1)
        feat_l5 = feat_l5.flatten(2).permute(0,2,1)

        for i in range(self.num_blocks):
            feat_l4 = self.rpn[i](feat_l3, feat_l4, feat_l5)
        feat_l4 = feat_l4.permute(0,2,1).reshape(B, C, H4, W4)
        return feat_l4


