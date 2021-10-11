from numpy.core.fromnumeric import reshape
import torch.nn as nn
import torch
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from .util import xcorr_depthwise, conv



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class PatchMLP(nn.Module):
    def __init__(self, cfg):
        super(PatchMLP, self).__init__()
        self.cfg = cfg

        '''clssification head'''
        self.cls_head = MLP(cfg.MODEL.HEAD.IN_DIM, cfg.MODEL.HEAD.IN_DIM, 2, 2)
        self.cen_head = MLP(cfg.MODEL.HEAD.IN_DIM, cfg.MODEL.HEAD.IN_DIM, 1, 2)
        self.box_head = MLP(cfg.MODEL.HEAD.IN_DIM, cfg.MODEL.HEAD.IN_DIM, 4, 2)     

    def forward(self, feat):
        B, L, C = feat.shape
        pred_cls = self.cls_head(feat)
        pred_cen = self.cen_head(feat)
        pred_box = self.box_head(feat)
        results = {'box': pred_box, 'cls': pred_cls, 'cen': pred_cen}    
        return results
    

class PatchHead(nn.Module):
    def __init__(self, hidden=256):
        super(PatchHead, self).__init__()

        '''clssification head'''
        self.cls_head = nn.Sequential(
                conv(hidden, hidden//2),
                conv(hidden//2, hidden//2),
                nn.Conv2d(hidden//2, 2, kernel_size=1)
        )
        self.box_head = nn.Sequential(
                conv(hidden, hidden//2),
                conv(hidden//2, hidden//2),
                nn.Conv2d(hidden//2, 4, kernel_size=1)
        )    


    def forward(self, feat, kernel):
        feat = xcorr_depthwise(feat, kernel)
        B, C, H, W = feat.shape
        pred_cls = self.cls_head(feat).permute(0, 2, 3, 1).reshape(-1, 2)
        pred_box = self.box_head(feat).permute(0, 2, 3, 1).reshape(B, -1, 4)
        return pred_cls, F.relu(pred_box)
    

def build_head(cfg):
    if cfg.MODEL.HEAD.TYPE == 'MLP':
        return PatchMLP(cfg)
    elif cfg.MODEL.HEAD.TYPE == 'CONV':
        return PatchHead(cfg.MODEL.HEAD.IN_DIM)
    else:
        print('Head not implemented')