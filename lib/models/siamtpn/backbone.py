"""
Backbone modules.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from lib.utils.misc import NestedTensor, is_main_process
import lib.models.siamtpn.shufflenet_v2 as shuffle_module
from .util import FrozenBatchNorm2d


class Backbone_shufflenet(nn.Module):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, model_size: str, 
                out_stages = (3), 
                freeze_bn: bool = None, 
                frozen_layers=('conv1'),):
        norm_layer = FrozenBatchNorm2d if freeze_bn else nn.BatchNorm2d
        backbone = shuffle_module.ShuffleNetV2(
            model_size,
            out_stages=out_stages,
            pretrain=is_main_process(),
            norm_layer = norm_layer)
        super().__init__()
        self.out_stages = out_stages
        self.num_channels = (backbone._stage_out_channels[i] for i in out_stages)
        self.body = backbone
        self.frozen_layers = frozen_layers
        for layer in self.frozen_layers:
            for p in getattr(backbone, layer).parameters():
                p.requires_grad_(False)

    def forward(self, x):
        return self.body(x)


def build_backbone(cfg):
    if 'shufflenet' in cfg.MODEL.BACKBONE.TYPE:
        model = Backbone_shufflenet(
                        cfg.MODEL.BACKBONE.MODEL_SIZE,
                        cfg.MODEL.BACKBONE.OUTPUT_STAGES,
                        cfg.TRAIN.FREEZE_BACKBONE_BN, 
                        cfg.TRAIN.FREEZE_LAYERS
        )
    return model
