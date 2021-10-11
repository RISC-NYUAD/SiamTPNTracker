from .base_actor import BaseActor
from .tpn import SiamTPNActor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss

def build_actor(cfg, net):
    if cfg.TRAIN.ACTOR=="TPNACTOR":
        objective = {'giou': giou_loss, 'l1': l1_loss, 'ce': CrossEntropyLoss(), 'center': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'ce': cfg.TRAIN.CLS_WEIGHT, 'center':cfg.TRAIN.CENTER_WEIGHT}
        actor = SiamTPNActor(net=net, objective=objective, loss_weight=loss_weight, cfg=cfg)
        return actor          

    raise RuntimeError(F"Actor not implemented")

