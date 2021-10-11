from easydict import EasyDict as edict
from numpy.core.numeric import False_
import yaml

"""
Add default.
"""
cfg = edict()

# MODEL
cfg.MODEL = edict()
cfg.MODEL.WITH_POS = True
cfg.MODEL.HIDDEN_DIM = [256, 512]
cfg.MODEL.POSITION_EMBEDDING = 'sine'  # sine or learned
# MODEL.BACKBONE
cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.TYPE = "resnet50"  # resnet50, resnext101_32x8d
cfg.MODEL.BACKBONE.OUTPUT_STAGES = [3, 4]
cfg.MODEL.BACKBONE.MODEL_SIZE = "1.0x"
# MODEL.ANCHOR
cfg.MODEL.ANCHOR = edict()
cfg.MODEL.ANCHOR.NUM=20
cfg.MODEL.ANCHOR.FACTOR=1 
cfg.MODEL.ANCHOR.BIAS=0.5
# MODEL.FPN
cfg.MODEL.FPN = edict()
cfg.MODEL.FPN.TYPE = "TREFPN"
cfg.MODEL.FPN.PRE_CONV = [256, 256]
cfg.MODEL.FPN.WITH_POS = False
cfg.MODEL.FPN.NBLOCKS = 2
cfg.MODEL.FPN.NHEADS = 4
cfg.MODEL.FPN.MLP_RATIOS = 4
cfg.MODEL.FPN.QKV_BIAS = True
cfg.MODEL.FPN.R_STRIDES = [2,2,1,1]

# MODEL.HEAD
cfg.MODEL.HEAD = edict()
cfg.MODEL.HEAD.TYPE = ""
cfg.MODEL.HEAD.IN_DIM = 256

# MODEL.NECK
cfg.MODEL.NECK = edict()
cfg.MODEL.NECK.TYPE = "POOLSHUFFLE"
cfg.MODEL.NECK.FILTER_SIZE = [4, 2]
cfg.MODEL.NECK.STRIDE = [16, 32]
cfg.MODEL.NECK.POOL_SQUARE = False
cfg.MODEL.NECK.FILTER_NORM = True 


# MODEL.OUTPUT
cfg.MODEL.OUTPUT = edict()
cfg.MODEL.OUTPUT.WITH_ANCHORS = True

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 500
cfg.TRAIN.LR_DROP_EPOCH = 400
cfg.TRAIN.BATCH_SIZE = 16
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
cfg.TRAIN.FREEZE_BACKBONE_BN = True
cfg.TRAIN.FREEZE_LAYERS = ['conv1', 'layer1']
cfg.TRAIN.PRINT_INTERVAL = 50
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.ACTOR = ""
cfg.TRAIN.SEG_WEIGHT = 1.
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.CLS_WEIGHT = 2.0
cfg.TRAIN.CENTER_WEIGHT = 1.0

# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1

# DATA
cfg.DATA = edict()
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT", "GOT10K_vottrain"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000
# DATA.SEARCH
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.SIZE = 320
cfg.DATA.SEARCH.FACTOR = 5.0
cfg.DATA.SEARCH.CENTER_JITTER = 4.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5
cfg.DATA.SEARCH.FEATSIZE = 20
cfg.DATA.SEARCH.NUMBER = 1
# DATA.TEMPLATE
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.SIZE = 320
cfg.DATA.TEMPLATE.FACTOR = 5.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.FEATSIZE = 20

# TEST
cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 5.0
cfg.TEST.TEMPLATE_SIZE = 320
cfg.TEST.SEARCH_FACTOR = 5.0
cfg.TEST.SEARCH_SIZE = 320
cfg.TEST.EPOCH = 500
cfg.TEST.HANNING_FACTOR = 0
cfg.TEST.PENALTY_K=0.04
cfg.TEST.WINDOW_INFLUENCE=0.44
cfg.TEST.LR=0.33
cfg.TEST.UPDATE_INTERVAL=25


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def update_config_from_file(cfg, filename):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        _update_config(cfg, exp_config)


