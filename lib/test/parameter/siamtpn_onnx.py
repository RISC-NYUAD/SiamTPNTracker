from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.default import cfg, update_config_from_file


def parameters(yaml_name: str, epoch=100):
    params = TrackerParams()
    save_dir = env_settings().save_dir
    # update default config from yaml file
    print('yaml name', yaml_name)
    yaml_file = './experiments/' + yaml_name + '.yaml'
    update_config_from_file(cfg, yaml_file)
    params.cfg = cfg

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE
    #params.multiobj_mode = 'parallel'

    # Network checkpoint path
    params.checkpoint = None

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params 
