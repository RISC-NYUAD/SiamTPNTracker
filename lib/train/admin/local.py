
class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = './results/'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.lasot_dir = '/home/daitao/data2/Lasot/LaSOTBenchmark/'
        self.got10k_dir = '/home/daitao/work/data/got10k/train/'
        self.trackingnet_dir = '/home/daitao/data/trackingnet/data/'
        self.coco_dir = '/home/daitao/data/coco/'
  

                

