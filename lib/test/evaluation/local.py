from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    settings.got10k_path = '/home/daitao/work/data/got10k/'
    settings.save_dir = './results/'
    settings.got_packed_results_path = './results/'
    settings.got_reports_path = './results/'
    settings.lasot_path = '/home/daitao/data2/Lasot/LaSOTBenchmark'
    settings.otb_path = '/home/daitao/work/track/dataset/OTB100/'
    settings.result_plot_path = './results/result_plots/'
    settings.results_path = './results/tracking_results'    # Where to store tracking results
    settings.trackingnet_path = '/home/daitao/data/trackingnet/data/'
    settings.uav_path = '/home/daitao/work/data/UAV123/'
    settings.vot_path = '/home/daitao/work/track/dataset/VOT2018/'

    return settings

