from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    settings.got10k_path = '/home/ardi/Desktop/Dataset/GOT-10k/got-10k'
    settings.save_dir = '/home/ardi/Desktop/project/SiamTPNTracker/results/'
    # settings.got_packed_results_path = './results/'
    # settings.got_reports_path = './results/'
    # settings.lasot_path = ''
    # settings.otb_path = ''
    # settings.result_plot_path = './results/result_plots/'
    # settings.results_path = './results/tracking_results' 
    # settings.trackingnet_path = ''
    # settings.uav_path =''
    # settings.vot_path = ''

    return settings

