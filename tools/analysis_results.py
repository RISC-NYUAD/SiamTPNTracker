
from PIL.Image import merge
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]
import sys
import argparse
sys.path.append('./')

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
parser.add_argument('tracker_param', type=str, help='Name of config file.')
parser.add_argument('--runid', type=int, default=None, help='The run id.', nargs='+')
parser.add_argument('--dataset_name', type=str, default='got10k_val', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
parser.add_argument('--epoch', type=int, default=100, help='epoch')


args = parser.parse_args()

trackers = []

params_dict = {'checkpoint': args.epoch}
params_dict['windows_factor'] = 0.5
params_dict['interval'] = 25
params_dict['debug'] = 0
params_dict['cpu'] = 0

trackers.extend(trackerlist(name=args.tracker_name, parameter_name=args.tracker_param, dataset_name=args.dataset_name,
                            run_ids=args.runid, display_name='Track', params_dict=params_dict))

if "got10k" in args.dataset_name:
    report_name = 'got10k'
else:
    report_name = args.dataset_name
merge_results=False
dataset = get_dataset(args.dataset_name)
plot_results(trackers, dataset, report_name, merge_results=merge_results, plot_types=('success', 'norm_prec'),
             skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
print_results(trackers, dataset, report_name, merge_results=merge_results, plot_types=('success', 'prec', 'norm_prec'))
print_per_sequence_results(trackers, dataset,report_name,merge_results=merge_results)
