import os
import sys
import argparse

sys.path.append('./')

from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker



def run_tracker(tracker_name, tracker_param, video_name=None, dataset_name='otb', sequence=None, debug=0, params_dict=None):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """
    dataset = get_dataset(dataset_name)
    if sequence is not None:
        dataset = [dataset[sequence]]
    tracker = Tracker(tracker_name, tracker_param, dataset_name, run_id=None, params_dict=params_dict)

    tracker.run_video(video_name, optional_box=None, debug=None, visdom_info=None, save_results=False)
 

def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of config file.')
    parser.add_argument('--video_name', type=str, help='the path of video')
    parser.add_argument('--dataset_name', type=str, default='got10k_val', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--windows_factor', type=float, default=0.01)
    parser.add_argument('--cpu', type=int, default=0)

    args = parser.parse_args()

    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence

    params_dict = {'checkpoint': args.epoch}
    params_dict['windows_factor'] = args.windows_factor
    params_dict['debug'] = args.debug
    params_dict['cpu'] = args.cpu

    run_tracker(args.tracker_name, args.tracker_param, args.video_name, args.dataset_name, seq_name, args.debug,
                params_dict=params_dict)


if __name__ == '__main__':
    main()