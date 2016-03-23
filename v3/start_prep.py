import argparse
import pickle
import sys

import prep_for_eval

from config import parse_config_file

def parse_args():

    parser = argparse.ArgumentParser(description='Compute the population statistics for the variance and mean')

    parser.add_argument('-t', '--tfrecords', dest='tfrecords',
                        help='paths to tfrecords files', type=str,
                        nargs='+', required=True)

#     parser.add_argument('-n', '--num_classes', dest='num_classes',
#                         help='The number of classes.',
#                         required=True, type=int)

#     parser.add_argument('-b', '--num_steps', dest='num_steps',
#                         help='The number of steps to take. (the number of batches to run)',
#                         required=True, type=int)

    # parser.add_argument('--input', dest='input',
    #                     help='path to a pickle file with the image and label data', type=str,
    #                     required=True)

    parser.add_argument('-d', '--chkdir', dest='checkpoint_dir',
                          help='path to directory where the checkpoint files are stored', type=str,
                          required=True)

    parser.add_argument('-s', '--savedir', dest='save_dir',
                          help='path to directory for storing the model', type=str,
                          required=True)

    # GVH: For GPU restriction use CUDA_VISIBLE_DEVICES=0 python start_test ....

    # parser.add_argument('--gpu', dest='gpu_id',
    #                     help='The gpu device number to run computations on',
    #                     default=None, type=int,
    #                     required=False)

    # Apparently the device filters parameter is ignored by TensorFlow right now.
    # parser.add_argument('--device_filters', dest='device_filters',
    #                     help='device filters to apply to the Session',
    #                     nargs='+', required=False, default=[])


    parser.add_argument('-c', '--config', dest='config_file',
                    help='Path to the configuration file',
                    required=True, type=str)

    parser.add_argument('--debug', dest='debug',
                        help='print info about the averages on each run',
                        action='store_true')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    print "Called with:"
    print args

    cfg = parse_config_file(args.config_file)
    cfg.PHASE = "TRAIN"
    if args.debug:
      cfg.DEBUG = True
    else:
      cfg.DEBUG = False

    print "Configurations:"
    print cfg

    prep_for_eval.prep(
      tfrecords=args.tfrecords,
      checkpoint_dir=args.checkpoint_dir,
      save_dir = args.save_dir,
      cfg=cfg
    )
