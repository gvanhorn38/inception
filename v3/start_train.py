import argparse
import sys

from config import parse_config_file
from train import train

def parse_args():

    parser = argparse.ArgumentParser(description='Finetune the Inception V3 network')

    parser.add_argument('-t', '--tfrecords', dest='tfrecords',
                        help='paths to tfrecords files that contain the training data', type=str,
                        nargs='+', required=True)

    # parser.add_argument('-n', '--num_classes', dest='num_classes',
#                         help='The number of classes.',
#                         required=True, type=int)

    parser.add_argument('--logdir', dest='logdir',
                          help='path to directory to store summary files and checkpoint files', type=str,
                          required=True)

    # GVH: For GPU restriction use CUDA_VISIBLE_DEVICES=0 python start_test ....

    # parser.add_argument('--gpu', dest='gpu_id',
    #                     help='The gpu device number to run computations on',
    #                     default=None, type=int,
    #                     required=False)
    #
    # # Apparently the device filters parameter is ignored by TensorFlow right now.
    # parser.add_argument('--device_filters', dest='device_filters',
    #                      help='device filters to apply to the Session',
    #                      nargs='+', required=False, default=[])

#     parser.add_argument('--iters', dest='max_iters',
#                         help='number of iterations to train',
#                         default=50000, type=int)
#
#     parser.add_argument('--lr', dest='learning_rate',
#                         help='The initial learning rate',
#                         default=1e-4, type=float)

    parser.add_argument('-c', '--config', dest='config_file',
                        help='Path to the configuration file',
                        required=True, type=str)

    parser.add_argument('--first_iteration', dest='first_iteration',
                        help='Is this the first iteration? i.e. should we restore the original inception weights?',
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
  
    print "Configurations:"
    print cfg

    train(
      tfrecords=args.tfrecords,
      logdir=args.logdir,
      cfg=cfg,
      first_iteration=args.first_iteration
    )
