import argparse
import cPickle as pickle
import sys

from config import parse_config_file
from train import train


def parse_args():

    parser = argparse.ArgumentParser(description='Train the multibox detection system')

    parser.add_argument('--tfrecords', dest='tfrecords',
                        help='paths to tfrecords files that contain the training data', type=str,
                        nargs='+', required=True)
    
    parser.add_argument('--priors', dest='priors',
                          help='path to the bounding box priors pickle file', type=str,
                          required=True)
    
    parser.add_argument('--logdir', dest='logdir',
                          help='path to directory to store summary files and checkpoint files', type=str,
                          required=True)

    parser.add_argument('--config', dest='config_file',
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
    # Modify the configuration hyper-parameters for prep purposes.
    cfg.PHASE = "TRAIN"
    cfg.USE_BATCH_STATISTICS = True

    print "Configurations:"
    print cfg
    
    with open(args.priors) as f:
        bbox_priors = pickle.load(f)
    
    train(
      tfrecords=args.tfrecords,
      bbox_priors=bbox_priors,
      logdir=args.logdir,
      cfg=cfg,
      first_iteration=args.first_iteration
    )
  
  