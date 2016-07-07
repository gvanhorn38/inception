import argparse
import pickle
import sys

from config import parse_config_file
import file_test

# GVH: For GPU restriction use CUDA_VISIBLE_DEVICES=0 python start_test ....

def parse_args():

    parser = argparse.ArgumentParser(description='Test an Inception V3 network')
    
    parser.add_argument('--priors', dest='priors',
                          help='path to the bounding box priors pickle file', type=str,
                          required=True)
    
    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir',
                          help='path to directory where the checkpoint files are stored. The latest model will be tested against.', type=str,
                          required=False, default=None)
    
    parser.add_argument('--model', dest='specific_model',
                          help='path to a specific model to test against. This has precedence over the checkpoint_dir argument.', type=str,
                          required=False, default=None)

    parser.add_argument('--config', dest='config_file',
                        help='Path to the configuration file',
                        required=True, type=str)

    args = parser.parse_args()
    
    if args.checkpoint_dir == None and args.specific_model == None:
      print "Either a checkpoint directory or a specific model needs to be specified."
      parser.print_help()
      sys.exit(1)
    
    return args

if __name__ == '__main__':

    args = parse_args()
    print "Called with:"
    print args

    cfg = parse_config_file(args.config_file)
    # Modify the configuration hyper-parameters for test purposes. 
    cfg.USE_BATCH_STATISTICS = False

    print "Configurations:"
    print cfg
    
    with open(args.priors) as f:
      bbox_priors = pickle.load(f)

    file_test.test(
      bbox_priors=bbox_priors,
      checkpoint_dir=args.checkpoint_dir,
      specific_model_path = args.specific_model,
      cfg=cfg
    )
