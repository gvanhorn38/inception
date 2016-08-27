import argparse
import pickle
import sys

from config import parse_config_file
import test
import dense_test

# GVH: For GPU restriction use CUDA_VISIBLE_DEVICES=0 python start_test ....

def parse_args():

    parser = argparse.ArgumentParser(description='Test an Inception V3 network')

    parser.add_argument('--tfrecords', dest='tfrecords',
                        help='paths to tfrecords files', type=str,
                        nargs='+', required=True)
    
    parser.add_argument('--priors', dest='priors',
                          help='path to the bounding box priors pickle file', type=str,
                          required=True)
    
    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir',
                          help='path to directory where the checkpoint files are stored. The latest model will be tested against.', type=str,
                          required=False, default=None)
    
    parser.add_argument('--model', dest='specific_model',
                          help='path to a specific model to test against. This has precedence over the checkpoint_dir argument.', type=str,
                          required=False, default=None)

    parser.add_argument('--summary_dir', dest='summary_dir',
                          help='path to directory to store summary files', type=str,
                          required=False, default=None)

    parser.add_argument('--config', dest='config_file',
                        help='Path to the configuration file',
                        required=True, type=str)
    
    parser.add_argument('--save_classification_results', dest='save_classification_results',
                        help='For each image, store the class that it was classified as',
                        action='store_true', default=False)
    
    parser.add_argument('--max_iterations', dest='max_iterations',
                        help='Maximum number of iterations to run',
                        required=False, type=int, default=None)
    
    parser.add_argument('--max_detections', dest='max_detections',
                        help='Maximum number of detection to store per image',
                        required=False, type=int, default=100) 
    
    parser.add_argument('--save_dir', dest='save_dir',
                        help='Directory to save the json result file.',
                        required=True, type=str)
    
    parser.add_argument('--dense', dest='dense',
                        help='For each image, extract and process crops from the image.',
                        action='store_true', default=False)                

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
    cfg.PHASE = "TEST"
    cfg.USE_BATCH_STATISTICS = False

    print "Configurations:"
    print cfg
    
    with open(args.priors) as f:
      bbox_priors = pickle.load(f)
    
    test_func = test.test
    if args.dense:
      test_func = dense_test.test
    
    test_func(
      tfrecords=args.tfrecords,
      bbox_priors=bbox_priors,
      checkpoint_dir=args.checkpoint_dir,
      specific_model_path = args.specific_model,
      save_dir = args.save_dir,
      max_detections = args.max_detections,
      cfg=cfg,
      #summary_dir = args.summary_dir,
      #save_classification_results=args.save_classification_results,
      #max_iterations = args.max_iterations
    )
