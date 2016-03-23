import argparse
import pickle
import sys
import time
import subprocess

from config import parse_config_file
import prep_for_eval
import test

def parse_args():

    parser = argparse.ArgumentParser(description='Compute the population statistics for the variance and mean')

    parser.add_argument('--train_records', dest='train_tfrecords',
                        help='paths to training tfrecords files', type=str,
                        nargs='+', required=True)

    parser.add_argument('--test_records', dest='test_tfrecords',
                        help='paths to testing tfrecords files', type=str,
                        nargs='+', required=True)

    parser.add_argument('--training_checkpoint_dir', dest='training_checkpoint_dir',
                          help='path to directory where the training checkpoint files are stored', type=str,
                          required=True)

    parser.add_argument('--preped_checkpoint_dir', dest='preped_checkpoint_dir',
                          help='path to directory for storing the evaluation model', type=str,
                          required=True)

    parser.add_argument('--summary_dir', dest='summary_dir',
                          help='path to directory to store evalutation summary files', type=str,
                          required=False, default=None)

    parser.add_argument('--config', dest='config_file',
                          help='Path to the configuration file',
                          required=True, type=str)

    parser.add_argument('--cuda_visible_devices', dest='cuda_visible_devices',
                          help='Specify the visible gpu.',
                          required=False, type=int, default=1)


    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

# GVH: For GPU restriction use CUDA_VISIBLE_DEVICES=0 python start_test ....

if __name__ == '__main__':

    args = parse_args()
    print "Called with:"
    print args

    cfg = parse_config_file(args.config_file)
    print "Configurations:"
    print cfg

    while True:

      if prep_for_eval.check_for_new_training_model(
        training_checkpoint_dir=args.training_checkpoint_dir,
        eval_checkpoint_dir = args.preped_checkpoint_dir):

        print "#####################"
        print "Running eval loop..."
        print "Prepping..."

        prep_cmd = """CUDA_VISIBLE_DEVICES=%d python v3/start_prep.py \
        --tfrecords %s \
        --training_checkpoint_dir %s \
        --preped_checkpoint_dir %s \
        --config %s"""
        filled_cmd = prep_cmd % (args.cuda_visible_devices, " ".join(args.train_tfrecords), args.training_checkpoint_dir, args.preped_checkpoint_dir, args.config_file)
        subprocess.call(filled_cmd, shell=True)

        print "Evaluating..."

        eval_cmd = """CUDA_VISIBLE_DEVICES=%d python v3/start_test.py \
        --tfrecords %s \
        --checkpoint_dir %s \
        --config %s"""
        if args.summary_dir:
          eval_cmd += " --summary_dir %s"
          filled_cmd = eval_cmd % (args.cuda_visible_devices, " ".join(args.test_tfrecords), args.preped_checkpoint_dir, args.config_file, args.summary_dir)
        else:
          filled_cmd = eval_cmd % (args.cuda_visible_devices, " ".join(args.test_tfrecords), args.preped_checkpoint_dir, args.config_file)
        subprocess.call(filled_cmd, shell=True)

      else:
        print "No new model, sleeping"
        time.sleep(60*5)
