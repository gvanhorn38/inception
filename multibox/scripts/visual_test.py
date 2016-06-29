import visual_test

import numpy as np

from priors import generate_priors

prior_bboxes = generate_priors()

print "Number of Priors: %d" % (len(prior_bboxes),)

import sys
sys.path.append('..')
from v3.config import parse_config_file

# Birds
# tfrecord_path = '/home/gvanhorn/Desktop/detection_trial_2/nabirds_detection_test.tfrecords'
# cfg_path = "/home/gvanhorn/Desktop/detection_trial_2/config.yaml"
# logdir = "/home/gvanhorn/Desktop/detection_trial_2"

# Coco 
tfrecord_path = '/home/gvanhorn/Desktop/coco_detection_1/test.tfrecords'
cfg_path = "/home/gvanhorn/Desktop/coco_detection_1/config.yaml"
logdir = "/home/gvanhorn/Desktop/coco_detection_1"

cfg = parse_config_file(cfg_path)
cfg.AUGMENT_IMAGE = False
cfg.USE_BATCH_STATISTICS = False
cfg.BATCH_SIZE = 1

visual_test.test([tfrecord_path], prior_bboxes, logdir + "/checkpoints", None, cfg)