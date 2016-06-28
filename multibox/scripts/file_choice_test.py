import file_choice_test

import numpy as np

from priors import generate_priors

prior_bboxes = generate_priors()

print "Number of Priors: %d" % (len(prior_bboxes),)

tfrecord_path = '/home/gvanhorn/Desktop/detection_trial/nabirds_detection_test.tfrecords'

import sys
sys.path.append('..')
from v3.config import parse_config_file

cfg = parse_config_file("/home/gvanhorn/Desktop/detection_trial/config.yaml")
cfg.AUGMENT_IMAGE = False
cfg.USE_BATCH_STATISTICS = False
cfg.BATCH_SIZE = 1

file_choice_test.test(prior_bboxes, "/home/gvanhorn/Desktop/detection_trial/checkpoints", None, cfg)