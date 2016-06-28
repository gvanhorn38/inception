import train

import numpy as np

from priors import generate_priors

prior_bboxes = generate_priors()

print "Number of Priors: %d" % (len(prior_bboxes),)

tfrecord_path = '/home/gvanhorn/Desktop/detection_trial_2/nabirds_detection_train.tfrecords'

import sys
sys.path.append('..')
from v3.config import parse_config_file

cfg = parse_config_file("/home/gvanhorn/Desktop/detection_trial_2/config.yaml")
cfg.USE_BATCH_STATISTICS = True

train.train([tfrecord_path], prior_bboxes, "/home/gvanhorn/Desktop/detection_trial_2", cfg, first_iteration=True)