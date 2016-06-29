import train

import numpy as np

from priors import generate_priors

prior_bboxes = generate_priors()

print "Number of Priors: %d" % (len(prior_bboxes),)

# Birds
# tfrecord_path = '/home/gvanhorn/Desktop/detection_trial_2/nabirds_detection_train.tfrecords'
# cfg_path = "/home/gvanhorn/Desktop/detection_trial_2/config.yaml"
# logdir = "/home/gvanhorn/Desktop/detection_trial_2"

# Coco
tfrecord_path = '/home/gvanhorn/Desktop/coco_detection_1/train.tfrecords'
cfg_path = "/home/gvanhorn/Desktop/coco_detection_1/config.yaml"
logdir = "/home/gvanhorn/Desktop/coco_detection_1"


import sys
sys.path.append('..')
from v3.config import parse_config_file

cfg = parse_config_file(cfg_path)
cfg.USE_BATCH_STATISTICS = True

train.train([tfrecord_path], prior_bboxes, logdir, cfg, first_iteration=False)