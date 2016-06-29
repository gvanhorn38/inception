from easydict import EasyDict

import debug
from priors import generate_priors

prior_bboxes = generate_priors()

#tfrecord_path = '/home/gvanhorn/Desktop/detection_trial_2/nabirds_detection_train.tfrecords'
tfrecord_path = '/home/gvanhorn/Desktop/coco_detection_1/train.tfrecords'
cfg_path = '/home/gvanhorn/Desktop/coco_detection_1/config.yaml'

debug.debug(tfrecord_path, prior_bboxes, cfg_path)