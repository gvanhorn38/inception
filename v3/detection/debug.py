from easydict import EasyDict
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import sys

from inception.v3.config import parse_config_file
import inputs


def debug(tfrecord_path, prior_bboxes, config_path = None):

  graph = tf.Graph()

  sess = tf.Session(graph = graph)
  
  # run a session to look at the images...
  with sess.as_default(), graph.as_default():

    tfrecords = [tfrecord_path]

    if config_path == None:
      cfg = EasyDict({
        'BATCH_SIZE' : 32,
        'INPUT_SIZE' : 299,
        'IMAGE_MEAN' : 128,
        'IMAGE_STD' : 128,
        'MAINTAIN_ASPECT_RATIO' : True,
        'NUM_INPUT_THREADS' : 2,
        'AUGMENT_IMAGE' : False,
        
        'RANDOM_FLIP' : True,
        'RANDOM_BBOX_SHIFT' : True,
        'MAX_BBOX_COORD_SHIFT' : 10,
        'RANDOM_CROP' : True
      })
    else:
      cfg = parse_config_file(config_path)
    # Input Nodes
    images, batched_bboxes, batched_num_bboxes, paths = inputs.input_nodes(
      tfrecords=tfrecords,
      bbox_priors = prior_bboxes,
      max_num_bboxes = cfg.MAX_NUM_BBOXES,
      num_epochs=None,
      batch_size=cfg.BATCH_SIZE,
      num_threads=cfg.NUM_INPUT_THREADS,
      add_summaries = True,
      augment=cfg.AUGMENT_IMAGE,
      shuffle_batch=False,
      cfg=cfg
    )
    
    
    coord = tf.train.Coordinator()
  

    tf.initialize_all_variables().run()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    done = False
    while not done:
      
      output = sess.run([images, batched_bboxes])
      for image, bboxes in zip(output[0], output[1]):

          plt.imshow((image * cfg.IMAGE_STD + cfg.IMAGE_MEAN).astype(np.uint8))
          
          # plot the ground truth bounding box
          for bbox in bboxes:
            xmin, ymin, xmax, ymax = bbox * cfg.INPUT_SIZE
            plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'b-')
          
          plt.show()
          t = raw_input("push button")
          if t != '':
            done = True

if __name__ == '__main__':
  debug(sys.argv[1], sys.argv[2])