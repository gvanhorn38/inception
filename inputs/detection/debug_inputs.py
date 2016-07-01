"""
Visualize the inputs to the network. 
"""

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import sys

from config import parse_config_file
import inputs

def debug(tfrecord_path, config_path = None):
  
  tfrecords = [tfrecord_path]
  cfg = parse_config_file(config_path)
  
  graph = tf.Graph()
  sess = tf.Session(graph = graph)
  
  # run a session to look at the images...
  with sess.as_default(), graph.as_default():

    # Input Nodes
    images, batched_bboxes, batched_num_bboxes, image_ids = inputs.input_nodes(
      tfrecords=tfrecords,
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
    
    plt.ion()

    tf.initialize_all_variables().run()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    done = False
    while not done:
      
      output = sess.run([images, batched_bboxes])
      for image, bboxes in zip(output[0], output[1]):
          
          plt.imshow((image * cfg.IMAGE_STD + cfg.IMAGE_MEAN).astype(np.uint8))
          
          # plot the ground truth bounding boxes
          for bbox in bboxes:
            xmin, ymin, xmax, ymax = bbox * cfg.INPUT_SIZE
            plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'b-')
          
          plt.show(block=False)
          
          t = raw_input("push button")
          if t != '':
            done = True
          plt.clf()
          
if __name__ == '__main__':
  debug(sys.argv[1], sys.argv[2])