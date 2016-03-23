from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import sys

from inception.v3.config import parse_config_file 
from construct import construct_network_input_nodes


def debug(tfrecord_path, config_path):

  graph = tf.get_default_graph()

  tfrecords = [tfrecord_path]

  cfg = parse_config_file(config_path)

  # Input Nodes
  images, labels_sparse, paths = construct_network_input_nodes(
    tfrecords=tfrecords,
    input_type=cfg.INPUT_TYPE,
    num_epochs=None,
    batch_size=cfg.BATCH_SIZE,
    num_threads=cfg.NUM_INPUT_THREADS,
    add_summaries = True,
    augment=False,
    shuffle_batch=False, 
    cfg=cfg
  )

  coord = tf.train.Coordinator()

  # run a session to look at the images...
  with tf.Session() as sess:

      tf.initialize_all_variables().run()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
      while True: 
        output = sess.run([images, labels_sparse])
        for image, label in zip(output[0], output[1]):
            
            plt.imshow((image * cfg.IMAGE_STD + cfg.IMAGE_MEAN).astype(np.uint8))
            plt.title("Class: %d" % (label,))
            plt.show()
            t = raw_input("push button")
            if t != '':
              return
            
if __name__ == '__main__':
  debug(sys.argv[1], sys.argv[2])