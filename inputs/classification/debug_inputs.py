from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import sys

from config import parse_config_file
from construct import construct_network_input_nodes


def debug(tfrecord_path, config_path):

  graph = tf.get_default_graph()

  tfrecords = [tfrecord_path]

  cfg = parse_config_file(config_path)

  # Input Nodes
  images, labels_sparse, instance_ids = construct_network_input_nodes(
    tfrecords=tfrecords,
    input_type=cfg.INPUT_TYPE,
    num_epochs=None,
    batch_size=cfg.BATCH_SIZE,
    num_threads=cfg.NUM_INPUT_THREADS,
    add_summaries = False,
    augment=cfg.AUGMENT_IMAGE,
    shuffle_batch=False,
    cfg=cfg
  )

  coord = tf.train.Coordinator()
  
  plt.ion()
  
  # run a session to look at the images...
  with tf.Session() as sess:

      tf.initialize_all_variables().run()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)

      while True:
        output = sess.run([images, labels_sparse, instance_ids])
        for image, label, image_id in zip(output[0], output[1], output[2]):

            plt.imshow((image * cfg.IMAGE_STD + cfg.IMAGE_MEAN).astype(np.uint8))
            plt.title("Class: %d\tImage: %s" % (label,image_id))
            plt.show(block=False)
            t = raw_input("push button")
            if t != '':
              return

if __name__ == '__main__':
  debug(sys.argv[1], sys.argv[2])
