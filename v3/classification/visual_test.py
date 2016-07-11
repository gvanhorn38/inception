from matplotlib import pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf
import time

import v3.classification.model as model
from inputs.classification.construct import construct_network_input_nodes
from network_utils import add_logits 
from proto_utils import _int64_feature, _float_feature, _bytes_feature

def test(tfrecords, checkpoint_dir, specific_model_path, cfg):
  """
  Test an Inception V3 network.
  
  Args:
    tfrecords (List): an array of file paths to tfrecord protocol buffer files.
    checkpoint_dir (str): a file path to a directory containing model checkpoint files. The 
      newest model will be used.
    specific_model_path (str): a file path to a specific model file. This argument has 
      precedence over `checkpoint_dir`.
    cfg: an EasyDict of configuration parameters.
  """  
  
  graph = tf.get_default_graph()

  sess_config = tf.ConfigProto(
    log_device_placement=False,
    allow_soft_placement = True,
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=cfg.SESSION_CONFIG.PER_PROCESS_GPU_MEMORY_FRACTION
    )
  )

  # Input Nodes
  images, labels_sparse, instance_ids = construct_network_input_nodes(tfrecords,
    input_type=cfg.INPUT_TYPE,
    num_epochs=1, # Go through all of the records once.
    batch_size=cfg.BATCH_SIZE, # Set to 1? 
    num_threads=cfg.NUM_INPUT_THREADS,
    add_summaries = False,
    augment=cfg.AUGMENT_IMAGE,
    shuffle_batch=False,
    cfg=cfg
  )

  # Inference Nodes
  features = model.build(graph, images, cfg=cfg)

  logits = add_logits(graph, features, cfg.NUM_CLASSES)

  class_scores, predicted_classes = tf.nn.top_k(logits, k=5)

  coord = tf.train.Coordinator()

  # Restore the moving average variables for the conv filters, beta and gamma for
  # batch normalization and the softmax params
  ema = tf.train.ExponentialMovingAverage(decay=cfg.MOVING_AVERAGE_DECAY)
  shadow_vars = {
    ema.average_name(var) : var
    for var in graph.get_collection('conv_params')
  }
  shadow_vars.update({
    ema.average_name(var) : var
    for var in graph.get_collection('batchnorm_params')
  })
  shadow_vars.update({
    ema.average_name(var) : var
    for var in graph.get_collection('softmax_params')
  })
  shadow_vars.update({
    ema.average_name(var) : var
    for var in graph.get_collection('batchnorm_mean_var')
  })

  # Restore the parameters
  saver = tf.train.Saver(shadow_vars, reshape=True)

  fetches = [labels_sparse, predicted_classes, images]

  # Restore a checkpoint file
  with tf.Session(config=sess_config) as sess:

    tf.initialize_all_variables().run()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:

      if specific_model_path == None:
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
          specific_model_path = ckpt.model_checkpoint_path
        else:
          print('No checkpoint file found')
          return

      # Restores from checkpoint
      saver.restore(sess, specific_model_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = int(specific_model_path.split('/')[-1].split('-')[-1])
      print "Found model for global step: %d" % (global_step,)
      
      plt.ion()
      done = False
      while not coord.should_stop() and not done:
        
        t = time.time()
        outputs = sess.run(fetches)
        dt = time.time()-t
        
        gt_labels = outputs[0]
        predicted_labels = outputs[1]
        imgs = outputs[2]
        
        for b in range(cfg.BATCH_SIZE):
          gt_label = gt_labels[b]
          pred_labels = predicted_labels[b]
          image = imgs[b]
        
          plt.imshow((image * cfg.IMAGE_STD + cfg.IMAGE_MEAN).astype(np.uint8))
          plt.axis('off')
          plt.title("GT Class: %d\nPred Class: %d" % (gt_label,pred_labels[0]))
          plt.show(block=False)
          t = raw_input("push button")
          if t != '':
            done=True
            break # break out of the batch
          plt.clf()

    except tf.errors.OutOfRangeError as e:
      pass

  coord.request_stop()
  coord.join(threads)
