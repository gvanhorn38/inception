import cPickle as pickle
import os
import sys
import time


import numpy as np
import tensorflow as tf

import v3
from inputs.construct import construct_network_input_nodes

def test(tfrecords, checkpoint_dir, specific_model_path, cfg, summary_dir=None, save_classification_results=False, save_logits=False):
  
  graph = tf.get_default_graph()

  sess_config = tf.ConfigProto(
    log_device_placement=False,
    #device_filters = device_filters,
    allow_soft_placement = True,
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=cfg.SESSION_CONFIG.PER_PROCESS_GPU_MEMORY_FRACTION
    )
  )

  # Input Nodes
  images, labels_sparse, image_paths = construct_network_input_nodes(tfrecords,
    input_type=cfg.INPUT_TYPE,
    num_epochs=1,
    batch_size=cfg.TEST_BATCH_SIZE,
    num_threads=cfg.NUM_INPUT_THREADS,
    add_summaries = False,
    augment=False,
    shuffle_batch=False,
    cfg=cfg
  )

  # Inference Nodes
  features = v3.build(graph, images, cfg.NUM_CLASSES, cfg=cfg)

  logits = v3.add_logits(graph, features, cfg.NUM_CLASSES)

  top_k_op = tf.nn.in_top_k(logits, labels_sparse, 1)
  
  if save_classification_results:
    class_scores, predicted_classes = tf.nn.top_k(logits)

  coord = tf.train.Coordinator()

  # The prep phase already switched in the moving average variables, so just restore them
  shadow_vars = {
    var.name[:-2] : var
    for var in graph.get_collection('conv_params')
  }
  shadow_vars.update({
    var.name[:-2] : var
    for var in graph.get_collection('batchnorm_params') # this is the beta and gamma parameters
  })
  shadow_vars.update({
    var.name[:-2] : var
    for var in graph.get_collection('softmax_params') # do we want a moving average on this?
  })
  # The prep phase computed the averages of the mean and variance, so load them in
  ema = tf.train.ExponentialMovingAverage(decay=cfg.MOVING_AVERAGE_DECAY)
  shadow_vars.update({
    ema.average_name(var) : var
    for var in graph.get_collection('batchnorm_mean_var')
  })

  # Restore the parameters
  saver = tf.train.Saver(shadow_vars)

  # Write the classification score to the summary directory if it was provided
  if summary_dir != None:
    if not os.path.exists(summary_dir):
      print "FAILED to find summary directory, it does not exists:"
      print summary_dir
      summary_writer = None
    else:
      summary_op = tf.merge_all_summaries()
      summary_writer = tf.train.SummaryWriter(summary_dir)
  else:
    summary_writer = None

  print_str = ', '.join([
    'Evaluated batch %d',
    'Total Number Correct: %d / %d',
    'Current Precision: %.3f',
    'Time/image (ms): %.1f'
  ])
  
  # Save the actually classifications for later processing
  if save_classification_results:
    image_paths_and_predictions = []
  
  if save_logits:
    saved_logits = None

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

      true_count = 0.0  # Counts the number of correct predictions.
      total_sample_count = 0
      step = 0
      while not coord.should_stop():

        t = time.time()
        outputs = sess.run([image_paths, predicted_classes, logits, top_k_op])
        dt = time.time()-t

        raw_image_paths = outputs[0]
        class_idxs = outputs[1]

        image_paths_and_predictions.extend(zip(raw_image_paths, class_idxs.ravel()))
        if save_logits:
          if saved_logits is None:
            saved_logits = outputs[2]
          else:
            saved_logits = np.vstack((saved_logits, outputs[2]))

        print "Evaluated step: %d" % (step,)

        predictions = outputs[3]

        true_count += np.sum(predictions)

        print print_str % (
          step,
          true_count,
          (step + 1) * cfg.TEST_BATCH_SIZE,
          true_count / ((step + 1.) * cfg.TEST_BATCH_SIZE),
          dt/cfg.TEST_BATCH_SIZE*1000
        )

        step += 1
        total_sample_count += cfg.TEST_BATCH_SIZE

    except tf.errors.OutOfRangeError as e:
      
      if save_classification_results:
        print "Saving classsification results"
        p = os.path.join(summary_dir, "classification_results-%d.pkl" % (global_step,))
        with open(p, 'w') as f:
          pickle.dump(image_paths_and_predictions, f)

      if save_logits:
        print "Saving logits"
        p = os.path.join(summary_dir, "saved_logits-%d.pkl" % (global_step,))
        with open(p, 'w') as f:
          pickle.dump(saved_logits, f)

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('Model %d: precision @ 1 = %.3f' % (global_step, precision))

      if summary_writer != None:
        summary = tf.Summary()
        summary.ParseFromString(sess.run(summary_op))
        summary.value.add(tag='Precision @ 1', simple_value=precision)
        summary_writer.add_summary(summary, global_step)


    except Exception as e:
      coord.request_stop(e)

  coord.request_stop()
  coord.join(threads)

  if summary_writer != None:
    summary_writer.flush()
    summary_writer.close()
