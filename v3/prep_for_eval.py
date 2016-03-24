"""
This code will freeze the learned parameters and pass the training data back through.
The mean and variance of the conv outputs will be computed and stored so that they
can be used for test time.
"""

import os
import time

import numpy as np
import tensorflow as tf

import v3
from inception.inputs.construct import construct_network_input_nodes

# import sys
# sys.path.append("/home/gvanhorn/code/vision_research/eccv_2016_parts/pose_proc")
# sys.path.append("/home/gvanhorn/code/vision_research/eccv_2016_parts/tensorflow_experiments/custom_inception_v3")
# import build_inception_v3
# import inputs

def check_for_new_training_model(training_checkpoint_dir, eval_checkpoint_dir):
  """
  See if there is a new training model to prep
  """
  # Examine the training model
  ckpt = tf.train.get_checkpoint_state(training_checkpoint_dir)
  if ckpt and ckpt.model_checkpoint_path:
    # Assuming model_checkpoint_path looks something like:
    #   /my-favorite-path/cifar10_train/model.ckpt-0,
    # extract global_step from it.
    try:
      training_global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
    except:
      print "couldn't parse: %s" % (ckpt.model_checkpoint_path,)
      print "ignoring..."
      return False
    # Examine the evaluation model
    ckpt = tf.train.get_checkpoint_state(eval_checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      eval_global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])

      if eval_global_step < training_global_step:
        return True
      else:
        return False
    else:
      return True
  else:
    return False

def prep(tfrecords, checkpoint_dir, save_dir, cfg):

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
    num_epochs=None,
    batch_size=cfg.BATCH_SIZE,
    num_threads=cfg.NUM_INPUT_THREADS,
    add_summaries = False,
    augment=False,
    shuffle_batch=True,
    cfg=cfg
  )

  # Inference Nodes
  features = v3.build(graph, images, cfg.NUM_CLASSES, cfg)
  logits = v3.add_logits(graph, features, cfg.NUM_CLASSES)


  # images, labels_sparse, image_paths = inputs.construct_network_input_nodes(tfrecords,
  #   input_type=cfg.INPUT_TYPE,
  #   num_epochs=None,
  #   batch_size=cfg.BATCH_SIZE,
  #   num_threads=cfg.NUM_INPUT_THREADS,
  #   add_summaries = False,
  #   augment=False,
  #   cfg=cfg
  # )
  #features = build_inception_v3.build(graph, images, cfg.NUM_CLASSES)
  #logits = build_inception_v3.add_logits(graph, features, cfg.NUM_CLASSES)


  top_k_op = tf.nn.in_top_k(logits, labels_sparse, 1)

  class_scores, predicted_classes = tf.nn.top_k(logits)

  global_step = tf.Variable(0, name='global_step', trainable=False)
  global_step_inc = tf.count_up_to(global_step, cfg.NUM_PREP_ITERATIONS, name=None)
  ema = tf.train.ExponentialMovingAverage(decay=cfg.MOVING_AVERAGE_DECAY, num_updates=global_step)

  # Restore the moving average variables for the conv filters, beta and gamma for
  # batch normalization and the softmax params
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
  restore_saver = tf.train.Saver(shadow_vars)

  # We will compute averages for the mean and variance of the conv filter outputs
  maintain_averages_op = ema.apply(graph.get_collection('batchnorm_mean_var'))

  # Create an op that will update the moving averages after each training
  # step.
  with tf.control_dependencies([top_k_op]):
    compute_averages_op = tf.group(maintain_averages_op)

  # make a saver for everything. The testing/eval code will use the variables stored
  # by this Saver
  saver = tf.train.Saver()

  coord = tf.train.Coordinator()

  print_str = ', '.join([
    'Evaluated batch %d',
    'Number correct: %d / %d',
    'Time/image (ms): %.1f'
  ])

  if cfg.DEBUG:
    m = ema.average(graph.get_tensor_by_name('conv/batchnorm/mean:0'))
    v = ema.average(graph.get_tensor_by_name('conv/batchnorm/variance:0'))

  step = 0
  # Restore a checkpoint file
  with tf.Session(config=sess_config) as sess:

    tf.initialize_all_variables().run()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:

      ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

      if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        restore_saver.restore(sess, ckpt.model_checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/cifar10_train/model.ckpt-0,
        # extract global_step from it.
        global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        print "Found model for global step: %d" % (global_step,)
      else:
        print('No checkpoint file found')
        return

      while step < cfg.NUM_PREP_ITERATIONS:
        if coord.should_stop():
            break
        t = time.time()

        if cfg.DEBUG:
          outputs = sess.run([top_k_op, compute_averages_op, m, v])
        else:
          outputs = sess.run([top_k_op, compute_averages_op, predicted_classes, labels_sparse, image_paths])

        global_step_inc.eval()

        dt = time.time()-t

        #for pred, gt, p in zip(outputs[2], outputs[3], outputs[4]):
        #    print "%s\t%s\t%s" % (pred, gt, p)

        predictions = outputs[0]
        true_count = np.sum(predictions)

        print print_str % (
          step,
          true_count, cfg.BATCH_SIZE,
          dt/cfg.BATCH_SIZE*1000
        )

        step += 1

        if cfg.DEBUG:
            m_output = outputs[2]
            v_output = outputs[3]
            print "Mean variable: %s" % m.name
            print m_output
            print "Variance variable: %s" % v.name
            print v_output

    except tf.errors.OutOfRangeError as e:
      pass

    except Exception as e:
      coord.request_stop(e)

    if step != 0:

        print "Ran for %d batches" % (step,)

        saver.save(
          sess=sess,
          save_path= os.path.join(save_dir, 'ft_inception_v3'),
          global_step=global_step
        )

  try:
    coord.request_stop()
    coord.join(threads)
  except tf.errors.CancelledError as e:
    pass
