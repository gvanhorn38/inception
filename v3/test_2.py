import cPickle as pickle
import os
import sys
import time


import numpy as np
import tensorflow as tf

import v3
from inception.inputs.construct import construct_network_input_nodes


def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def test(tfrecords, checkpoint_dir, specific_model_path, cfg, summary_dir=None, save_classification_results=False, save_logits=False, max_iterations=None):

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
  images, labels_sparse, image_paths, instance_ids = construct_network_input_nodes(tfrecords,
    input_type=cfg.INPUT_TYPE,
    num_epochs=1,
    batch_size=cfg.BATCH_SIZE,
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

  

  fetches = [top_k_op]
  if save_classification_results :
    fetches += [image_paths, predicted_classes, logits, instance_ids, labels_sparse]

  # keep a reference around
  classification_writer = None

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
      
      # Save the actually classifications for later processing
      if save_classification_results: 
        # create a writer for storing the tfrecords
        output_path = os.path.join(summary_dir, 'classification_results-%d.tfrecords' % (global_step,))
        classification_writer = tf.python_io.TFRecordWriter(output_path)
      
      true_count = 0.0  # Counts the number of correct predictions.
      total_sample_count = 0
      step = 0
      while not coord.should_stop():
        
        if max_iterations != None and step > max_iterations:
          break
        
        t = time.time()
        outputs = sess.run(fetches)
        dt = time.time()-t

        if save_classification_results:
          batch_image_paths = outputs[1]
          batch_pred_class_ids = outputs[2].astype(int)
          batch_logits = outputs[3].astype(float)
          batch_instance_ids = outputs[4]
          batch_gt_class_ids = outputs[5].astype(int)
          
          for i in range(cfg.BATCH_SIZE):
            
            feature={}
            feature['label'] = _int64_feature([batch_gt_class_ids[i]])
            feature['path'] = _bytes_feature([batch_image_paths[i]])
            feature['instance_id'] = _bytes_feature([batch_instance_ids[i]])
            feature['logits'] = _float_feature(batch_logits[i])
            feature['pred_label'] = _int64_feature(batch_pred_class_ids[i])
            
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            classification_writer.write(example.SerializeToString())
            

        predictions = outputs[0]
        true_count += np.sum(predictions)

        print print_str % (
          step,
          true_count,
          (step + 1) * cfg.BATCH_SIZE,
          true_count / ((step + 1.) * cfg.BATCH_SIZE),
          dt/cfg.BATCH_SIZE*1000
        )

        step += 1
        total_sample_count += cfg.BATCH_SIZE

    except tf.errors.OutOfRangeError as e:
      pass

    #except Exception as e:
    #  print e
    #  coord.request_stop(e)

    if save_classification_results:
      # close the writer
      classification_writer.close()

    # Compute precision @ 1.
    precision = true_count / total_sample_count
    print('Model %d: precision @ 1 = %.3f' % (global_step, precision))
    
    # keep a conveinence file with the precision
    precision_file = os.path.join(summary_dir, "precision_summary.txt")
    with open(precision_file, 'a') as f:
      print >> f, 'Model %d: precision @ 1 = %.3f' % (global_step, precision)
    
    if summary_writer != None:
      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)

  coord.request_stop()
  coord.join(threads)

  if summary_writer != None:
    summary_writer.flush()
    summary_writer.close()
