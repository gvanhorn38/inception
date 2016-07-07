import numpy as np
import os
import sys
import tensorflow as tf
import time

import v3.classification.model as model
from inputs.classification.construct import construct_network_input_nodes
from network_utils import add_logits 
from proto_utils import _int64_feature, _float_feature, _bytes_feature

def test(tfrecords, checkpoint_dir, specific_model_path, cfg, summary_dir=None, 
  save_classification_results=False, max_iterations=None):
  """
  Test an Inception V3 network.
  
  Args:
    tfrecords (List): an array of file paths to tfrecord protocol buffer files.
    checkpoint_dir (str): a file path to a directory containing model checkpoint files. The 
      newest model will be used.
    specific_model_path (str): a file path to a specific model file. This argument has 
      precedence over `checkpoint_dir`.
    cfg: an EasyDict of configuration parameters.
    summary_dir (str): if specified, the tensorboard summary files will be writen to this 
      directory.
    save_classification_results (bool): if True, then the code will create a tfrecords 
      file containing the logits and predicted class for each image. 
    max_iterations (int): The maximum number of batches to execute. Leave as None to go
      through all records.
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
    batch_size=cfg.BATCH_SIZE,
    num_threads=cfg.NUM_INPUT_THREADS,
    add_summaries = False,
    augment=cfg.AUGMENT_IMAGE,
    shuffle_batch=False,
    cfg=cfg
  )

  # Inference Nodes
  features = model.build(graph, images, cfg=cfg)

  logits = add_logits(graph, features, cfg.NUM_CLASSES)

  top_1_op = tf.nn.in_top_k(logits, labels_sparse, 1)
  top_5_op = tf.nn.in_top_k(logits, labels_sparse, 5)
  
  if save_classification_results:
    class_scores, predicted_classes = tf.nn.top_k(logits)

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

  fetches = [top_1_op, top_5_op]
  fetch_indices = {
    'top_1_op' : 0,
    'top_5_op' : 1,
  }
  if save_classification_results :
    fetches += [predicted_classes, logits, instance_ids, labels_sparse]
    fetch_indices['predicted_classes'] = 2
    fetch_indices['logits'] = 3
    fetch_indices['instance_ids'] = 4
    fetch_indices['labels_sparse'] = 5

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
      classification_writer = None
      if save_classification_results: 
        # create a writer for storing the tfrecords
        output_path = os.path.join(summary_dir, 'classification_results-%d.tfrecords' % (global_step,))
        classification_writer = tf.python_io.TFRecordWriter(output_path)
      
      true_count = 0.0  # Counts the number of correct predictions.
      in_top_5_count = 0.0 # Counts the number of predictions that are in the top 5
      total_sample_count = 0
      step = 0
      while not coord.should_stop():
        
        if max_iterations != None and step > max_iterations:
          break
        
        t = time.time()
        outputs = sess.run(fetches)
        dt = time.time()-t

        if save_classification_results:
          batch_pred_class_ids = outputs[fetch_indices['predicted_classes']].astype(int)
          batch_logits = outputs[fetch_indices['logits']].astype(float)
          batch_instance_ids = outputs[fetch_indices['instance_ids']]
          batch_gt_class_ids = outputs[fetch_indices['labels_sparse']].astype(int)
          
          for i in range(cfg.BATCH_SIZE):
            
            feature={}
            feature['label'] = _int64_feature([batch_gt_class_ids[i]])
            feature['instance_id'] = _bytes_feature([batch_instance_ids[i]])
            feature['logits'] = _float_feature(batch_logits[i])
            feature['pred_label'] = _int64_feature(batch_pred_class_ids[i])
            
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            classification_writer.write(example.SerializeToString())
            

        predictions = outputs[fetch_indices['top_1_op']]
        true_count += np.sum(predictions)
        
        predictions_at_5 = outputs[fetch_indices['top_5_op']]
        in_top_5_count += np.sum(precision_at_5)
        
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

    if save_classification_results:
      # close the writer
      classification_writer.close()

    # Compute precision @ 1 and @5.
    precision_at_1 = true_count / total_sample_count
    precision_at_5 = in_top_5_count / total_sample_count
    print('Model %d: precision @ 1 = %.3f' % (global_step, precision_at_1))
    print('          precision @ 5 = %.3f' % (global_step, precision_at_5))
    
    # keep a conveinence file with the precision (append to an existing file)
    if summary_dir != None:
      precision_file = os.path.join(summary_dir, "precision_summary.txt")
      with open(precision_file, 'a') as f:
        print >> f, 'Model %d: precision @ 1 = %.3f' % (global_step, precision)
    
    if summary_writer != None:
      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary.value.add(tag='Precision @ 5', simple_value=precision_at_5)
      summary_writer.add_summary(summary, global_step)

  coord.request_stop()
  coord.join(threads)

  if summary_writer != None:
    summary_writer.flush()
    summary_writer.close()
