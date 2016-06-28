"""
Intersection over Union test? 
"""


from easydict import EasyDict
import numpy as np
from scipy.optimize import linear_sum_assignment
import sys
import tensorflow as tf

import build
from inputs import input_nodes

sys.path.append('..')
import v3.v3 as v3

import os
import time

def _restore_net(checkpoint_dir):

  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  if ckpt:
    ckpt_file = ckpt.model_checkpoint_path
    print "Found checkpoint: %s" % (ckpt_file,)

  return ckpt


def test(tfrecords, bbox_priors, checkpoint_dir, specific_model_path, cfg):
  
  graph = tf.Graph()
  
  sess_config = tf.ConfigProto(
    log_device_placement=False,
    #device_filters = device_filters,
    allow_soft_placement = True,
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=cfg.SESSION_CONFIG.PER_PROCESS_GPU_MEMORY_FRACTION
    )
  )
  sess = tf.Session(graph=graph, config=sess_config)
  
  with graph.as_default(), sess.as_default():
    
    images, batched_bboxes, batched_num_bboxes, paths, gtb, best_prior_indices = input_nodes(
      tfrecords=tfrecords,
      bbox_priors = bbox_priors,
      max_num_bboxes=cfg.MAX_NUM_BBOXES,
      num_epochs=1,
      batch_size=cfg.BATCH_SIZE,
      num_threads=cfg.NUM_INPUT_THREADS,
      add_summaries = False,
      #augment=False,
      shuffle_batch=False,
      cfg=cfg
    )
    features = v3.build(graph, images, None, cfg)
    
    mixed_10_output = graph.get_tensor_by_name('mixed_10/join:0')
    
    locations, confidences = build.add_detection_heads(graph, mixed_10_output, num_bboxes_per_cell=5, batch_size=cfg.BATCH_SIZE, cfg=cfg)
    
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
    saver = tf.train.Saver(shadow_vars)

    fetches = [locations, confidences, gtb]
    
    coord = tf.train.Coordinator()
    
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
      
      print_str = ', '.join([
        'Step: %d',
        'Avg Overlap: %.4f',
        'Time/image (ms): %.1f'
      ])
      
      total_sample_count = 0
      step = 0
      done = False
      total_overlap = 0.
      while not coord.should_stop() and not done:
        
        t = time.time()
        outputs = sess.run(fetches)
        dt = time.time()-t
        
        locs = outputs[0]
        confs = outputs[1]
        gt_bboxes = outputs[2]
        
        for b in range(cfg.BATCH_SIZE):
          
          # Draw the GT Box
          gt_bbox = gt_bboxes[b][0] # Assume 1
          gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_bbox
          
          indices = np.argsort(confs[b].ravel())[::-1]
          
          # Draw the most confident box in red
          loc = locs[b][indices[0]]
          prior = bbox_priors[indices[0]]
          pred_xmin, pred_ymin, pred_xmax, pred_ymax = prior + loc
          
          x1 = max(gt_xmin, pred_xmin)
          y1 = max(gt_ymin, pred_ymin)
          x2 = min(gt_xmax, pred_xmax)
          y2 = min(gt_ymax, pred_ymax)
          
          w = max(0, x2 - x1)
          h = max(0, y2 - y1)
          
          intersection = w * h
          gt_area = (gt_xmax - gt_xmin) * (gt_ymax - gt_ymin)
          pred_area = (pred_xmax - pred_xmin) * (pred_ymax - pred_ymin)
          
          overlap = intersection / (1.0 * gt_area + pred_area - intersection)
          
          total_overlap += overlap
          
        step += 1
        print print_str % (step, total_overlap / (step * cfg.BATCH_SIZE * 1.0), (dt / cfg.BATCH_SIZE) * 1000)
          

    except tf.errors.OutOfRangeError as e:
      pass
      
    coord.request_stop()
    coord.join(threads)
            