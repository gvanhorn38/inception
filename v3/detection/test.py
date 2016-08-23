"""
For testing a detection system, we will take an approach similar to the COCO detection 
challenge. Detection results will be stored and 
"""

import os
import time
import json

from easydict import EasyDict
import numpy as np
from scipy.optimize import linear_sum_assignment
import sys
import tensorflow as tf

import model
from inputs.detection.construct import construct_network_input_nodes

def intersection_over_union(gt_bbox, pred_bbox):
  
  gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_bbox
  pred_xmin, pred_ymin, pred_xmax, pred_ymax = pred_bbox
  
  x1 = max(gt_xmin, pred_xmin)
  y1 = max(gt_ymin, pred_ymin)
  x2 = min(gt_xmax, pred_xmax)
  y2 = min(gt_ymax, pred_ymax)
  
  w = max(0, x2 - x1)
  h = max(0, y2 - y1)
  
  intersection = w * h
  gt_area = (gt_xmax - gt_xmin) * (gt_ymax - gt_ymin)
  pred_area = (pred_xmax - pred_xmin) * (pred_ymax - pred_ymin)
          
  iou = intersection / (1.0 * gt_area + pred_area - intersection)
  
  return iou

def test(tfrecords, bbox_priors, checkpoint_dir, specific_model_path, save_dir, max_detections, cfg):
  
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
    
    images, batched_bboxes, batched_num_bboxes, image_ids = construct_network_input_nodes(
      tfrecords=tfrecords,
      max_num_bboxes=cfg.MAX_NUM_BBOXES,
      num_epochs=1,
      batch_size=cfg.BATCH_SIZE,
      num_threads=cfg.NUM_INPUT_THREADS,
      add_summaries = False,
      augment=cfg.AUGMENT_IMAGE,
      shuffle_batch=False,
      cfg=cfg
    )
    
    features = model.build(graph, images, cfg)
    
    locations, confidences = model.add_detection_heads(
      graph, 
      features, 
      num_bboxes_per_cell=5, 
      batch_size=cfg.BATCH_SIZE, 
      cfg=cfg
    )
    
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

    fetches = [locations, confidences, image_ids]
    
    coord = tf.train.Coordinator()
    
    tf.initialize_all_variables().run()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    detection_results = []
    
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
        img_ids = outputs[2]
        
        for b in range(cfg.BATCH_SIZE):
          
          indices = np.argsort(confs[b].ravel())[::-1]
          img_id = img_ids[b]
          
          for i in range(max_detections):
            loc = locs[b][indices[i]]
            conf = confs[b][indices[i]]          
            prior = bbox_priors[indices[i]]
            
            pred_xmin, pred_ymin, pred_xmax, pred_ymax = prior + loc
            
            pred_w = pred_xmax - pred_xmin
            pred_h = pred_ymax - pred_ymin
            
            detection_results.append({
              "image_id" : int(img_id),
              "category_id" : 0,
              "bbox" : [pred_xmin, pred_ymin, pred_w, pred_h],
              "score" : float(conf),
            })
            
        step += 1
        print print_str % (step, (dt / cfg.BATCH_SIZE) * 1000)
          

    except tf.errors.OutOfRangeError as e:
      pass
      
    coord.request_stop()
    coord.join(threads)
    
    # save the results
    save_path = os.path.join(save_dir, "results-%d.json" % global_step)
    with open(save_path, 'w') as f: 
      json.dump(detection_results, f)
            