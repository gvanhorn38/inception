"""
Take multiple crops from an image and combine the results, in an attempt to boost mAP.
"""

from matplotlib import pyplot as plt
from scipy.misc import imresize

import json
import numpy as np
import os
import sys
import tensorflow as tf
import time

import model
import patch_utils

DEBUG = False

def single_image_input(tfrecords, cfg):
  
    filename_queue = tf.train.string_input_producer(
      tfrecords,
      num_epochs=1
    )

    # Construct a Reader to read examples from the .tfrecords file
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # Parse an Example to access the Features
    features = tf.parse_single_example(
      serialized_example,
      features = {
        'image/id' : tf.FixedLenFeature([], tf.string),
        'image/encoded'  : tf.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin' : tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin' : tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax' : tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax' : tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/count' : tf.FixedLenFeature([], tf.int64)
      }
    )

    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    image_id = features['image/id']
    
    return image, image_id

def test(tfrecords, bbox_priors, checkpoint_dir, specific_model_path, save_dir, max_detections, cfg):
  """
  Args:
    tfrecords (list) : a list of file paths to tfrecords
    bbox_priors (list) : a list of [x1, y1, x2, y2] bounding box priors
    checkpoint_dir (str) : a directory path to where the checkpoints are stored
    specific_model_path (str) : a file path to a specific model
    save_dir (str) : a directory path where to store the detection results
    max_detections (int) : the maximum number of detections to store per image
    cfg (EasyDict) : configuration parameters
  """
  
  # shape: [number of priors, 4]
  bbox_priors = np.array(bbox_priors)
  
  # Important to override the batch size to ensure its 1
  cfg.BATCH_SIZE = 1
  
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
    
    # A graph to read one image out of the tfrecord files
    single_image, single_image_id = single_image_input(tfrecords, cfg)
    
    # A graph to predict bounding boxes
    # The placeholder will be filled in with crops
    feed_image = tf.placeholder(tf.int8, [None, None, 3])
    feed_image_height = tf.placeholder(tf.int32, [])
    feed_image_width = tf.placeholder(tf.int32, [])
    image_to_prep = tf.reshape(feed_image, tf.pack([feed_image_height, feed_image_width, 3])) 
    float_image = tf.cast(image_to_prep, tf.float32)
    resized_image = tf.image.resize_images(float_image, cfg.INPUT_SIZE, cfg.INPUT_SIZE)
    resized_image -= cfg.IMAGE_MEAN
    resized_image /= cfg.IMAGE_STD
    resized_image = tf.expand_dims(resized_image, 0)
    features = model.build(graph, resized_image, cfg)
    
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
    
    coord = tf.train.Coordinator()
    
    tf.initialize_all_variables().run()
    tf.initialize_local_variables().run() # This is needed for the filename_queue
    
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    detection_results = []
    
    if DEBUG:
      plt.ion()
    
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
      
      step = 0
      while not coord.should_stop():
        
        if DEBUG:
          if step == 5:
            break
        
        start_time = time.time()
        
        image, image_id = sess.run([single_image, single_image_id])
        image_dims = image.shape[:2]
        
        # Process the original sized image
        outputs = sess.run([locations, confidences], {
          feed_image : image,
          feed_image_height : image_dims[0],
          feed_image_width : image_dims[1]
        })
        
        bboxes = outputs[0][0] + bbox_priors
        confs = outputs[1][0]
        
        # Restrict to the max number of detections
        sorted_idxs = np.argsort(confs.ravel())[::-1]
        sorted_idxs = sorted_idxs[:max_detections]
        all_bboxes = bboxes[sorted_idxs]
        all_confs = confs[sorted_idxs].ravel()
        
        # We could try to do some batching here....
        # Now process the crops
        for patch_dims, offset in [[(299, 299), (149, 149)], [(185, 185), (93, 93)]]:
          patches, patch_offsets = patch_utils.extract_patches(image, patch_dims, offset)
          for patch, patch_offset in zip(patches, patch_offsets):
            outputs = sess.run([locations, confidences], {
              feed_image : patch,
              feed_image_height : patch_dims[0],
              feed_image_width : patch_dims[1]
            })
            
            # Get the predictions, don't forget that there is a batch size of 1
            predicted_bboxes = outputs[0][0] + bbox_priors
            predicted_confs = outputs[1][0]
            
            # Keep only the predictions that are completely contained in the [0.1, 0.1, 0.9, 0.9] square
            # for this patch
            filtered_bboxes, filtered_confs = patch_utils.filter_proposals(predicted_bboxes, predicted_confs) 
            
            # No valid predictions? 
            if filtered_bboxes.shape[0] == 0:
              continue
            
            # Lets get rid of some of the predictions
            sorted_idxs = np.argsort(filtered_confs.ravel())[::-1]
            sorted_idxs = sorted_idxs[:max_detections]
            filtered_bboxes = filtered_bboxes[sorted_idxs]
            filtered_confs = filtered_confs[sorted_idxs]
            
            # Convert the bounding boxes to the original image dimensions
            converted_bboxes = patch_utils.convert_proposals(
              bboxes = filtered_bboxes, 
              offset = patch_offset, 
              patch_dims = patch_dims, 
              image_dims = image_dims
            )
            
            # Add the bboxes and confs to our collection
            all_bboxes = np.vstack([all_bboxes, converted_bboxes])
            all_confs = np.hstack([all_confs, filtered_confs.ravel()])
        
        # Lets make sure the coordinates are in the proper order
        # Its interesting that we don't enforce this anywhere in the model
        proper_bboxes = []
        for loc in all_bboxes:
        
          pred_xmin, pred_ymin, pred_xmax, pred_ymax = loc
          
          # Not sure what we want to do here. Its interesting that we don't enforce this anywhere in the model
          if pred_xmin > pred_xmax:
            t = pred_xmax
            pred_xmax = pred_xmin
            pred_xmin = t
          if pred_ymin > pred_ymax:
            t = pred_ymax
            pred_ymax = pred_ymin
            pred_ymin = t
          
          proper_bboxes.append([pred_xmin, pred_ymin, pred_xmax, pred_ymax])
        all_bboxes = np.array(proper_bboxes)
        
        if DEBUG:
          plt.figure('all detections')
          plt.clf()  
          plt.imshow(imresize(image, [299, 299]))
          r = ""
          b = 0
          while r == "":
            xmin, ymin, xmax, ymax = all_bboxes[b] * np.array([299, 299, 299, 299])
            print "BBox: [%0.3f, %0.3f, %0.3f, %0.3f]" % (xmin, ymin, xmax, ymax)
            print "Conf: ", all_confs[b]
            plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'b-')
            plt.show()
            b+= 1
            r = raw_input("push button")
        
          print "Number of total bboxes: %d" % (all_bboxes.shape[0], )
          
        # Lets do non-maximum suppression for the final predictions
        locs, confs = patch_utils.non_max_suppression(all_bboxes, all_confs, jaccard_threshold=0.85)          
        
        # Now process the predictions
        indices = np.argsort(confs.ravel())[::-1]
        
        if DEBUG:
          print "Number of nms bboxes: %d" % (locs.shape[0], )
          print "Removed %d detections" % (all_bboxes.shape[0] - locs.shape[0],) 
        
          print "Sorted confidences: %s" % (confs[indices][:10],)
        
          plt.figure('top detections')
          plt.clf()
          plt.imshow(imresize(image, [299, 299]))
          r = ""
          b = 0
          while r == "":
            xmin, ymin, xmax, ymax = locs[indices[b]] * np.array([299, 299, 299, 299])
            print "BBox: [%0.3f, %0.3f, %0.3f, %0.3f]" % (xmin, ymin, xmax, ymax)
            print "Conf: ", confs[indices[b]]
            plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'b-')
            plt.show()
            b+= 1
            r = raw_input("push button")
 
        for index in indices[0:max_detections]:
          loc = locs[index].ravel()
          conf = confs[index]          
          
          pred_xmin, pred_ymin, pred_xmax, pred_ymax = loc
          
          detection_results.append({
            "image_id" : int(image_id), # converts  from np.array
            "bbox" : [pred_xmin, pred_ymin, pred_xmax, pred_ymax],
            "score" : float(conf), # converts from np.array
          })
        
        dt = time.time()-start_time     
        step += 1
        print print_str % (step, (dt / cfg.BATCH_SIZE) * 1000)
          

    except tf.errors.OutOfRangeError as e:
      pass
      
    coord.request_stop()
    coord.join(threads)
    
    # save the results
    save_path = os.path.join(save_dir, "dense-results-%d.json" % global_step)
    with open(save_path, 'w') as f: 
      json.dump(detection_results, f)
            