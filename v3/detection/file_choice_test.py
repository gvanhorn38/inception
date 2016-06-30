from easydict import EasyDict
from matplotlib import pyplot as plt
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

from scipy.misc import imread
from inputs import resize_image_maintain_aspect_ratio


def test(bbox_priors, checkpoint_dir, specific_model_path, cfg):
  
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
    
    images = tf.placeholder(tf.float32, [1, 299, 299, 3])
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

    fetches = [locations, confidences]
    
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
      
      done = False
      while not coord.should_stop() and not done:
        
        file_path = raw_input("File Path:")
        while not os.path.exists(file_path):
          file_path = raw_input("File Path:")
        
        img = imread(file_path)
        resized_image, h, w = resize_image_maintain_aspect_ratio(img.astype(np.float32), cfg.INPUT_SIZE, cfg.INPUT_SIZE)
        preped_image = (np.copy(resized_image) - cfg.IMAGE_MEAN) / cfg.IMAGE_STD
        imgs = np.expand_dims(preped_image, axis=0)
        
        t = time.time()
        outputs = sess.run(fetches, {images : imgs})
        dt = time.time()-t
        
        locs = outputs[0]
        confs = outputs[1]
          
        # Show the image
        plt.figure()
        plt.imshow(img)
        plt.figure()
        plt.imshow(resized_image.astype(np.int8))

        indices = np.argsort(confs[0].ravel())[::-1]
        print "Confidences: %s" % ([confs[0][i][0] for i in indices[:5]],)
        
        # Draw the most confident box in red
        loc = locs[0][indices[0]]
        prior = bbox_priors[indices[0]]
        xmin, ymin, xmax, ymax = (prior + loc) * cfg.INPUT_SIZE
        plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'r-')
        
        # Draw the next 4 boxes in green
        for index in indices[1:5]:
        
          loc = locs[0][index]
          prior = bbox_priors[index]
          
          xmin, ymin, xmax, ymax = (prior + loc) * cfg.INPUT_SIZE
          plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'g-')
        
        plt.show()
        
        t = raw_input("push button")
        if t != '':
          done = True
          
       

    except tf.errors.OutOfRangeError as e:
      pass
      
    coord.request_stop()
    coord.join(threads)

    
            