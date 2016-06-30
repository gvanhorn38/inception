
from matplotlib import pyplot as plt
import numpy as np
import os
import tensorflow as tf
import time

import model
from inputs.detection.construct import construct_network_input_nodes


def visual_test(tfrecords, bbox_priors, checkpoint_dir, specific_model_path, cfg):
  
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
    
    images, batched_bboxes, batched_num_bboxes, paths = construct_network_input_nodes(
      tfrecords=tfrecords,
      max_num_bboxes=cfg.MAX_NUM_BBOXES,
      num_epochs=None,
      batch_size=cfg.BATCH_SIZE,
      num_threads=cfg.NUM_INPUT_THREADS,
      add_summaries = False,
      augment=cfg.AUGMENT_IMAGE,
      shuffle_batch=False,
      cfg=cfg
    )
    features = model.build(graph, images, cfg)
    
    locations, confidences = model.add_detection_heads(graph, features, num_bboxes_per_cell=5, batch_size=cfg.BATCH_SIZE, cfg=cfg)
    
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

    fetches = [locations, confidences, images, batched_bboxes, batched_num_bboxes]
    
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
      
      total_sample_count = 0
      step = 0
      done = False
      while not coord.should_stop() and not done:
        
        t = time.time()
        outputs = sess.run(fetches)
        dt = time.time()-t
        
        locs = outputs[0]
        confs = outputs[1]
        imgs = outputs[2]
        gt_bboxes = outputs[3]
        gt_num_bboxes = outputs[4]
        
        print locs.shape
        print confs.shape
        
        for b in range(cfg.BATCH_SIZE):
          
          # Show the image
          image = imgs[b]
          plt.imshow((image * cfg.IMAGE_STD + cfg.IMAGE_MEAN).astype(np.uint8))
          
           # Draw the GT Boxes in blue
          for i in range(gt_num_bboxes[b]):
            gt_bbox = gt_bboxes[b][i]
            xmin, ymin, xmax, ymax = gt_bbox * cfg.INPUT_SIZE
            plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'b-')

          # Draw the most confident boxes in red
          indices = np.argsort(confs[b].ravel())[::-1]
          for index in indices[0:gt_num_bboxes[b]]:
          
            loc = locs[b][index].ravel()
            prior = np.array(bbox_priors[index])
            
            xmin, ymin, xmax, ymax = (prior + loc) * cfg.INPUT_SIZE
            plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'r-')
          
          plt.show()
          
          t = raw_input("push button")
          if t != '':
            done = True
          
       

    except tf.errors.OutOfRangeError as e:
      pass
      
    coord.request_stop()
    coord.join(threads)

    
            