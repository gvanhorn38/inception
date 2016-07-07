import glob
from matplotlib import pyplot as plt
import numpy as np
import os
import readline
from scipy.misc import imread
import tensorflow as tf
import time

import model
from inputs.detection.inputs import resize_image_maintain_aspect_ratio



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

    fetches = [locations, confidences]
    
    coord = tf.train.Coordinator()
    
    tf.initialize_all_variables().run()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    # File path tab completion. A bit lazy...
    def complete(text, state):
      options = [x for x in glob.glob(text+'*')] + [None]
      return options[state]
      
    readline.set_completer_delims('\t')
    readline.parse_and_bind("tab: complete")
    readline.set_completer(complete)
    
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
      
      bbox_colors = [
        ('r', 'red'), 
        ('b', 'blue'), 
        ('g', 'green'),
        ('c', 'cyan'), 
        ('m', 'magenta'),
        ('y', 'yellow'), 
        ('k', 'black'),
        ('w', 'white')
      ]
      
      done = False
      
      fig = plt.figure("Detected Objects")
      
      while not coord.should_stop() and not done:
        
        file_path = raw_input("File Path:")
        while not os.path.exists(file_path) or not os.path.isfile(file_path):
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
        plt.imshow(resized_image.astype(np.int8))
        plt.axis('off')
        
        # pos = [left, bottom, width, height]
        plt.gca().set_position([0, .2, .8, .8])
        
        figure_text = ""
        
        indices = np.argsort(confs[0].ravel())[::-1]
        current_index = 0
        bbox_index = 0
        
        # Draw the most confident box 
        loc = locs[0][indices[current_index]]
        prior = bbox_priors[indices[current_index]]
        xmin, ymin, xmax, ymax = (prior + loc) * cfg.INPUT_SIZE
        plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], '%s-' % bbox_colors[bbox_index][0])
        
        figure_text += "BBox %d, color: %s, confidence : %0.3f\n" % (current_index, bbox_colors[bbox_index][1], confs[0][indices[current_index]][0])
        
        current_index += 1
        bbox_index = current_index % len(bbox_colors)
        fig_y = .03
        fig_x = .1
        
        fig.text(fig_y, fig_x, figure_text)
        
        plt.show()
        
        while True:
          t = raw_input("Press `b` for next bounding box, `n` for next image:")
          if t == 'b':
            loc = locs[0][indices[current_index]]
            prior = bbox_priors[indices[current_index]]
            xmin, ymin, xmax, ymax = (prior + loc) * cfg.INPUT_SIZE
            plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], '%s-' % bbox_colors[bbox_index][0])
            
            figure_text =  "BBox %d, color: %s, confidence : %0.3f\n" % (current_index, bbox_colors[bbox_index][1], confs[0][indices[current_index]][0])
            
            fig.text(fig_y, fig_x - (.03 * (current_index % 5)), figure_text)  
            
            current_index += 1
            bbox_index = current_index % len(bbox_colors)
            
            if current_index % 5 == 0:
              fig_y = .5 
            
            plt.show()
          elif t == 'n':
            break
          else:
            done = True
            break
        
        plt.clf()  
       

    except tf.errors.OutOfRangeError as e:
      pass
      
    coord.request_stop()
    coord.join(threads)

    
            