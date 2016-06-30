from easydict import EasyDict
import numpy as np
import sys
import tensorflow as tf

import model
from inputs.detection.construct import construct_network_input_nodes

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

def train(tfrecords, bbox_priors, logdir, cfg, first_iteration=False):
  
  graph = tf.Graph()
  
  sess_config = tf.ConfigProto(
    log_device_placement=False,
    #device_filters = device_filters,
    allow_soft_placement = True,
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=cfg.SESSION_CONFIG.PER_PROCESS_GPU_MEMORY_FRACTION
    )
  )
  session = tf.Session(graph=graph, config=sess_config)
  
  with graph.as_default(), session.as_default():
    
    images, batched_bboxes, batched_num_bboxes, paths = construct_network_input_nodes(
      tfrecords=tfrecords,
      max_num_bboxes=cfg.MAX_NUM_BBOXES,
      num_epochs=None,
      batch_size=cfg.BATCH_SIZE,
      num_threads=cfg.NUM_INPUT_THREADS,
      add_summaries = True,
      augment=cfg.AUGMENT_IMAGE,
      shuffle_batch=True,
      capacity = cfg.QUEUE_CAPACITY,
      min_after_dequeue = cfg.QUEUE_MIN,
      cfg=cfg
    )
    
    features = model.build(graph, images, cfg)
    
    if first_iteration: 
      # conv kernels, gamma and beta for batch normalization
      original_inception_vars = [v for v in graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]

    
    locations, confidences = model.add_detection_heads(
      graph, 
      features, 
      num_bboxes_per_cell=5, 
      batch_size=cfg.BATCH_SIZE, 
      cfg=cfg
    )
    
    location_loss, confidence_loss, matching = model.add_loss(graph, locations, confidences, batched_bboxes, batched_num_bboxes, bbox_priors, cfg)
    loss = location_loss + confidence_loss
    
    global_step = tf.Variable(0, name='global_step', trainable=False)

    learning_rate = tf.train.exponential_decay(
      learning_rate=cfg.LEARNING_RATE,
      global_step=global_step,
      decay_steps=cfg.LEARNING_RATE_DECAY_STEPS,
      decay_rate=cfg.LEARNING_RATE_DECAY,
      staircase=cfg.LEARNING_RATE_STAIRCASE
    )

    # Create the optimizer
    optimizer = tf.train.RMSPropOptimizer(
      learning_rate = learning_rate,
      decay = cfg.RMSPROP_DECAY, # 0.9, # Parameter setting from the arxiv paper
      epsilon = cfg.RMSPROP_EPSILON  # 1.0 #Parameter setting from the arxiv paper
    )

    # Compute the gradients using the loss
    gradients = optimizer.compute_gradients(loss)
    # Apply the gradients
    optimize_op = optimizer.apply_gradients(
      grads_and_vars = gradients,
      global_step = global_step
    )
    
    ema = tf.train.ExponentialMovingAverage(
      decay=cfg.MOVING_AVERAGE_DECAY,
      num_updates=global_step
    )
    maintain_averages_op = ema.apply(
      tf.get_collection('conv_params') +
      tf.get_collection('batchnorm_params') +
      tf.get_collection('softmax_params') + 
      tf.get_collection('batchnorm_mean_var')
    )

    # Create an op that will update the moving averages after each training
    # step.  This is what we will use in place of the usual training op.
    with tf.control_dependencies([optimize_op]):
      training_op = tf.group(maintain_averages_op)
    
    save_dir = os.path.join(logdir, 'checkpoints')
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
    print "Storing checkpoint files in %s" % (save_dir,)
    # Create a saver to snapshot the model
    saver = tf.train.Saver(
      # Save all variables
      max_to_keep = 3,
      keep_checkpoint_every_n_hours = 1
    )

    # Look to see if there is a checkpoint file
    ckpt = _restore_net(save_dir)

    # If this is the first iteration then restore the original inception weights
    if first_iteration:
      original_inception_saver = tf.train.Saver(
        var_list=original_inception_vars
      )

    # Add some more summaries
    tf.scalar_summary('primary_loss', loss)
    tf.scalar_summary('location_loss', location_loss)
    tf.scalar_summary('confidence_loss', confidence_loss)
    tf.scalar_summary(learning_rate.op.name, learning_rate)
    tf.histogram_summary('confidences', confidences)
    #for grad, var in gradients:
    #  tf.histogram_summary(var.op.name, var)
    #  tf.histogram_summary(var.op.name + '/gradients', grad)
    summary_op = tf.merge_all_summaries()

    # create the directory to hold the summary outputs
    summary_logdir = os.path.join(logdir, 'summary')
    if not os.path.exists(summary_logdir):
      os.makedirs(summary_logdir)
    print "Storing summary files in %s" % (summary_logdir,)

    # create the summary writer to write the event files
    summar_writer = tf.train.SummaryWriter(
      summary_logdir,
      graph_def=graph.as_graph_def(),
      max_queue=10,
      flush_secs=30
    )
    
    fetches = [loss, training_op, learning_rate, confidences, matching, location_loss, confidence_loss]

    # Print some information to the command line on each run
    print_str = ', '.join([
      'Step: %d',
      'LR: %.4f',
      'Loss: %.4f',
      'Time/image (ms): %.1f'
    ])

    # Now create a training coordinator that will control the different threads
    coord = tf.train.Coordinator()
    
    # make sure to initialize all of the variables
    tf.initialize_all_variables().run()

    # launch the queue runner threads
    threads = tf.train.start_queue_runners(sess=session, coord=coord)

    # If there was a checkpoint file, then restore it.
    if ckpt and first_iteration:
      original_inception_saver.restore(session, ckpt.model_checkpoint_path)
    elif ckpt:
      saver.restore(session, ckpt.model_checkpoint_path)

    try:
      step = global_step.eval()
      while step < cfg.NUM_TRAIN_ITERATIONS:
        if coord.should_stop():
          break

        t = time.time()
        fetched = session.run(fetches)
        dt = time.time() - t

        # increment the global step counter
        step = global_step.eval()

        print print_str % (step, fetched[2], fetched[0], (dt / cfg.BATCH_SIZE) * 1000)
        
        if False:
          confs = fetched[3][0]
          print "Conf stats: Min: %0.3f\tMax: %0.3f\tMean: %0.3f\tMedian: %0.3f\tArgMax: %d" % (np.min(confs), np.max(confs), np.mean(confs), np.median(confs), np.argmax(confs))
          matches = fetched[4][:len(bbox_priors)]
          print "Matched: %d" % (np.argmax(matches),)
          print "GT Match: %d" % fetched[7][0]
          print "Loss : %0.5f + %0.5f = %0.5f" % (fetched[5], fetched[6], fetched[0])
          print   
         
        if (step % 50) == 0:
          print "writing summary"
          summar_writer.add_summary(session.run(summary_op), global_step=step)

        if (step % cfg.SAVE_EVERY_N_ITERATIONS) == 0:
          print "saving model"
          saver.save(
            sess=session,
            save_path= os.path.join(save_dir, 'ft_inception_v3'),
            global_step=step
          )


    except Exception as e:
      # Report exceptions to the coordinator.
      coord.request_stop(e)

    # When done, ask the threads to stop. It is innocuous to request stop twice.
    coord.request_stop()
    # And wait for them to actually do it.
    coord.join(threads)
            