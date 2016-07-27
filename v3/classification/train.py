"""
Train the network for classification.
"""

import os
import time

import numpy as np
import tensorflow as tf

from network_utils import add_logits
import v3.classification.model as model
from inputs.classification.construct import construct_network_input_nodes

def add_loss(graph, logits, labels_sparse, scale=1.0):
  """
  Add the loss layers to the network.
  
  Args:
    graph: the graph to add the operations to. 
    logits: unscaled log probabilities, must have shape = [batch_size, num_classes]
    labels_sparse: Each entry labels_sparse[i] must be an index in [0, num_classes). 
      Other values will result in a loss of 0, but incorrect gradient computations.
    scale: A multiplicative factor for the loss. 
  """
  
  with graph.name_scope('loss'):
    batch_size, num_classes = logits.get_shape().as_list()
    
    # Convert the sparse labels to a dense matrix
    # labels_dense = tf.sparse_to_dense(
#       sparse_indices = tf.transpose(
#         tf.pack([tf.range(batch_size), labels_sparse])
#       ),
#       output_shape = [batch_size, num_classes],
#       sparse_values = np.ones(batch_size, dtype='float32')
#     )
#     loss = tf.nn.softmax_cross_entropy_with_logits(logits, labels_dense)
    
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels_sparse)
    
    # mean across the batch
    loss = tf.reduce_mean(loss, name='loss')
    
    # scale the loss
    loss = tf.mul(loss, scale)

    tf.add_to_collection('losses', loss)

  return loss

  # sum of the cross entropy loss and the l2 norm weight loss
  #return tf.add_n(tf.get_collection('losses'), name='total_loss')

def train(tfrecords, logdir, cfg, first_iteration=False, restore_initial_auxiliary_variables=False):
  """
  Driver function for training the network.
  
  Args:
    tfrecords: an array of file paths to tfrecord protocol buffer files
    logdir: a file path to a directory to store data and information for the training 
      procedure
    cfg: an EasyDict with configuration parameters.
    first_iteration (bool): If this is the first iteration of the training procedure than 
      take care to restore only certain variables.
    restore_initial_auxiliary_variables (bool): If the base network was saved with the 
      auxillary classification heads, then restore those variables too. 
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
  images, labels_sparse, instance_ids = construct_network_input_nodes(
    tfrecords=tfrecords,
    input_type=cfg.INPUT_TYPE,
    num_epochs=None,  # Just keep going through the data
    batch_size=cfg.BATCH_SIZE,
    num_threads=cfg.NUM_INPUT_THREADS,
    add_summaries = True,
    augment=cfg.AUGMENT_IMAGE,
    shuffle_batch=True,
    capacity = cfg.QUEUE_CAPACITY,
    min_after_dequeue = cfg.QUEUE_MIN,
    cfg=cfg
  )

  # Inference Nodes
  features = model.build(graph, images, cfg)
  
  if first_iteration:
    
    # Create the auxiliary classification nodes and restore the variables
    if restore_initial_auxiliary_variables:
      if cfg.USE_EXTRA_CLASSIFICATION_HEAD:
        mixed_7_output = graph.get_tensor_by_name('mixed_7/join:0')
        second_head_features = model.add_layers_for_second_classification_head(graph, mixed_7_output, cfg)
      
      if cfg.USE_THIRD_CLASSIFICATION_HEAD:
        mixed_2_output = graph.get_tensor_by_name('mixed_2/join:0')
        third_head_features = model.add_layers_for_third_classification_head(graph, mixed_2_output, cfg)
    
  # conv kernels, gamma and beta for batch normalization
  original_inception_vars = [v for v in graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]

  # Add the softmax logits
  logits = add_logits(graph, features, cfg.NUM_CLASSES)

  # Loss Nodes
  primary_loss = add_loss(graph, logits, labels_sparse)

  # Add in the auxiliary classification heds
  if cfg.USE_EXTRA_CLASSIFICATION_HEAD:
    # Create the auxiliary classification nodes if its not the first iteration, or if we don't want to restore them from a previous model
    if not first_iteration or not restore_initial_auxiliary_variables:
      mixed_7_output = graph.get_tensor_by_name('mixed_7/join:0')
      second_head_features = model.add_layers_for_second_classification_head(graph, mixed_7_output, cfg)
    second_head_logits = add_logits(graph, second_head_features, cfg.NUM_CLASSES)
    second_head_loss = add_loss(graph, second_head_logits, labels_sparse, 0.3)

  if cfg.USE_THIRD_CLASSIFICATION_HEAD:
    # Create the auxiliary classification nodes if its not the first iteration, or if we don't want to restore them from a previous model
    if not first_iteration or not restore_initial_auxiliary_variables:
      mixed_2_output = graph.get_tensor_by_name('mixed_2/join:0')
      third_head_features = model.add_layers_for_third_classification_head(graph, mixed_2_output, cfg)
    third_head_logits = add_logits(graph, third_head_features, cfg.NUM_CLASSES)
    third_head_loss = add_loss(graph, third_head_logits, labels_sparse, 0.3)

  # Regularized Loss
  # sum of the cross entropy loss and the l2 norm weight loss
  total_loss =  tf.add_n(tf.get_collection('losses'), name='total_loss')

  # Create a global counter (to be incremented by the optimizer)
  global_step = tf.Variable(0, name='global_step', trainable=False)

  learning_rate = tf.train.exponential_decay(
    learning_rate=cfg.LEARNING_RATE,
    global_step=global_step,
    decay_steps=cfg.LEARNING_RATE_DECAY_STEPS,
    decay_rate=cfg.LEARNING_RATE_DECAY,
    staircase=cfg.LEARNING_RATE_STAIRCASE
  )
  
  # Which variables are trainable? 
  if cfg.FIXED_BASE:
    var_list = [v for v in graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v not in original_inception_vars]
  else:
    var_list = [v for v in graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
  
  # Create the optimizer
  optimizer = tf.train.RMSPropOptimizer(
    learning_rate = learning_rate,
    decay = cfg.RMSPROP_DECAY, # 0.9, # Parameter setting from the arxiv paper
    epsilon = cfg.RMSPROP_EPSILON  # 1.0 #Parameter setting from the arxiv paper
  )

  # Compute the gradients using the loss
  gradients = optimizer.compute_gradients(total_loss, var_list=var_list)
  # Apply the gradients
  optimize_op = optimizer.apply_gradients(
    grads_and_vars = gradients,
    global_step = global_step
  )

  # Add in the moving average nodes
  # we want to add moving averages for the learnable params:
  # the conv kernels and the batch norm gamma and beta
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
  
  # Make sure the that there is a `checkpoints` directory
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
  ckpt = tf.train.get_checkpoint_state(save_dir)
  if ckpt:
    ckpt_file = ckpt.model_checkpoint_path
    print "Found checkpoint: %s" % (ckpt_file,)

  # If this is the first iteration then restore the original inception weights
  if first_iteration:
    original_inception_saver = tf.train.Saver(
      var_list=original_inception_vars
    )

  # Add some more summaries
  tf.histogram_summary(features.op.name, features)
  tf.histogram_summary(logits.op.name, logits)
  tf.scalar_summary('primary_loss', primary_loss)
  if cfg.USE_EXTRA_CLASSIFICATION_HEAD:
      tf.scalar_summary('secondary_loss', second_head_loss)
  if cfg.USE_THIRD_CLASSIFICATION_HEAD:
      tf.scalar_summary('tertiary_loss', third_head_loss)
  tf.scalar_summary(total_loss.op.name, total_loss)
  tf.scalar_summary(learning_rate.op.name, learning_rate)
  for grad, var in gradients:
    tf.histogram_summary(var.op.name, var)
    tf.histogram_summary(var.op.name + '/gradients', grad)
  summary_op = tf.merge_all_summaries()

  # create the directory to hold the summary outputs
  summary_logdir = os.path.join(logdir, 'summary')
  if not os.path.exists(summary_logdir):
    os.makedirs(summary_logdir)
  print "Storing summary files in %s" % (summary_logdir,)

  # create the summary writer to write the event files
  summar_writer = tf.train.SummaryWriter(
    summary_logdir,
    graph=graph,
    max_queue=10,
    flush_secs=30
  )


  # Have the session run and evaluate the following ops / tensors:
  fetches = [total_loss, training_op, primary_loss]
  if cfg.USE_EXTRA_CLASSIFICATION_HEAD:
    fetches.append(second_head_loss)
  if cfg.USE_THIRD_CLASSIFICATION_HEAD:
    fetches.append(third_head_loss)

  # Print some information to the command line on each run
  print_str = ', '.join([
    'Step: %d',
    'Loss: %.4f',
    'Time/image (ms): %.1f'
  ])

  # Now create a training coordinator that will control the different threads
  coord = tf.train.Coordinator()

  with tf.Session(graph=graph, config=sess_config) as sess:
    # make sure to initialize all of the variables
    tf.initialize_all_variables().run()

    # launch the queue runner threads
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # If there was a checkpoint file, then restore it.
    if ckpt and first_iteration:
      original_inception_saver.restore(sess, ckpt.model_checkpoint_path)
    elif ckpt:
      saver.restore(sess, ckpt.model_checkpoint_path)

    try:
      step = global_step.eval()
      while step < cfg.NUM_TRAIN_ITERATIONS:
        if coord.should_stop():
          break

        t = time.time()
        fetched = sess.run(fetches)
        dt = time.time() - t

        # increment the global step counter
        step = global_step.eval()

        print print_str % (step, fetched[0], (dt / cfg.BATCH_SIZE) * 1000)
        print "Primary loss: %0.3f" % (fetched[2],)
        if cfg.USE_EXTRA_CLASSIFICATION_HEAD:
          print "Secondary loss: %0.3f" % (fetched[3],)
        if cfg.USE_THIRD_CLASSIFICATION_HEAD:
          print "Tertiary loss: %0.3f" % (fetched[4],)

        if (step % 50) == 0:
          print "writing summary"
          summar_writer.add_summary(sess.run(summary_op), global_step=step)

        if (step % cfg.SAVE_EVERY_N_ITERATIONS) == 0:
          print "saving model"
          saver.save(
            sess=sess,
            save_path= os.path.join(save_dir, 'ft_inception_v3'),
            global_step=step
          )    
    
    except Exception as e:
     # Report exceptions to the coordinator.
     coord.request_stop(e)
  
  try:   
    # When done, ask the threads to stop. It is innocuous to request stop twice.
    coord.request_stop()
    # And wait for them to actually do it.
    coord.join(threads)
  
  except tensorflow.errors.CancelledError:
    pass
