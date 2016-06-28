import os
import sys
import time


import numpy as np
import tensorflow as tf

import v3

from config import parse_config_file

def create_initial_training_model(primary_model_path, auxiliary_model_path, output_path, cfg):
  
  """
  We want to save off the primary weights as well as the secondary and tertiary classification head weights
  """
  
  USE_EXTRA_CLASSIFICATOIN_HEAD = True
  USE_THIRD_CLASSIFICATION_HEAD = True
  
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
  images = tf.zeros([cfg.BATCH_SIZE, cfg.INPUT_SIZE, cfg.INPUT_SIZE, 3])

  # Inference Nodes (num_classes is not used)
  features = v3.build(graph, images, cfg.NUM_CLASSES, cfg=cfg)
    

  # The prep phase already switched in the moving average variables, so just restore them
  shadow_vars = {
    var.name[:-2] : var
    for var in graph.get_collection('conv_params')
  }
  shadow_vars.update({
    var.name[:-2] : var
    for var in graph.get_collection('batchnorm_params') # this is the beta and gamma parameters
  })
  # Restore the primary parameters
  primary_restore = tf.train.Saver(shadow_vars)
  
  # Add these after computing the shadow vars so that we pull these in from the auxiliary model
  # Technically we could be using the averaged value? 
  if USE_EXTRA_CLASSIFICATOIN_HEAD:
    mixed_7_output = graph.get_tensor_by_name('mixed_7/join:0')
    second_head_features = v3.add_layers_for_second_classification_head(graph, mixed_7_output, cfg)

  if USE_THIRD_CLASSIFICATION_HEAD:
    mixed_2_output = graph.get_tensor_by_name('mixed_2/join:0')
    third_head_features = v3.add_layers_for_third_classification_head(graph, mixed_2_output, cfg)
  
  # auxiliary parameters
  auxiliary_variables = {}
  if USE_EXTRA_CLASSIFICATOIN_HEAD or USE_THIRD_CLASSIFICATION_HEAD:
    for var in tf.trainable_variables():
      var_name = var.name[:-2]
      if var_name not in shadow_vars:
        print var_name
        auxiliary_variables[var_name] = var 
  auxiliary_restore = tf.train.Saver(auxiliary_variables)
  
  # save all the variables
  saver = tf.train.Saver()
  
  with graph.as_default():
    with tf.Session(graph=graph) as sess:

      tf.initialize_all_variables().run()
      
      # Restore the primary variables
      primary_restore.restore(sess, primary_model_path)
      
      # Restore the auxiliary variables
      if USE_EXTRA_CLASSIFICATOIN_HEAD or USE_THIRD_CLASSIFICATION_HEAD:
        auxiliary_restore.restore(sess, auxiliary_model_path)
      
      # Create something that we can restore from
      saver.save(
        sess=sess,
        save_path= os.path.join(output_path),
      )
    
if __name__ == '__main__':
  
  primary_model_path = sys.argv[1]
  auxiliary_model_path = sys.argv[2]
  output_path = sys.argv[3]
  cfg_path = sys.argv[4]
  
  cfg = parse_config_file(cfg_path)
  cfg.PHASE = "TRAIN"
  cfg.USE_BATCH_STATISTICS = True

  create_initial_training_model(primary_model_path, auxiliary_model_path, output_path, cfg)