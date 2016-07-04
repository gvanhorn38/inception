"""
Customizes the Inception V3 model for classification. 
"""

import tensorflow as tf

from network_utils import add_avg_pool, add_conv, add_fully_connected
import v3.model as model

# For the original inception-v3, input should be coming from mixed_7
# This auxillary loss should be weighted by 0.3 (as stated in the `Going Deeper with Convolutions` paper)
def add_layers_for_second_classification_head(graph, input, cfg):
  """
  Add an extra classification head to the graph, right after the second block of
  inception modules.
  """

  with graph.name_scope("second_head"):
    pool_2 = add_avg_pool(
      graph=graph,
      input=input,
      ksize=[1, 5, 5, 1],
      strides=[1, 3, 3, 1],
      padding = "VALID",
      name="pool_2"
    )

    conv = add_conv(
      graph = graph,
      input = pool_2,
      shape =  [1, 1, 768, 128],
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )
    
    fc = add_fully_connected(graph, conv, 1024)
    
  return fc

def add_layers_for_third_classification_head(graph, input, cfg):
  """
  Add a third classification head to the graph, right after the first block of
  inception modules.
  """

  with graph.name_scope("third_head"):
    
    pool_2 = add_avg_pool(
      graph=graph,
      input=input,
      ksize=[1, 5, 5, 1],
      strides=[1, 3, 3, 1],
      padding = "VALID",
      name="pool_2"
    )

    conv = add_conv(
      graph = graph,
      input = pool_2,
      shape =  [1, 1, 288, 128],
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )
    
    fc = add_fully_connected(graph, conv, 1024)

  return fc


def build(graph, inputs, cfg):
  """
  Build the inception model, and add a final layer of pooling. 
  
  Args:
    graph: the graph to add the operations to. 
    inputs: the input tensor, shape : [batch_size, height, width, dims]
    cfg: EasyDict with configuration params
  """
  
  base_features = model.build(graph, inputs, cfg)
  
  # Pool the features from the last inception layer
  pool_3 = add_avg_pool(
    graph=graph,
    input=base_features,
    ksize=[1, 8, 8, 1],
    strides=[1, 1, 1, 1],
    padding = "VALID",
    name="pool_3"
  )

  # reshape to the final feature dimension. 
  features = tf.reshape(pool_3, [-1, 2048], name='features')
  
  return features
  