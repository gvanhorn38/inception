import network

def add_logits(graph, features, num_classes, wd=SOFTMAX_WD):
  with graph.name_scope("softmax") as scope:

    features_dim = features.get_shape().as_list()[-1]

    weights = _variable_with_weight_decay(
      name='weights',
      shape=[features_dim, num_classes],
      initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
      wd=wd
    )
    graph.add_to_collection('softmax_params', weights)
     
    biases = _variable_on_cpu(
      name='biases', 
      shape=[num_classes], 
      initializer=tf.constant_initializer(0.0)
    )
    graph.add_to_collection('softmax_params', biases)

    softmax_linear = tf.nn.xw_plus_b(features, weights, biases, name="logits")

  return softmax_linear

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
  
  base_features = network.build(graph, inputs, cfg)
  
  # Note that pool_2 was probably on an "alternate head"
  # pool_3
  pool_3 = add_avg_pool(
    graph=graph,
    input=base_features,
    ksize=[1, 8, 8, 1],
    strides=[1, 1, 1, 1],
    padding = "VALID",
    name="pool_3"
  )

  features = tf.reshape(pool_3, [-1, 2048], name='features')
  
  
  return features
  