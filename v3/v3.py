import tensorflow as tf

# Default Values
# WD : weight decay (regularization term)
CONV_KERNEL_WD = 1e-5
BATCHNORM_GAMMA_WD = 1e-5
BATCHNORM_BETA_WD = 0
BATCHNORM_VARIANCE_EPSILON = 0.001
BATCHNORM_SCALE_AFTER_NORM = True
SOFTMAX_WD = 4e-5
SOFTMAX_BIAS_WD = 0
FC_WD = 4e-5
DROPOUT_KEEP_PROB = 0.3

def default_config():

  return  {
    "CONV_KERNEL_WD" : CONV_KERNEL_WD,
    "BATCHNORM_GAMMA_WD" : BATCHNORM_GAMMA_WD,
    "BATCHNORM_BETA_WD" : BATCHNORM_BETA_WD,
    "BATCHNORM_VARIANCE_EPSILON" : BATCHNORM_VARIANCE_EPSILON,
    "BATCHNORM_SCALE_AFTER_NORM" : BATCHNORM_SCALE_AFTER_NORM,
    "SOFTMAX_WD" : SOFTMAX_WD,
    "SOFTMAX_BIAS_WD" : SOFTMAX_BIAS_WD,
    
    # Secondary classification head
    "USE_SECOND_HEAD" : USE_SECOND_HEAD,
    "SECOND_HEAD_FC_WD" : SECOND_HEAD_FC_WD,
    "SECOND_HEAD_DROP_OUT_KEEP_PROB" : SECOND_HEAD_DROP_OUT_KEEP_PROB,
    
    # Tertiary classification head
    "USE_THIRD_HEAD" : USE_THIRD_HEAD,
    "THIRD_HEAD_FC_WD" : THIRD_HEAD_FC_WD,
    "THIRD_HEAD_DROP_OUT_KEEP_PROB" : THIRD_HEAD_DROP_OUT_KEEP_PROB
  }

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  
  with tf.device('/cpu:0'):
    #var = tf.get_variable(name, shape=shape, initializer=initializer)
    var = tf.Variable(initial_value = initializer(shape), name=name)
  return var

def _variable_with_weight_decay(name, shape, initializer, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """

  var = _variable_on_cpu(name, shape, initializer)
  if wd:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


# Convolution + Batch Normalization + Activation Function
def add_conv(
    graph, input, 
    # Convolution Kernel parameters
    shape, strides, padding, 
    wd=CONV_KERNEL_WD, 
    # Batchnorm params
    beta_wd=BATCHNORM_BETA_WD, 
    gamma_wd=BATCHNORM_GAMMA_WD, 
    variance_epsilon=BATCHNORM_VARIANCE_EPSILON, 
    scale_after_normalization=BATCHNORM_SCALE_AFTER_NORM,
    use_batch_statistics=True,
    batch_normalize=True,
    add_relu=True
  ):

  with graph.name_scope('conv') as conv_scope:

    # Convolution
    kernel = _variable_with_weight_decay(
      name='conv2d_params',
      shape=shape,
      initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False, seed=None, dtype=tf.float32),
      wd=wd
    )
    conv = tf.nn.conv2d(input, filter=kernel, strides=strides, padding=padding)
    graph.add_to_collection('conv_params', kernel)
    if batch_normalize:
      with graph.name_scope('batchnorm') as batch_scope:
        
        beta = _variable_with_weight_decay(
          name='beta', 
          shape=[shape[3]], 
          initializer=tf.truncated_normal_initializer(stddev=0.04),
          wd=beta_wd
        )
        graph.add_to_collection('batchnorm_params', beta)
        
        gamma = _variable_with_weight_decay(
          name='gamma', 
          shape=[shape[3]], 
          initializer=tf.truncated_normal_initializer(stddev=0.04), 
          wd=gamma_wd
        )
        graph.add_to_collection('batchnorm_params', gamma)

        # During training, we want to use the batch statistics
        if use_batch_statistics:
          mu, var = tf.nn.moments(conv, [0, 1, 2], name=batch_scope)
          mean = graph.get_operation_by_name(batch_scope + "mean").outputs[0]

        # During testing and eval we want to use the global statistics
        # This need to be filled in by the moving averages version
        else:
          mean = _variable_with_weight_decay(
            name='mean',
            shape=[1, 1, 1, shape[3]],
            initializer=tf.constant_initializer(0.0), 
            wd=0
          )
          mu = tf.squeeze(mean, squeeze_dims=[0, 1, 2])
        
          var = _variable_with_weight_decay(
            name='variance',
            shape=[shape[3]],
            initializer=tf.constant_initializer(0.0), 
            wd=0
          )
          # we should probably be using this...
          #unbiased_var = var * (train_batch_size / (train_batch_size - 1))

        # We'll use this collection to add these to the moving average operation
        graph.add_to_collection('batchnorm_mean_var', mean)
        graph.add_to_collection('batchnorm_mean_var', var)

        batchnorm = tf.nn.batch_norm_with_global_normalization(
          t = conv, m=mu, v=var, beta=beta, gamma=gamma,
          variance_epsilon= variance_epsilon,
          scale_after_normalization=scale_after_normalization,
          name=batch_scope
        )

      # Activation Function
      # We could add some checks to confirm that the values are finite
      activations = tf.nn.relu(batchnorm, name=conv_scope)
    elif add_relu:
      activations = tf.nn.relu(conv, name=conv_scope)
    else:
      activations = conv
  return activations

def add_max_pool(graph, input, ksize, strides, padding, name=None):

  # We could add some checks to confirm that the values are finite

  pool = tf.nn.max_pool(
    value=input,
    ksize=ksize,
    strides=strides,
    padding=padding,
    name=name
  )

  return pool

def add_avg_pool(graph, input, ksize, strides, padding, name=None):

  # We could add some checks to confirm that the values are finite

  pool = tf.nn.avg_pool(
    value=input,
    ksize=ksize,
    strides=strides,
    padding=padding,
    name=name
  )

  return pool

def add_fully_connected(graph, input, output_dim,
  wd = FC_WD,
  dropout_keep_prob = DROPOUT_KEEP_PROB
  ):
  
  # Fully connected layer
  # Move everything into depth so we can perform a single matrix multiply.
  with graph.name_scope("fc") as scope:
    dim = 1
    shape = input.get_shape().as_list()
    batch_size = shape[0]
    for d in shape[1:]:
      dim *= d
    reshaped = tf.reshape(input, [batch_size, dim])

    # weights
    weights = _variable_with_weight_decay(
      name='weights',
      shape=[dim, output_dim],
      initializer= tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
      wd=wd
    )
    biases = _variable_on_cpu(
      name='biases', 
      shape=[output_dim], 
      initializer=tf.constant_initializer(0.0)
    )
    fc = tf.nn.relu(tf.nn.xw_plus_b(reshaped, weights, biases), name=scope)
    
    features = tf.nn.dropout(fc, dropout_keep_prob)
  
  return features

# Figure 5 configuration
# This isn't exactly figure 5. There is a 5x5 filter bank here.
# So its like a combo of Figure 4 and Figure 5
def add_figure5(graph, input,
  conv_shape,
  tower_conv_shape,
  tower_conv_1_shape,
  tower_1_conv_shape,
  tower_1_conv_1_shape,
  tower_1_conv_2_shape,
  tower_2_conv_shape,
  cfg):

  conv = add_conv(
    graph = graph,
    input = input,
    shape = conv_shape, # [1, 1, 192, 64],    # [1, 1, 256, 64],   # [1, 1, 288, 64]
    strides = [1, 1, 1, 1],
    padding = "SAME",
    use_batch_statistics=cfg.USE_BATCH_STATISTICS
  )

  # tower
  with graph.name_scope("tower"):

    tower_conv = add_conv(
      graph = graph,
      input = input,
      shape = tower_conv_shape, #[1, 1, 192, 48],   # [1, 1, 256, 48],    # [1, 1, 288, 48]
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

    tower_conv_1 = add_conv(
      graph = graph,
      input = tower_conv,
      shape = tower_conv_1_shape, #[5, 5, 48, 64],  # [5, 5, 48, 64],    # [5, 5, 48, 64]
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

  # tower_1
  with graph.name_scope("tower"):

    tower_1_conv = add_conv(
      graph = graph,
      input = input,
      shape = tower_1_conv_shape, #[1, 1, 192, 64],   # [1, 1, 256, 64],    # [1, 1, 288, 64]
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

    tower_1_conv_1 = add_conv(
      graph = graph,
      input = tower_1_conv,
      shape = tower_1_conv_1_shape, #[3, 3, 64, 96],  # [3, 3, 64, 96] ,   # [3, 3, 64, 96]
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

    tower_1_conv_2 = add_conv(
      graph = graph,
      input = tower_1_conv_1,
      shape = tower_1_conv_2_shape, #[3, 3, 96, 96],  # [3, 3, 96, 96] ,   # [3, 3, 96, 96]
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

  # tower_2
  with graph.name_scope("tower"):

    tower_2_pool = add_avg_pool(
      graph=graph,
      input=input,
      ksize=[1, 3, 3, 1],  # [1, 3, 3, 1]   # [1, 3, 3, 1]
      strides=[1, 1, 1, 1],
      padding = "SAME",
      name="pool"
    )

    tower_2_conv = add_conv(
      graph = graph,
      input = tower_2_pool,
      shape = tower_2_conv_shape, #[1, 1, 192, 32],   # [1, 1, 256, 64],  # [1, 1, 288, 64]
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

  return tf.concat(
    concat_dim=3,
    values = [conv, tower_conv_1, tower_1_conv_2, tower_2_conv],
    name="join"
  )


# First grid size reduction
# mixed_3
def add_figure10_1(graph, input, cfg):

  conv = add_conv(
    graph = graph,
    input = input,
    shape =  [3, 3, 288, 384],
    strides = [1, 2, 2, 1],
    padding = "VALID",
    use_batch_statistics=cfg.USE_BATCH_STATISTICS
  )

  with graph.name_scope("tower"):
    tower_conv = add_conv(
      graph = graph,
      input = input,
      shape =  [1, 1, 288, 64],
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

    tower_conv_1 = add_conv(
      graph = graph,
      input = tower_conv,
      shape =  [3, 3, 64, 96],
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

    tower_conv_2 = add_conv(
      graph = graph,
      input = tower_conv_1,
      shape =  [3, 3, 96, 96],
      strides = [1, 2, 2, 1],
      padding = "VALID",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

  pool = add_max_pool(
    graph=graph,
    input=input,
    ksize=[1, 3, 3, 1],
    strides=[1, 2, 2, 1],
    padding = "VALID",
    name="pool"
  )

  return tf.concat(
    concat_dim=3,
    values = [conv, tower_conv_2, pool],
    name="join"
  )


def add_figure6(graph, input,
  conv_shape,
  tower_conv_shape,
  tower_conv_1_shape,
  tower_conv_2_shape,
  tower_1_conv_shape,
  tower_1_conv_1_shape,
  tower_1_conv_2_shape,
  tower_1_conv_3_shape,
  tower_1_conv_4_shape,
  tower_2_conv_shape,
  cfg):

  conv = add_conv(
    graph = graph,
    input = input,
    shape =  conv_shape,
    strides = [1, 1, 1, 1],
    padding = "SAME",
    use_batch_statistics=cfg.USE_BATCH_STATISTICS
  )

  # tower
  with graph.name_scope("tower"):
    tower_conv = add_conv(
      graph = graph,
      input = input,
      shape =  tower_conv_shape,
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

    tower_conv_1 = add_conv(
      graph = graph,
      input = tower_conv,
      shape =  tower_conv_1_shape,
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

    tower_conv_2 = add_conv(
      graph = graph,
      input = tower_conv_1,
      shape =  tower_conv_2_shape,
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

  # tower_1
  with graph.name_scope("tower"):
    tower_1_conv = add_conv(
      graph = graph,
      input = input,
      shape =  tower_1_conv_shape,
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

    tower_1_conv_1 = add_conv(
      graph = graph,
      input = tower_1_conv,
      shape =  tower_1_conv_1_shape,
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

    tower_1_conv_2 = add_conv(
      graph = graph,
      input = tower_1_conv_1,
      shape =  tower_1_conv_2_shape,
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

    tower_1_conv_3 = add_conv(
      graph = graph,
      input = tower_1_conv_2,
      shape =  tower_1_conv_3_shape,
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

    tower_1_conv_4 = add_conv(
      graph = graph,
      input = tower_1_conv_3,
      shape =  tower_1_conv_4_shape,
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

  # tower_2
  with graph.name_scope("tower"):

    tower_2_pool = add_avg_pool(
      graph=graph,
      input=input,
      ksize=[1, 3, 3, 1],
      strides=[1, 1, 1, 1],
      padding = "SAME",
      name="pool"
    )

    tower_2_conv= add_conv(
      graph = graph,
      input = tower_2_pool,
      shape =  tower_2_conv_shape,
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

  return tf.concat(
    concat_dim=3,
    values = [conv, tower_conv_2, tower_1_conv_4, tower_2_conv],
    name="join"
  )

# Second grid size reduction
# mixed_8
def add_figure10_2(graph, input, cfg):

  # tower
  with graph.name_scope("tower"):
    tower_conv = add_conv(
      graph = graph,
      input = input,
      shape =  [1, 1, 768, 192],
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

    tower_conv_1 = add_conv(
      graph = graph,
      input = tower_conv,
      shape =  [3, 3, 192, 320],
      strides = [1, 2, 2, 1],
      padding = "VALID",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

  # tower_1
  with graph.name_scope("tower"):
    tower_1_conv = add_conv(
      graph = graph,
      input = input,
      shape =  [1, 1, 768, 192],
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

    tower_1_conv_1 = add_conv(
      graph = graph,
      input = tower_1_conv,
      shape =  [1, 7, 192, 192],
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

    tower_1_conv_2 = add_conv(
      graph = graph,
      input = tower_1_conv_1,
      shape =  [7, 1, 192, 192],
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

    tower_1_conv_3 = add_conv(
      graph = graph,
      input = tower_1_conv_2,
      shape =  [3, 3, 192, 192],
      strides = [1, 2, 2, 1],
      padding = "VALID",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

  # pool
  pool = add_max_pool(
    graph=graph,
    input=input,
    ksize=[1, 3, 3, 1],
    strides=[1, 2, 2, 1],
    padding = "VALID",
    name="pool"
  )

  return tf.concat(
    concat_dim=3,
    values = [tower_conv_1, tower_1_conv_3, pool],
    name="join"
  )

def add_figure7(graph, input,
  conv_shape,
  tower_conv_shape,
  tower_mixed_conv_shape,
  tower_mixed_conv_1_shape,
  tower_1_conv_shape,
  tower_1_conv_1_shape,
  tower_1_mixed_conv_shape,
  tower_1_mixed_conv_1_shape,
  tower_2_conv_shape,
  use_avg_pool,
  cfg):

  # conv
  conv = add_conv(
    graph = graph,
    input = input,
    shape =  conv_shape,
    strides = [1, 1, 1, 1],
    padding = "SAME",
    use_batch_statistics=cfg.USE_BATCH_STATISTICS
  )

  # tower
  with graph.name_scope("tower"):
    tower_conv = add_conv(
      graph = graph,
      input = input,
      shape =  tower_conv_shape,
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

    with graph.name_scope("mixed"):
      tower_mixed_conv = add_conv(
        graph = graph,
        input = tower_conv,
        shape =  tower_mixed_conv_shape,
        strides = [1, 1, 1, 1],
        padding = "SAME",
        use_batch_statistics=cfg.USE_BATCH_STATISTICS
      )

      tower_mixed_conv_1 = add_conv(
        graph = graph,
        input = tower_conv,
        shape =  tower_mixed_conv_1_shape,
        strides = [1, 1, 1, 1],
        padding = "SAME",
        use_batch_statistics=cfg.USE_BATCH_STATISTICS
      )

  # tower_1
  with graph.name_scope("tower"):
    tower_1_conv = add_conv(
      graph = graph,
      input = input,
      shape =  tower_1_conv_shape,
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

    tower_1_conv_1 = add_conv(
      graph = graph,
      input = tower_1_conv,
      shape =  tower_1_conv_1_shape,
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

    with graph.name_scope("mixed"):
      tower_1_mixed_conv = add_conv(
        graph = graph,
        input = tower_1_conv_1,
        shape =  tower_1_mixed_conv_shape,
        strides = [1, 1, 1, 1],
        padding = "SAME",
        use_batch_statistics=cfg.USE_BATCH_STATISTICS
      )

      tower_1_mixed_conv_1 = add_conv(
        graph = graph,
        input = tower_1_conv_1,
        shape =  tower_1_mixed_conv_1_shape,
        strides = [1, 1, 1, 1],
        padding = "SAME",
        use_batch_statistics=cfg.USE_BATCH_STATISTICS
      )

  # tower_2
  with graph.name_scope("tower"):
    if use_avg_pool:
      tower_2_pool = add_avg_pool(
          graph=graph,
          input=input,
          ksize=[1, 3, 3, 1],
          strides=[1, 1, 1, 1],
          padding = "SAME",
          name="pool"
        )
    else:
      tower_2_pool = add_max_pool(
        graph=graph,
        input=input,
        ksize=[1, 3, 3, 1],
        strides=[1, 1, 1, 1],
        padding = "SAME",
        name="pool"
      )

    tower_2_conv = add_conv(
      graph = graph,
      input = tower_2_pool,
      shape =  tower_2_conv_shape,
      strides = [1, 1, 1, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )

  return tf.concat(
    concat_dim=3,
    values = [conv, tower_mixed_conv, tower_mixed_conv_1, tower_1_mixed_conv, tower_1_mixed_conv_1, tower_2_conv],
    name="join"
  )

# Rather than passing the graph around, we could do `with graph.as_default():`
# GVH: num_classes is not used
def build(graph, input, num_classes, cfg):

  # conv
  conv = add_conv(
    graph = graph,
    input = input,
    shape = [3, 3, 3, 32],
    strides = [1, 2, 2, 1],
    padding = "VALID",
    use_batch_statistics=cfg.USE_BATCH_STATISTICS
  )

  # conv_1
  conv_1 = add_conv(
    graph = graph,
    input = conv,
    shape = [3, 3, 32, 32],
    strides = [1, 1, 1, 1],
    padding = "VALID",
    use_batch_statistics=cfg.USE_BATCH_STATISTICS
  )

  # conv_2
  conv_2 = add_conv(
    graph = graph,
    input = conv_1,
    shape = [3, 3, 32, 64],
    strides = [1, 1, 1, 1],
    padding = "SAME",
    use_batch_statistics=cfg.USE_BATCH_STATISTICS
  )

  # pool
  pool = add_max_pool(
    graph=graph,
    input=conv_2,
    ksize=[1, 3, 3, 1],
    strides=[1, 2, 2, 1],
    padding = "VALID",
    name="pool"
  )

  # conv_3
  conv_3 = add_conv(
    graph = graph,
    input = pool,
    shape = [1, 1, 64, 80],
    strides = [1, 1, 1, 1],
    padding = "VALID",
    use_batch_statistics=cfg.USE_BATCH_STATISTICS
  )

  # conv_4
  conv_4 = add_conv(
    graph = graph,
    input = conv_3,
    shape = [3, 3, 80, 192],
    strides = [1, 1, 1, 1],
    padding = "VALID",
    use_batch_statistics=cfg.USE_BATCH_STATISTICS
  )

  # pool_1
  pool_1 = add_max_pool(
    graph=graph,
    input=conv_4,
    ksize=[1, 3, 3, 1],
    strides=[1, 2, 2, 1],
    padding = "VALID",
    name="pool"
  )

  #################
  # First block of inception modules
  # 3 modules as specified in Figure 5

  # mixed
  with graph.name_scope("mixed"):
    mixed = add_figure5(graph, pool_1,
      conv_shape = [1, 1, 192, 64],
      tower_conv_shape = [1, 1, 192, 48],
      tower_conv_1_shape = [5, 5, 48, 64],
      tower_1_conv_shape = [1, 1, 192, 64],
      tower_1_conv_1_shape = [3, 3, 64, 96],
      tower_1_conv_2_shape = [3, 3, 96, 96],
      tower_2_conv_shape = [1, 1, 192, 32],
      cfg=cfg
    )

  # mixed_1
  with graph.name_scope("mixed"):
    mixed_1 = add_figure5(graph, mixed,
      conv_shape = [1, 1, 256, 64],
      tower_conv_shape = [1, 1, 256, 48],
      tower_conv_1_shape = [5, 5, 48, 64],
      tower_1_conv_shape = [1, 1, 256, 64],
      tower_1_conv_1_shape = [3, 3, 64, 96],
      tower_1_conv_2_shape = [3, 3, 96, 96],
      tower_2_conv_shape = [1, 1, 256, 64],
      cfg=cfg
    )


  # mixed_2
  with graph.name_scope("mixed"):
    mixed_2 = add_figure5(graph, mixed_1,
      conv_shape = [1, 1, 288, 64],
      tower_conv_shape = [1, 1, 288, 48],
      tower_conv_1_shape = [5, 5, 48, 64],
      tower_1_conv_shape = [1, 1, 288, 64],
      tower_1_conv_1_shape = [3, 3, 64, 96],
      tower_1_conv_2_shape = [3, 3, 96, 96],
      tower_2_conv_shape = [1, 1, 288, 64],
      cfg=cfg
    )

  # End first block of inception modules
  #################

  # First Inception module for grid size reduction
  with graph.name_scope("mixed"):
    mixed_3 = add_figure10_1(graph, mixed_2, cfg=cfg)

  # Second block of inception modules
  # 4 modules as specified in Figure 6
  # NOTE: rather than 5 modules as specified in the paper, there are only 4 modules in
  # the graph def
  # mixed_4
  with graph.name_scope("mixed"):
    mixed_4 = add_figure6(graph, mixed_3,
      conv_shape =            [1, 1, 768, 192],
      tower_conv_shape =      [1, 1, 768, 128],
      tower_conv_1_shape =    [1, 7, 128, 128],
      tower_conv_2_shape =    [7, 1, 128, 192],
      tower_1_conv_shape =    [1, 1, 768, 128],
      tower_1_conv_1_shape =  [7, 1, 128, 128],
      tower_1_conv_2_shape =  [1, 7, 128, 128],
      tower_1_conv_3_shape =  [7, 1, 128, 128],
      tower_1_conv_4_shape =  [1, 7, 128, 192],
      tower_2_conv_shape =    [1, 1, 768, 192],
      cfg=cfg
    )

  # mixed_5
  with graph.name_scope("mixed"):
    mixed_5 = add_figure6(graph, mixed_4,
      conv_shape =            [1, 1, 768, 192],
      tower_conv_shape =      [1, 1, 768, 160],
      tower_conv_1_shape =    [1, 7, 160, 160],
      tower_conv_2_shape =    [7, 1, 160, 192],
      tower_1_conv_shape =    [1, 1, 768, 160],
      tower_1_conv_1_shape =  [7, 1, 160, 160],
      tower_1_conv_2_shape =  [1, 7, 160, 160],
      tower_1_conv_3_shape =  [7, 1, 160, 160],
      tower_1_conv_4_shape =  [1, 7, 160, 192],
      tower_2_conv_shape =    [1, 1, 768, 192],
      cfg=cfg
    )

  # mixed_6
  with graph.name_scope("mixed"):
    mixed_6 = add_figure6(graph, mixed_5,
      conv_shape =            [1, 1, 768, 192],
      tower_conv_shape =      [1, 1, 768, 160],
      tower_conv_1_shape =    [1, 7, 160, 160],
      tower_conv_2_shape =    [7, 1, 160, 192],
      tower_1_conv_shape =    [1, 1, 768, 160],
      tower_1_conv_1_shape =  [7, 1, 160, 160],
      tower_1_conv_2_shape =  [1, 7, 160, 160],
      tower_1_conv_3_shape =  [7, 1, 160, 160],
      tower_1_conv_4_shape =  [1, 7, 160, 192],
      tower_2_conv_shape =    [1, 1, 768, 192],
      cfg=cfg
    )

  # mixed_7
  with graph.name_scope("mixed"):
    mixed_7 = add_figure6(graph, mixed_6,
      conv_shape =            [1, 1, 768, 192],
      tower_conv_shape =      [1, 1, 768, 192],
      tower_conv_1_shape =    [1, 7, 192, 192],
      tower_conv_2_shape =    [7, 1, 192, 192],
      tower_1_conv_shape =    [1, 1, 768, 192],
      tower_1_conv_1_shape =  [7, 1, 192, 192],
      tower_1_conv_2_shape =  [1, 7, 192, 192],
      tower_1_conv_3_shape =  [7, 1, 192, 192],
      tower_1_conv_4_shape =  [1, 7, 192, 192],
      tower_2_conv_shape =    [1, 1, 768, 192],
      cfg=cfg
    )

  # End second block of inception modules
  #################

  # Second Inception module for grid size reduction
  with graph.name_scope("mixed"):
    mixed_8 = add_figure10_2(graph, mixed_7, cfg=cfg)


  #################
  # Third block of inception modules
  # 2 modules as specified in Figure 7
  # mixed_9
  with graph.name_scope("mixed"):
    mixed_9 = add_figure7(graph, mixed_8,
      conv_shape                  = [1, 1, 1280, 320],
      tower_conv_shape            = [1, 1, 1280, 384],
      tower_mixed_conv_shape      = [1, 3, 384, 384],
      tower_mixed_conv_1_shape    = [3, 1, 384, 384],
      tower_1_conv_shape          = [1, 1, 1280, 448],
      tower_1_conv_1_shape        = [3, 3, 448, 384],
      tower_1_mixed_conv_shape    = [1, 3, 384, 384],
      tower_1_mixed_conv_1_shape  = [3, 1, 384, 384],
      tower_2_conv_shape          = [1, 1, 1280, 192],
      use_avg_pool = True,
      cfg=cfg
    )

  with graph.name_scope("mixed"):
    mixed_10 = add_figure7(graph, mixed_9,
      conv_shape                  = [1, 1, 2048, 320],
      tower_conv_shape            = [1, 1, 2048, 384],
      tower_mixed_conv_shape      = [1, 3, 384, 384],
      tower_mixed_conv_1_shape    = [3, 1, 384, 384],
      tower_1_conv_shape          = [1, 1, 2048, 448],
      tower_1_conv_1_shape        = [3, 3, 448, 384],
      tower_1_mixed_conv_shape    = [1, 3, 384, 384],
      tower_1_mixed_conv_1_shape  = [3, 1, 384, 384],
      tower_2_conv_shape          = [1, 1, 2048, 192],
      use_avg_pool = False, # use a max pool
      cfg=cfg
    )


  # Note that pool_2 was probably on an "alternate head"
  # pool_3
  pool_3 = add_avg_pool(
    graph=graph,
    input=mixed_10,
    ksize=[1, 8, 8, 1],
    strides=[1, 1, 1, 1],
    padding = "VALID",
    name="pool_3"
  )

  features = tf.reshape(pool_3, [-1, 2048], name='features')

  return features

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
