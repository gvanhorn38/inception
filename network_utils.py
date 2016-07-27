import tensorflow as tf

CONV_KERNEL_WD = 1e-5
BATCHNORM_GAMMA_WD = 1e-5
BATCHNORM_BETA_WD = 0
BATCHNORM_VARIANCE_EPSILON = 0.001
BATCHNORM_SCALE_AFTER_NORM = True
SOFTMAX_WD = 4e-5
SOFTMAX_BIAS_WD = 0
FC_WD = 4e-5
DROPOUT_KEEP_PROB = 0.3

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

def print_operations(graph):
  # Print out the basic input -> inference -> loss operations for the graph
  for op in graph.get_operations():
      print op.type.ljust(35), '\t', op.name

def print_trainable_variables(graph):
  
  for var in graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    print var.name.ljust(20), '\t', var.get_shape().as_list()