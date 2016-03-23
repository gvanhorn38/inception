"""
This file will dump the inception network to a few formats. Basically this will convert
all of the constants into Variables for the trainable parameters
"""
import os
import sys
import numpy as np
import tensorflow as tf

INCEPTION_V3_IMAGE_SIZE = 299
INCEPTION_V3_IMAGE_MEAN = 128
INCEPTION_V3_IMAGE_STD = 128

def _build_inference_nodes(images, graph, sess_config):
  """
  Add the inference nodes from the inception architecture to the graph.

  NOTE: Variables() default to being trainable

  """

  # we will replace this op with our images
  input_op = graph.get_operation_by_name('Mul')

  # Several options here:
  # pool_3 : raw 2048 features
  # pool_3/_reshape raw 2048 features reshaped for softmax computation
  # softmax : softmax weights
  features_op = graph.get_operation_by_name('pool_3/_reshape')

  ops = graph.get_operations()

  # Find the operations that correspond to the convolutional weights
  # *_params for the convolutional kernels
  # weights for the softmax weights
  weights_ops = [op for op in ops if 'params' in op.name]# or 'weights' in op.name]

  # Find the operations that correspond to the batch normalization biases
  # 'biases' for the softmax bias
  bias_ops = [op for op in ops if 'beta' in op.name or 'gamma' in op.name]# or 'biases' in op.name]

  # Grab the remaining operations for reuse
  reuse_ops = graph.get_operations()
  image_input_index = reuse_ops.index(input_op)
  feature_output_index = reuse_ops.index(features_op)
  reuse_ops = reuse_ops[image_input_index+1:feature_output_index+1]
  # Filter out the weight ops and the
  # batch normalization ops (both the bias and the moving mean / variance)
  reuse_ops = [
    op for op in reuse_ops if (op not in weights_ops) and
                              (op not in bias_ops) and
                              ('moving' not in op.name)
  ]

  # Pull out the current values of the weights and bias
  with tf.Session(graph=graph, config=sess_config):
    weights_orig = dict([(op.name, op.outputs[0].eval()) for op in weights_ops])
    bias_orig = dict([(op.name, op.outputs[0].eval()) for op in bias_ops])

  # T will hold all of the new tensor Variables.
  T = {
    name : tf.Variable(
                initial_value = value,
                name = name,
                collections=['variables', 'weights'] # why weights and not trainable_variables or ...?
          )
    for name, value in weights_orig.iteritems()
  }
  T.update({
    name : tf.Variable(value, name=name)
    for name, value in bias_orig.iteritems()
  })

  # Add in our custom image node
  T[input_op.name] = images

  # Now add in the the remaining nodes of the graph
  for op in reuse_ops:

    # If this is a batch normalization layer, then use our new variables
    # Note: Getting rid of the moving mean and variance will affect the classification
    # performance. You wont ge the same results without them.
    if op.type == 'BatchNormWithGlobalNormalization':
      # t == tensor output of convolution
      t, beta, gamma = [T[op.inputs[i].op.name] for i in [0, 3, 4]]
      mu, var = tf.nn.moments(t, [0, 1, 2], name=op.name+'/')

      T[op.name] = tf.nn.batch_norm_with_global_normalization(
        t = t, m=mu, v=var, beta=beta, gamma=gamma,
        variance_epsilon=1e-8, scale_after_normalization=True,
        name = op.name
      )

    # copy the operation into the graph
    else:
      copied_op = images.graph.create_op(
        op_type = op.type,
        # Convert the Variables to Tensors
        inputs = [tf.convert_to_tensor(T[t.op.name]) for t in op.inputs],
        dtypes = [o.dtype for o in op.outputs],
        name = op.name,
        attrs = op.node_def.attr
      )

      T[op.name] = copied_op.outputs[0]

def dump(output_dir):

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  graph = tf.Graph() #tf.get_default_graph()
  graph_def = tf.GraphDef()
  graph_def_file = '../inception_v3/inception_dec_2015/tensorflow_inception_graph.pb'

  # Read in the graph definition
  with tf.gfile.FastGFile(graph_def_file, 'rb') as f:
    graph_def.ParseFromString(f.read())

  # Apply the graph definition to the graph
  with graph.as_default():
    tf.import_graph_def(graph_def, name='')

  # Save off a text format version
  tf.train.write_graph(graph_def, output_dir, 'const_inception_graph.pb', as_text=False)
  tf.train.write_graph(graph_def, output_dir, 'const_inception_graph.pbtxt', as_text=True)

  # we can test the graph:
  print "Testing the original inception network:"
  #image_data = tf.gfile.FastGFile("/Users/GVH/Desktop/cropped_panda.jpg", 'rb').read()
  with tf.Session(graph=graph) as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    predictions = sess.run(softmax_tensor)#,
                           #{'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)

    top_k = predictions.argsort()[-5:][::-1]
    for idx in top_k:
      print "%d : %0.5f" % (idx, predictions[idx])


  # construct a placeholder for the images
  #with graph.as_default():
  #images = tf.placeholder(tf.float32, name="images")

  if True:
    image_path = tf.placeholder(tf.string, shape=[], name='image_path')#"/Users/GVH/Desktop/cropped_panda.jpg",
    image = tf.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    images = tf.expand_dims(image, 0)
    images = tf.image.resize_images(images, INCEPTION_V3_IMAGE_SIZE, INCEPTION_V3_IMAGE_SIZE)
    images -= INCEPTION_V3_IMAGE_MEAN
    images /= INCEPTION_V3_IMAGE_STD
    batch = images
  else:
    batch = tf.zeros(shape=[1, INCEPTION_V3_IMAGE_SIZE, INCEPTION_V3_IMAGE_SIZE, 3], name="images")

  # Convert the constants to variables
  _build_inference_nodes(batch, graph, None)

  #for var in batch.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
  #  print var.name

  default_graph = batch.graph

  tf.train.write_graph(default_graph.as_graph_def(), output_dir, 'var_inception_graph.pb', as_text=False)
  tf.train.write_graph(default_graph.as_graph_def(), output_dir, 'var_inception_graph.pbtxt', as_text=True)

  assign_ops = []
  for op in default_graph.get_operations():
     if op.type == 'Assign':
        assign_ops.append(op)


  saver = tf.train.Saver(default_graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
  checkpoint_dir = os.path.join(output_dir, 'initial_checkpoint')
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

  #summary_dir = os.path.join(output_dir, 'summary')
  #if not os.path.exists(summary_dir):
  #  os.makedirs(summary_dir)
  #summary = tf.train.SummaryWriter(logdir=summary_dir)

  with default_graph.as_default():
    with tf.Session(graph=default_graph) as sess:

      tf.initialize_all_variables().run()

      #sess.run(assign_ops)

      # Only run the test if you have extracted the softmax weight too
      if False:
        softmax = default_graph.get_operation_by_name('softmax').outputs[0]

        predictions = sess.run(softmax, feed_dict={image_path : "/Users/GVH/Desktop/cropped_panda.jpg"})

        predictions = np.squeeze(predictions)

        print predictions.shape

        top_k = predictions.argsort()[-5:][::-1]
        for idx in top_k:
          print "%d : %0.5f" % (idx, predictions[idx])

      # Create something that we can restore from
      saver.save(
        sess=sess,
        save_path= os.path.join(checkpoint_dir, 'original_inception_v3_with_variables'),
      )

      #graph_def = sess.graph.as_graph_def(add_shapes=True)
      #summary.add_graph(graph_def)

  #summary.flush()
  #summary.close()

if __name__ == '__main__':

  dump(sys.argv[1])
