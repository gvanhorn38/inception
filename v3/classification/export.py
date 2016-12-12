"""
Export the classification model for use with TensorFlow Serving.

To use this script, the tensorflow.contrib.session_bundle.exporter module needs to 
be able to import the manifest.proto protocol buffer. Currently, this protocol buffer is
not available unless you compile it yourself:

PYTHON = <path to your python installation>
SERVING = <path to your TensorFlow Serving github repo directory>
cd $PYTHON/site-packages/tensorflow/contrib/session_bundle
protoc -I=$SERVING/tensorflow/tensorflow/contrib/session_bundle \
--python_out='.' $SERVING/tensorflow/tensorflow/contrib/session_bundle/manifest.proto

"""

import argparse
from matplotlib import pyplot as plt
import os
import pprint
import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter
from tensorflow.python.framework import graph_util

from config import parse_config_file
from network_utils import add_logits
import v3.classification.model as model

def export(model_path, export_path, export_version, export_for_serving, cfg):
  
  graph = tf.get_default_graph()

  sess_config = tf.ConfigProto(
    log_device_placement=False,
    allow_soft_placement = True,
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=cfg.SESSION_CONFIG.PER_PROCESS_GPU_MEMORY_FRACTION
    )
  )
  
  # GVH: This is a little tricky. 
  #   tf.image.decode_jpeg does not have a batch implementation, creating a bottleneck
  #   for batching. We can request the user to send in a raveled image, but this will 
  #   increase our transport size over the network. Also, should we assume that the images
  #   have been completely preprocessed by the user? (mean subtracted, scaled by std, etc?)
  
  # GVH: We could just make this a switch and let the user decide what to do.
  
  # JPEG bytes:
#   jpegs = tf.placeholder(tf.string, shape=(1))
#   image_buffer = tf.squeeze(jpegs, [0])
#   image = tf.image.decode_jpeg(image_buffer, channels=3)
#   image = tf.cast(image, tf.float32)
#   images = tf.expand_dims(image, 0)
#   images = tf.image.resize_images(images, cfg.INPUT_SIZE, cfg.INPUT_SIZE)
#   images -= cfg.IMAGE_MEAN
#   images /= cfg.IMAGE_STD
  
  # For now we'll assume that the user is sending us a raveled array, totally preprocessed. 
  image_data = tf.placeholder(tf.float32, [None, cfg.INPUT_SIZE * cfg.INPUT_SIZE * 3], name="images")
  images = tf.reshape(image_data, [-1, cfg.INPUT_SIZE, cfg.INPUT_SIZE, 3])
  
  
  features = model.build(graph, images, cfg)
  logits = add_logits(graph, features, cfg.NUM_CLASSES)
  class_scores, predicted_classes = tf.nn.top_k(logits, k=cfg.NUM_CLASSES)
  
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

  # Restore the variables
  saver = tf.train.Saver(shadow_vars, reshape=True)
  
  with tf.Session(graph=graph, config=sess_config) as sess:
    
    tf.global_variables_initializer()
    
    saver.restore(sess, model_path)

    # TODO: Change to options flag
    if export_for_serving:
      export_saver = tf.train.Saver(sharded=True)
      model_exporter = exporter.Exporter(export_saver)
      signature = exporter.classification_signature(input_tensor=image_data, scores_tensor=class_scores, classes_tensor=predicted_classes)
      model_exporter.init(sess.graph.as_graph_def(),
                          default_graph_signature=signature)
      model_exporter.export(export_path, tf.constant(export_version), sess)
    
    else:
      v2c = graph_util.convert_variables_to_constants
      deploy_graph_def = v2c(sess, graph.as_graph_def(), [logits.name[:-2]])
    
      if not os.path.exists(export_path):
          os.makedirs(export_path)
      save_path = os.path.join(export_path, 'constant_model-%d.pb' % (export_version,))
      with open(save_path, 'wb') as f:
          f.write(deploy_graph_def.SerializeToString())
    
def parse_args():

    parser = argparse.ArgumentParser(description='Test an Inception V3 network')

    parser.add_argument('--model', dest='specific_model',
                          help='Path to the specific model you want to export.',
                          required=True, type=str)

    parser.add_argument('--export_path', dest='export_path',
                          help='Path to a directory where the exported model will be saved.',
                          required=True, type=str)
    
    parser.add_argument('--export_version', dest='export_version',
                        help='Version number of the model.',
                        required=True, type=int)
    
    parser.add_argument('--config', dest='config_file',
                        help='Path to the configuration file',
                        required=True, type=str)
    
    parser.add_argument('--serving', dest='serving',
                        help='Export for TensorFlow Serving usage. Otherwise, a constant graph will be generated.',
                        action='store_true', default=False)
    

    args = parser.parse_args()
    
    return args

if __name__ == '__main__':

    args = parse_args()
    print "Called with:"
    print pprint.pprint(args)

    cfg = parse_config_file(args.config_file)
    
    # We want to use the global statistics for testing
    cfg.USE_BATCH_STATISTICS = False

    print "Configurations:"
    print pprint.pprint(cfg)

    export(args.specific_model, args.export_path, args.export_version, args.serving, cfg=cfg)
  
