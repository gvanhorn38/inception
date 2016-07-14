"""
Export the classification model for use with TensorFlow Serving.
This assumes that you have installed the TensorFlow Serving python module.
"""
import argparse
from matplotlib import pyplot as plt
import os
import pprint
import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter

from config import parse_config_file
from network_utils import add_logits
import v3.classification.model as model

def export(model_path, export_path, export_version, cfg):
  
  graph = tf.get_default_graph()

  sess_config = tf.ConfigProto(
    log_device_placement=False,
    allow_soft_placement = True,
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=cfg.SESSION_CONFIG.PER_PROCESS_GPU_MEMORY_FRACTION
    )
  )
  
  # GVH: This needs to be moved into the inputs module
  #images = tf.placeholder(tf.float32, [None, None, None, 3], name="images")
  
  # Input transformation.
  # TODO(b/27776734): Add batching support.
  jpegs = tf.placeholder(tf.string, shape=(1))
  image_buffer = tf.squeeze(jpegs, [0])
  # Decode the string as an RGB JPEG.
  # Note that the resulting image contains an unknown height and width
  # that is set dynamically by decode_jpeg. In other words, the height
  # and width of image is unknown at compile-time.
  image = tf.image.decode_jpeg(image_buffer, channels=3)
  # After this point, all image pixels reside in [0,1)
  # until the very end, when they're rescaled to (-1, 1).  The various
  # adjust_* ops all require this range for dtype float.
  #image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  image = tf.cast(image, tf.float32)
  images = tf.expand_dims(image, 0)
  
  images = tf.image.resize_images(images, cfg.INPUT_SIZE, cfg.INPUT_SIZE)
  images -= cfg.IMAGE_MEAN
  images /= cfg.IMAGE_STD
  
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
    
    tf.initialize_all_variables().run()
    
    saver.restore(sess, model_path)

    export_saver = tf.train.Saver(sharded=True)
    model_exporter = exporter.Exporter(export_saver)
    signature = exporter.classification_signature(input_tensor=jpegs, scores_tensor=class_scores, classes_tensor=predicted_classes)
    model_exporter.init(sess.graph.as_graph_def(),
                        default_graph_signature=signature)
    model_exporter.export(export_path, tf.constant(export_version), sess)
    
    # TODO: Change to debug flag
    if False:
      plt.ion()
      while True:
        file_path = raw_input("File Path:")
        while not os.path.exists(file_path) or not os.path.isfile(file_path):
          file_path = raw_input("File Path:")
          
        with open(file_path, 'rb') as f:
          # See inception_inference.proto for gRPC request/response details.
          data = f.read()
        
        outputs = sess.run([class_scores, predicted_classes, image], {jpegs : [data]})
        
        print outputs[:2]
        img = outputs[2]
        plt.imshow(img)
        plt.show()
        t = raw_input("Press a button:")
        if t != '':
          break
    
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

    export(args.specific_model, args.export_path, args.export_version, cfg=cfg)
  