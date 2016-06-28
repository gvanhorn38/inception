import numpy as np
import tensorflow as tf

import image_augmentations


def extract_bbox(image, bbox, random_expand=False):
  
  image_height, image_width = image.shape[:2]
  
  # the bbox should be in normalized coordinates
  x, y, w, h = bbox
  bbox_x = int(image_width * x) 
  bbox_y = int(image_height * y)
  bbox_width = int(image_width * w)
  bbox_height = int(image_height * h)
  
  # Basic protection
  bbox_x = max(0, bbox_x)
  bbox_y = max(0, bbox_y)
  bbox_x2 = min(image_width, bbox_x + bbox_width)
  bbox_y2 = min(image_height, bbox_y + bbox_height)
  
  if random_expand:
    
    # Sample new coordinates for the bbox
    bbox_x = np.random.randint(0, bbox_x + 1)
    bbox_y = np.random.randint(0, bbox_y + 1)
    bbox_x2 = np.random.randint(bbox_x2, image_width + 1)
    bbox_y2 = np.random.randint(bbox_y2, image_height + 1)
  
  bbox = image[bbox_y:bbox_y2, bbox_x:bbox_x2]
  return bbox
  


def input_nodes(
  # An array of paths to tfrecords files
  tfrecords,

  # Data augmentation depends on whether we are in train vs (test / eval) mode
  augment=True,

  # number of times to read the tfrecords
  num_epochs=None,

  # Data queue feeding the model
  batch_size=32,
  num_threads=2,
  shuffle_batch = True,
  capacity = 1000,
  min_after_dequeue = 96,

  # And tensorboard summaries of the images
  add_summaries=True,

  # Global configuration
  cfg=None):

  with tf.name_scope('inputs'):

    filename_queue = tf.train.string_input_producer(
      tfrecords,
      num_epochs=num_epochs
    )

    # Construct a Reader to read examples from the .tfrecords file
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # Parse an Example to access the Features
    features = tf.parse_single_example(
      serialized_example,
      features = {
        'path'  : tf.FixedLenFeature([], tf.string),
        'label' : tf.FixedLenFeature([], tf.int64),
        'bbox'  : tf.FixedLenFeature([4], tf.float32),
        'instance' : tf.FixedLenFeature([], tf.string)
      }
    )

  
    path = features['path']
    label = tf.cast(features['label'], tf.int32)
    bbox = features['bbox'] # [x, y, width, height]
    instance_id = features['instance']
    
    image = tf.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    label = tf.cast(features['label'], tf.int32)

    if add_summaries:
      tf.image_summary('orig_image', tf.expand_dims(image, 0))
    
    # Randomly expand the bounding box
    expand_bbox = False
    if 'BBOX_OPTIONS' in cfg:
      if 'RANDOM_EXPAND' in cfg.BBOX_OPTIONS and cfg.BBOX_OPTIONS.RANDOM_EXPAND:
        expand_bbox = True
    
    # Extract the bounding box
    bbox_data = tf.py_func(extract_bbox, [image, bbox, expand_bbox], [tf.float32], name="extract_bbox")
    image = bbox_data[0]
    
    if augment:
      image = image_augmentations.augment(image, cfg)
      image.set_shape([cfg.INPUT_SIZE, cfg.INPUT_SIZE, 3])
    else:
      if cfg.MAINTAIN_ASPECT_RATIO:
        # Resize the image up, then pad with 0s
        params = [image, tf.constant(cfg.INPUT_SIZE), tf.constant(cfg.INPUT_SIZE)]
        output = tf.py_func(image_augmentations.resize_image_maintain_aspect_ratio, params, [tf.float32], name="resize_maintain_aspect_ratio")
        image = output[0]
        image.set_shape([cfg.INPUT_SIZE, cfg.INPUT_SIZE, 3])

        #image = image_augmentations.resize_image_preserve_aspect_ratio(image, cfg.INPUT_SIZE, cfg.INPUT_SIZE)

      else:
        image.set_shape(tf.TensorShape([tf.Dimension(None), tf.Dimension(None), tf.Dimension(3)])) 
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [cfg.INPUT_SIZE, cfg.INPUT_SIZE])
        #image = tf.image.resize_images(image, cfg.INPUT_SIZE, cfg.INPUT_SIZE)
        image = tf.squeeze(image, [0])
  
    if add_summaries:
      tf.image_summary('train_image', tf.expand_dims(image, 0))

    image -= cfg.IMAGE_MEAN
    image /= cfg.IMAGE_STD

    # Place the images on another queue that will be sampled by the model
    if shuffle_batch:
      images, sparse_labels, paths, instance_ids = tf.train.shuffle_batch(
        [image, label, path, instance_id],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity= capacity, #batch_size * (num_threads + 2),
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue= min_after_dequeue, # 3 * batch_size,
        seed = cfg.RANDOM_SEED,
      )

    else:
      images, sparse_labels, paths, instance_ids = tf.train.batch(
        [image, label, path, instance_id],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity= capacity, #batch_size * (num_threads + 2),
        enqueue_many=False
      )

  # return a batch of images and their labels
  return images, sparse_labels, paths, instance_ids
  
  