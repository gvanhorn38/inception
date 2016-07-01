"""
Feed the whole image to the network.
"""

import tensorflow as tf

import image_augmentations

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
  """
  Whole image

  tfrecords : an array of paths to .tfrecords files containing the Example protobufs

  """

  with tf.name_scope('inputs'):

    # Have a queue that produces the paths to the .tfrecords
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
      }
    )

    path = features['path']
    image = tf.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    label = tf.cast(features['label'], tf.int32)

    if add_summaries:
      tf.image_summary('orig_image', tf.expand_dims(image, 0))

    if augment:
      image = image_augmentations.augment(image, cfg)
    else:
      if cfg.MAINTAIN_ASPECT_RATIO:
        # Resize the image up, then pad with 0s
        params = [image, tf.constant(cfg.INPUT_SIZE), tf.constant(cfg.INPUT_SIZE)]
        output = tf.py_func(image_augmentations.resize_image_maintain_aspect_ratio, params, [tf.float32], name="resize_maintain_aspect_ratio")
        image = output[0]
        image.set_shape([cfg.INPUT_SIZE, cfg.INPUT_SIZE, 3])

      else:
        image = tf.image.resize_images(image, cfg.INPUT_SIZE, cfg.INPUT_SIZE)

    if add_summaries:
      tf.image_summary('train_image', tf.expand_dims(image, 0))

    image -= cfg.IMAGE_MEAN
    image /= cfg.IMAGE_STD

    # Place the images on another queue that will be sampled by the model
    if shuffle_batch:
      images, sparse_labels, paths = tf.train.shuffle_batch(
        [image, label, path],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity= capacity, #batch_size * (num_threads + 2),
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue= min_after_dequeue, # 3 * batch_size,
        seed = cfg.RANDOM_SEED,
      )

    else:
      images, sparse_labels, paths = tf.train.batch(
        [image, label, path],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity= capacity, #batch_size * (num_threads + 2),
        enqueue_many=False
      )

  # return a batch of images and their labels
  return images, sparse_labels, paths, None
