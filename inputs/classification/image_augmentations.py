import numpy as np
from scipy.misc import imresize
import tensorflow as tf
  
def resize_image_maintain_aspect_ratio(image, target_height, target_width):
  """
  image needs to be RGB image that has been casted to float32
  """
  
  height, width, _ = image.shape
  
  if height > width:
    new_height = target_height
    factor = new_height / float(height)
    new_width = int(np.round(width * factor))
  else:
    new_width = target_width
    factor = new_width / float(width)
    new_height = int(np.round(height * factor))
  
  resized_image = imresize(
    image,
    (new_height, new_width)
  )

  output = np.zeros((target_height, target_width, 3), dtype=np.float32)
  output[:new_height, :new_width, :] = resized_image
  
  return output  

def augment(image, cfg):

  options = cfg.IMAGE_AUGMENTATIONS
  
  if options.FLIP_LEFT_RIGHT:
    image = tf.image.random_flip_left_right(image)

  if options.CROP:
    
    # We want the size to be larger, and then will crop a region out of it
    target_size = tf.to_int32(cfg.INPUT_SIZE * options.CROP_UPSCALE_FACTOR)
    
    if cfg.MAINTAIN_ASPECT_RATIO:
      # Resize the image up, then pad with 0s
      #image = resize_image_preserve_aspect_ratio(image, target_size, target_size)
      
      params = [image, target_size, target_size]
      output = tf.py_func(resize_image_maintain_aspect_ratio, params, [tf.float32], name="resize_maintain_aspect_ratio")
      image = output[0]
      

    else:
      # Just resize it
      image = tf.image.resize_images(
        image,
        (target_size,
        target_size)
      )
    
    image = tf.random_crop(image, [cfg.INPUT_SIZE, cfg.INPUT_SIZE, 3])
  
  else:
    # Just resize it
    if cfg.MAINTAIN_ASPECT_RATIO:
      # Resize the image up, then pad with 0s
      #image = resize_image_preserve_aspect_ratio(image, target_size, target_size)
      
      params = [image, tf.constant(cfg.INPUT_SIZE), tf.constant(cfg.INPUT_SIZE)]
      output = tf.py_func(resize_image_maintain_aspect_ratio, params, [tf.float32], name="resize_maintain_aspect_ratio")
      image = output[0]
    else:
      image = tf.image.resize_images(
        image,
        cfg.INPUT_SIZE,
        cfg.INPUT_SIZE
      )
  
  if options.BRIGHTNESS:
    image = tf.image.random_brightness(image, max_delta=63)

  if options.CONTRAST:
    image = tf.image.random_contrast(image, lower=0.2, upper=1.8)

  return image