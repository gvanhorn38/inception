
import whole_image

WHOLE_IMAGE_INPUT = 'whole_image_input' # Pass the entire image in 
BBOX_INPUT = 'bbox_input' # Extract the bounding box
PART_REGION_INPUT = 'part_region_input' # Extract a pose based on the parts
PATCH_REGION_INPUT = 'patch_region_input' # Extract a patch around specific parts

def construct_network_input_nodes(
    tfrecords, **kwargs
    # input_type=WHOLE_IMAGE_INPUT, 
#     num_epochs=None, 
#     batch_size=32, 
#     num_threads=2,
#     shuffle=True,
#     add_summaries=True, 
#     augment=True,
#     cfg=None
  ):

  if input_type == WHOLE_IMAGE_INPUT:
    return whole_image.input_nodes(
      tfrecords=tfrecords, **kwargs
      # augment=augment,
#       num_epochs=num_epochs, 
#       batch_size=batch_size, 
#       num_threads=num_threads,
#       shuffle = shuffle, 
#       capacity = 128,
#       min_after_dequeue = 96,
#   
#       # And tensorboard summaries of the images
#       add_summaries=True, 
#   
#       # Global configuration
#       cfg=None
    )
  else:
    raise ValueError('unknown input type')