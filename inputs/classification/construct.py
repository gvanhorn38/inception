"""
This is a wrapper file for bringing in the bounding box and whole image input pipelines.
"""

import bbox
import whole_image

WHOLE_IMAGE_INPUT = 'whole_image_input' # Pass the entire image in 
BBOX_INPUT = 'bbox_input' # Extract the bounding box
PART_REGION_INPUT = 'part_region_input' # Extract a pose based on the parts
PATCH_REGION_INPUT = 'patch_region_input' # Extract a patch around specific parts

def construct_network_input_nodes(
    tfrecords, 
    input_type=WHOLE_IMAGE_INPUT, 
    **kwargs
  ):

  if input_type == WHOLE_IMAGE_INPUT:
    return whole_image.input_nodes(tfrecords=tfrecords, **kwargs)
  elif input_type == BBOX_INPUT:
    return bbox.input_nodes(tfrecords=tfrecords, **kwargs)
  else:
    raise ValueError('unknown input type')