"""
These are utility functions for doing dense prediction of an image. 

We want to take multiple crops from the image and process them.

We want to take the union of all proposals, and then do some non-max-suppression.
"""
from matplotlib import pyplot as plt
import numpy as np


def extract_patches(image, patch_dims, strides):
  """
  Args:
    image (np.array) : the image to extract the patches from
    patch_dims (tuple) : the (height, width) size of the patch to extract from the image (assumed to be square)
    strides (tuple) : the (y, x) stride of the patches (in height and width)
    
  Returns:
    list : the patches 
    list : offsets for each patch 
  """
  
  image_height, image_width = image.shape[:2]
  patch_height, patch_width = patch_dims
  h_stride, w_stride = strides
  
  patches = []
  patch_offsets = []
  
  for h in range(0,image_height-patch_height+1,h_stride):
    for w in range(0,image_width-patch_width+1,w_stride):
      
      p = image[h:h+patch_height, w:w+patch_width]
      patches.append(p)
      patch_offsets.append((h, w))
  
  return patches, patch_offsets
      
def filter_proposals(bboxes, confidences):
  """We want to filter out proposals that are not completely contained in the square [.1, .1, .9, .9]
  
  Args: 
    bboxes np.array: proposed bboxes [x1, y1, x2, y2]
    confidences np.array: confidences for the proposed boxes
  
  Returns:
    np.array : the filtered bboxes
    np.array : the confidences for the bboxes
  """
  
  filtered_bboxes = []
  filtered_confidences = []
  
  for bbox, conf in zip(bboxes, confidences):
    if np.any(bbox < .1) or np.any(bbox > .9):
      continue
    filtered_bboxes.append(bbox)
    filtered_confidences.append(conf)
  
  return np.array(filtered_bboxes), np.array(filtered_confidences)

def convert_proposals(bboxes, offset, patch_dims, image_dims):
  """Convert the coordinates of the proposed bboxes to account for the offset of the patch
  
  Args:
    bboxes (np.array) : the proposed bboxes [x1, y1, x2, y2]
    offset (tuple) : the (y, x) offset of the patch in relation to the image
    patch_dims (tuple) : the (height, width) dimensions of the patch
    image_dims (tuple) : the (height, width) dimensions of the image
  Returns:
    np.array : the converted bounding boxes
  """
  
  x_scale = patch_dims[1] / float(image_dims[1])
  y_scale = patch_dims[0] / float(image_dims[0])
  
  x_offset = offset[1] / float(image_dims[1])
  y_offset = offset[0] / float(image_dims[0])
  
  converted_bboxes = bboxes * np.array([x_scale, y_scale, x_scale, y_scale]) + np.array([x_offset, y_offset, x_offset, y_offset])
  
  return converted_bboxes

def intersection_over_union(bbox1, bbox2):
   
  bbox1_xmin, bbox1_ymin, bbox1_xmax, bbox1_ymax = bbox1
  bbox2_xmin, bbox2_ymin, bbox2_xmax, bbox2_ymax = bbox2

  x1 = max(bbox1_xmin, bbox2_xmin)
  y1 = max(bbox1_ymin, bbox2_ymin)
  x2 = min(bbox1_xmax, bbox2_xmax)
  y2 = min(bbox1_ymax, bbox2_ymax)

  w = max(0, x2 - x1)
  h = max(0, y2 - y1)

  intersection = w * h
  bbox1_area = (bbox1_xmax - bbox1_xmin) * (bbox1_ymax - bbox1_ymin)
  bbox2_area = (bbox2_xmax - bbox2_xmin) * (bbox2_ymax - bbox2_ymin)
        
  iou = intersection / (1.0 * bbox1_area + bbox2_area - intersection)

  return iou

# GVH: This could use a speed up  
def non_max_suppression(bboxes, confs, jaccard_threshold=0.85):
  """Perform non-max suppression on the bboxes, and return the filtered bboxes and confs
  
  What do we want to do about the confidences? For boxes that get merged, should we take 
  the most confident prediction? 
  
  Should we do filtering on the confidences first? 
  
  Args:
    bboxes (np.array) : [[x1, y1, x2, y2]] bounding boxes
    confs (np.array) : The confidences for the bounding boxes
    jaccard_threshold : overlap threshold, in the range [0, 1]
  
  Returns:
    np.array : the filtered bboxes
    np.array : the filtered confidences
  """
  if len(bboxes) == 0:
    return bboxes, confs
    
  
  # We need an order to process the bounding boxes, we'll choose the lower y point
  idxs = np.argsort(bboxes[:,3]).tolist()
  
  # These are the indices of the bounding boxes that we have chosen
  selected_indices = []
  
  while len(idxs) > 0:
    
    last = len(idxs) - 1
    i = idxs[last]
    
    current_bbox = bboxes[i]
    indices_to_merge = [i]
    
    for pos in xrange(0, last):
  
      j = idxs[pos]
      test_bbox = bboxes[j]
      
      iou = intersection_over_union(current_bbox, test_bbox)

      if iou > jaccard_threshold:
        indices_to_merge.append(j)
    
    if len(indices_to_merge) == 1:
      selected_indices.append(indices_to_merge[0])
    else:
      # We have multiple bounding boxes, and we only want to keep one of them.
      # We want to keep the box with the largest confidence
      conf_indx = np.argmax(confs[indices_to_merge])
      indx_to_keep = indices_to_merge[conf_indx]
      selected_indices.append(indx_to_keep)
    
    for idx in indices_to_merge:  
      idxs.remove(idx)
  
  return bboxes[selected_indices], confs[selected_indices]
  
def test():
  
  image_dims = (600, 800)
  image = np.zeros([image_dims[0], image_dims[1], 3])
  
  offset = (299, 299)
  patch_dims = (299, 299)
  
  image[offset[0]: offset[0] + patch_dims[0], offset[1]: offset[1] + patch_dims[1], :] = 1
  
  pred_bboxes = np.array([[.1, .1, .9, .9]])
  pred_confs = np.array([.9])
  
  conv_bboxes = convert_proposals(pred_bboxes, offset, patch_dims, image_dims)
  
  plt.imshow(image)
  
  xmin, ymin, xmax, ymax = conv_bboxes[0] * np.array([image_dims[1], image_dims[0], image_dims[1], image_dims[0]])
  plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'b-')
  
  plt.show()
  
