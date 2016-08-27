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

# Malisiewicz et al.  
# http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
def non_max_suppression(bboxes, confs, jaccard_threshold=0.85):
  """Perform non-max suppression on the bboxes, and return the filtered bboxes and confs
  
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

  # This list will contain the indices of the bboxes we will keep
  pick = []

  # grab the coordinates of the bounding boxes
  x1 = bboxes[:,0]
  y1 = bboxes[:,1]
  x2 = bboxes[:,2]
  y2 = bboxes[:,3]

  # compute the area of the bounding boxes and sort the bounding
  # boxes by the bottom-right y-coordinate of the bounding box
  # This means that we will keep the bounding box with the largest bottom-right y coord!
  area = (x2 - x1 + 1) * (y2 - y1 + 1)
  idxs = np.argsort(y2)

  # keep looping while some indexes still remain in the indexes
  # list
  while len(idxs) > 0:
    # grab the last index in the indexes list and add the
    # index value to the list of picked indexes
    last = len(idxs) - 1
    i = idxs[last]
    pick.append(i)

    # find the largest (x, y) coordinates for the start of
    # the bounding box and the smallest (x, y) coordinates
    # for the end of the bounding box
    xx1 = np.maximum(x1[i], x1[idxs[:last]])
    yy1 = np.maximum(y1[i], y1[idxs[:last]])
    xx2 = np.minimum(x2[i], x2[idxs[:last]])
    yy2 = np.minimum(y2[i], y2[idxs[:last]])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    # compute the ratio of overlap
    overlap = (w * h) / area[idxs[:last]]

    # delete all indexes from the index list that have
    idxs = np.delete(idxs, np.concatenate(([last],
      np.where(overlap > jaccard_threshold)[0])))

  # return only the bounding boxes that were picked
  return bboxes[pick], confs[pick]
  
  

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
  
