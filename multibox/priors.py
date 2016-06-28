"""
This file is used to generate the prior bounding boxes.
"""

import numpy as np
from matplotlib import pyplot as plt

def generate_priors():

  # Its important to be in descending order
  grids = [8, 6, 4, 3, 2, 1]

  min_scale = 0.1
  max_scale = 0.95
  num_scales = len(grids)
  scales = []
  for i in range(1, num_scales+1):
    scales.append(min_scale + (max_scale - min_scale) * (i - 1) / (num_scales - 1))

  aspect_ratios = [1, 2, 3, 1./2, 1./3] 

  prior_bboxes = []

  for k, (grid, scale) in enumerate(zip(grids, scales)):
    
    # special case for the 1x1 cell ( we only need one aspect ratio)
    if grid == 1:
      
      center_i = 0.5
      center_j = 0.5
      
      w = scale 
      h = scale

      x1 = center_j - (w / 2.)
      x2 = center_j + (w / 2.)
      y1 = center_i - (h / 2.)
      y2 = center_i + (h / 2.)

      prior_bboxes.append([
        max(x1, 0.),
        max(y1, 0.),
        min(x2, 1.),
        min(y2, 1.)
      ])
    else:
      for i in range(grid):
        for j in range(grid):
      
          center_i = (i + 0.5) / grid
          center_j = (j + 0.5) / grid
      
          for a in aspect_ratios:
            w = scale * np.sqrt(a)
            h = scale / np.sqrt(a)
        
            x1 = center_j - (w / 2.)
            x2 = center_j + (w / 2.)
            y1 = center_i - (h / 2.)
            y2 = center_i + (h / 2.)
        
            prior_bboxes.append([
              max(x1, 0.),
              max(y1, 0.),
              min(x2, 1.),
              min(y2, 1.)
            ])
            
            # Add another square box of a different size
            # GVH: Not sure if we actually need this
            if False:
              if a == 1:
                if k == num_scales - 1:
                  s = np.sqrt(scale * 1.1)
                else:
                  s = np.sqrt(scale * scales[k+1])
              
                w = s * np.sqrt(a)
                h = s / np.sqrt(a)
        
                x1 = center_j - (w / 2.)
                x2 = center_j + (w / 2.)
                y1 = center_i - (h / 2.)
                y2 = center_i + (h / 2.)
        
                prior_bboxes.append([
                  max(x1, 0.),
                  max(y1, 0.),
                  min(x2, 1.),
                  min(y2, 1.)
                ])
  
  return prior_bboxes
      
      
def PrintBox(loc, height, width, style='r-'):
    """A utility function to help visualizing boxes."""
    xmin, ymin, xmax, ymax = loc[0] * width, loc[1] * height, loc[2] * width, loc[3] * height 
    plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], style)


def test():
  
  prior_bboxes = generate_priors()
  
  p = '/Users/GVH/Downloads/1d7d4099e90a459ebc8b257c593e095e.jpg'
  from scipy.misc import imread
  from scipy.misc import imresize
  image = imread(p)
  ir = imresize(image, (299, 299, 3))
  plt.imshow(ir)

  num_bboxes_per_cell = len(aspect_ratios) #+ 1

  # One of the 8x8 cells
  offset = 8*4 + 4
  PrintBox(prior_bboxes[num_bboxes_per_cell*offset+ 0], 299, 299, 'r-')
  PrintBox(prior_bboxes[num_bboxes_per_cell*offset+ 1], 299, 299, 'b-')
  PrintBox(prior_bboxes[num_bboxes_per_cell*offset+ 2], 299, 299, 'g-')
  PrintBox(prior_bboxes[num_bboxes_per_cell*offset+ 3], 299, 299, 'c-')
  PrintBox(prior_bboxes[num_bboxes_per_cell*offset+ 4], 299, 299, 'm-')
  PrintBox(prior_bboxes[num_bboxes_per_cell*offset+ 5], 299, 299, 'y-')

  # One of the 6x6 cells
  offset = 8*8 + 6 * 3 + 2
  PrintBox(prior_bboxes[num_bboxes_per_cell*offset+ 0], 299, 299, 'r-')
  PrintBox(prior_bboxes[num_bboxes_per_cell*offset+ 1], 299, 299, 'b-')
  PrintBox(prior_bboxes[num_bboxes_per_cell*offset+ 2], 299, 299, 'g-')
  PrintBox(prior_bboxes[num_bboxes_per_cell*offset+ 3], 299, 299, 'c-')
  PrintBox(prior_bboxes[num_bboxes_per_cell*offset+ 4], 299, 299, 'm-')
  PrintBox(prior_bboxes[num_bboxes_per_cell*offset+ 5], 299, 299, 'y-')

  # One of the 4x4 cells
  offset = 8*8 + 6*6 + 4*2 + 2
  PrintBox(prior_bboxes[num_bboxes_per_cell*offset+ 0], 299, 299, 'r-')
  PrintBox(prior_bboxes[num_bboxes_per_cell*offset+ 1], 299, 299, 'b-')
  PrintBox(prior_bboxes[num_bboxes_per_cell*offset+ 2], 299, 299, 'g-')
  PrintBox(prior_bboxes[num_bboxes_per_cell*offset+ 3], 299, 299, 'c-')
  PrintBox(prior_bboxes[num_bboxes_per_cell*offset+ 4], 299, 299, 'm-')
  PrintBox(prior_bboxes[num_bboxes_per_cell*offset+ 5], 299, 299, 'y-')

  # One of the 3x3 cells
  offset = 8*8 + 6*6 + 4*4 + 3*1 + 1
  PrintBox(prior_bboxes[num_bboxes_per_cell*offset+ 0], 299, 299, 'r-')
  PrintBox(prior_bboxes[num_bboxes_per_cell*offset+ 1], 299, 299, 'b-')
  PrintBox(prior_bboxes[num_bboxes_per_cell*offset+ 2], 299, 299, 'g-')
  PrintBox(prior_bboxes[num_bboxes_per_cell*offset+ 3], 299, 299, 'c-')
  PrintBox(prior_bboxes[num_bboxes_per_cell*offset+ 4], 299, 299, 'm-')
  PrintBox(prior_bboxes[num_bboxes_per_cell*offset+ 5], 299, 299, 'y-')

  # One of the 2x2 cells
  offset = 8*8 + 6*6 + 4*4 + 3*3 + 2*1 + 0
  PrintBox(prior_bboxes[num_bboxes_per_cell*offset+ 0], 299, 299, 'r-')
  PrintBox(prior_bboxes[num_bboxes_per_cell*offset+ 1], 299, 299, 'b-')
  PrintBox(prior_bboxes[num_bboxes_per_cell*offset+ 2], 299, 299, 'g-')
  PrintBox(prior_bboxes[num_bboxes_per_cell*offset+ 3], 299, 299, 'c-')
  PrintBox(prior_bboxes[num_bboxes_per_cell*offset+ 4], 299, 299, 'm-')
  PrintBox(prior_bboxes[num_bboxes_per_cell*offset+ 5], 299, 299, 'y-')
      