import numpy as np
import train

batch_size = 1
num_predictions = 5
max_num_gt_bboxes = 2
locations = np.zeros([batch_size * num_predictions, 4])
confidences = np.zeros([batch_size * num_predictions])
gt_bboxes = np.zeros([batch_size, max_num_gt_bboxes, 4])
num_gt_bboxes = np.zeros([batch_size]).astype(int)

locations[0] = np.array([0.1, 0.1, 0.3, 0.3])
locations[1] = np.array([0.2, 0.3, 0.4, 0.4])
locations[2] = np.array([0.5, 0.4, 0.6, 0.6])
locations[3] = np.array([0.7, 0.8, 0.9, 0.9])
locations[4] = np.array([0.8, 0.3, 0.95, 0.5])

confidences[0] = .7
confidences[1] = .8
confidences[2] = .6
confidences[3] = .3
confidences[4] = .7

gt_bboxes[0][0] = np.array([0.25, 0.25, 0.35, 0.35])
gt_bboxes[0][1] = np.array([0.77, 0.33, 0.9, 0.53])
num_gt_bboxes[0] = 2

partitions, stacked_gt_bboxes = train.compute_assignments(locations, confidences, gt_bboxes, num_gt_bboxes, batch_size)


##############################

batch_size = 2
num_predictions = 5
max_num_gt_bboxes = 1
locations = np.zeros([batch_size * num_predictions, 4])
confidences = np.zeros([batch_size * num_predictions])
gt_bboxes = np.zeros([batch_size, max_num_gt_bboxes, 4])
num_gt_bboxes = np.zeros([batch_size]).astype(int)

locations[0] = np.array([0.1, 0.1, 0.3, 0.3])
locations[1] = np.array([0.2, 0.3, 0.4, 0.4])
locations[2] = np.array([0.5, 0.4, 0.6, 0.6])
locations[3] = np.array([0.7, 0.8, 0.9, 0.9])
locations[4] = np.array([0.8, 0.3, 0.95, 0.5])

locations[5] = np.array([0.1, 0.1, 0.3, 0.3])
locations[6] = np.array([0.2, 0.3, 0.4, 0.4])
locations[7] = np.array([0.5, 0.4, 0.6, 0.6])
locations[8] = np.array([0.7, 0.8, 0.9, 0.9])
locations[9] = np.array([0.8, 0.3, 0.95, 0.5])

confidences[0] = .7
confidences[1] = .8
confidences[2] = .6
confidences[3] = .3
confidences[4] = .7

confidences[5] = .7
confidences[6] = .8
confidences[7] = .6
confidences[8] = .3
confidences[9] = .7

gt_bboxes[0][0] = np.array([0.25, 0.25, 0.35, 0.35])
gt_bboxes[1][0] = np.array([0.77, 0.33, 0.9, 0.53])
num_gt_bboxes[0] = 1
num_gt_bboxes[1] = 1

partitions, stacked_gt_bboxes = train.compute_assignments(locations, confidences, gt_bboxes, num_gt_bboxes, batch_size)


[21792, 4]
[21792]
[32, 1, 4]
[32]


c = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
x = np.array(
    [[1, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 1]]
   )
t = 0
for i in range(5):
  t += (1 - np.sum(x[i])) * np.log(1 - c[i])
t *= -1

print t

r = 0
for i in range(5):
  for j in range(2):
    r += x[i][j] * np.log(1. - c[i]) - (1. / 2.) * np.log(1 - c[i]) 
    
print r