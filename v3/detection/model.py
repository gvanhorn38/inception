import numpy as np
from scipy.optimize import linear_sum_assignment
import tensorflow as tf

from network_utils import add_avg_pool, add_conv
import v3.model as v3_model

small_epsilon = 1e-10

def compute_assignments(locations, confidences, gt_bboxes, num_gt_bboxes, batch_size, alpha):
  """
  locations: [batch_size * num_predictions, 4]
  confidences: [batch_size * num_predictions]
  gt_bboxes: [batch_size, max num gt_bboxes, 4]
  num_gt_bboxes : [batch_size]  The number of gt bboxes in each image of the batch
  """
  
  num_predictions = locations.shape[0] / batch_size
  assignment_partitions = np.zeros(batch_size * num_predictions, dtype=np.int32)
  stacked_gt_bboxes = []
  
  log_confidences = np.log(confidences)
  v = 1. - confidences
  v[v > 1.] = 1.
  v[v <= 0] = small_epsilon
  log_one_minus_confidences = np.log(v)
  
  # Go through each image in the batch
  for b in range(batch_size):
    
    offset = b * num_predictions
    
    # we need to construct the cost matrix
    C = np.zeros((num_predictions, num_gt_bboxes[b]))
    for j in range(num_gt_bboxes[b]):
      C[:, j] = (alpha / 2.) * (np.linalg.norm(locations[offset:offset+num_predictions] - gt_bboxes[b][j], axis=1))**2 - log_confidences[offset:offset+num_predictions] + log_one_minus_confidences[offset:offset+num_predictions]
    
    #print C
    
    # Compute the assignments
    row_ind, col_ind = linear_sum_assignment(C)
    
    #print row_ind, col_ind
    
    for r, c in zip(row_ind, col_ind):
      assignment_partitions[offset + r] = 1
      stacked_gt_bboxes.append(gt_bboxes[b][c])
    
  return [assignment_partitions, np.array(stacked_gt_bboxes)]


def add_detection_heads(graph, inputs, num_bboxes_per_cell, batch_size, cfg):
  
  with graph.name_scope("detection"):
    
    # 8 x 8 grid cells
    with graph.name_scope("8x8"):
    
      detect_8_conv = add_conv(
        graph = graph,
        input = inputs,
        shape = [1, 1, 2048, 96],
        strides = [1, 1, 1, 1],
        padding = "SAME",
        use_batch_statistics=cfg.USE_BATCH_STATISTICS
      )
    
      detect_8_conv_1 = add_conv(
        graph = graph,
        input = detect_8_conv,
        shape = [3, 3, 96, 96],
        strides = [1, 1, 1, 1],
        padding = "SAME",
        use_batch_statistics=cfg.USE_BATCH_STATISTICS
      )
    
      detect_8_locations = add_conv(
        graph = graph,
        input = detect_8_conv_1,
        shape = [1, 1, 96, num_bboxes_per_cell * 4],
        strides = [1, 1, 1, 1],
        padding = "SAME",
        use_batch_statistics=cfg.USE_BATCH_STATISTICS,
        batch_normalize=False,
        add_relu = False
      )
      detect_8_locations = tf.reshape(detect_8_locations, [batch_size, -1])
      
      detect_8_confidences = add_conv(
        graph = graph,
        input = detect_8_conv_1,
        shape = [1, 1, 96, num_bboxes_per_cell],
        strides = [1, 1, 1, 1],
        padding = "SAME",
        use_batch_statistics=cfg.USE_BATCH_STATISTICS,
        batch_normalize=False,
        add_relu = False
      )
      detect_8_confidences = tf.reshape(detect_8_confidences, [batch_size, -1])
    
    # 6 x 6 grid cells
    with graph.name_scope("6x6"):
      
      detect_6_conv = add_conv(
        graph = graph,
        input = inputs,
        shape = [3, 3, 2048, 96],
        strides = [1, 1, 1, 1],
        padding = "SAME",
        use_batch_statistics=cfg.USE_BATCH_STATISTICS
      )
    
      detect_6_conv_1 = add_conv(
        graph = graph,
        input = detect_6_conv,
        shape = [3, 3, 96, 96],
        strides = [1, 1, 1, 1],
        padding = "VALID",
        use_batch_statistics=cfg.USE_BATCH_STATISTICS
      )
      
      #print "Detect 6 Conv 1:"
      #print detect_6_conv_1.get_shape().as_list()
      
      detect_6_locations = add_conv(
        graph = graph,
        input = detect_6_conv_1,
        shape = [1, 1, 96, num_bboxes_per_cell * 4],
        strides = [1, 1, 1, 1],
        padding = "SAME",
        use_batch_statistics=cfg.USE_BATCH_STATISTICS,
        batch_normalize=False,
        add_relu = False
      )
      detect_6_locations = tf.reshape(detect_6_locations, [batch_size, -1])
    
      detect_6_confidences = add_conv(
        graph = graph,
        input = detect_6_conv_1,
        shape = [1, 1, 96, num_bboxes_per_cell],
        strides = [1, 1, 1, 1],
        padding = "SAME",
        use_batch_statistics=cfg.USE_BATCH_STATISTICS,
        batch_normalize=False,
        add_relu = False
      )
      detect_6_confidences = tf.reshape(detect_6_confidences, [batch_size, -1])
    
    conv = add_conv(
      graph = graph,
      input = inputs,
      shape = [3, 3, 2048, 256],
      strides = [1, 2, 2, 1],
      padding = "SAME",
      use_batch_statistics=cfg.USE_BATCH_STATISTICS
    )
    
    #print "Conv Shape:"
    #print conv.get_shape().as_list()
    
    # 4 x 4 grid cells
    with graph.name_scope("4x4"):
      detect_4_conv = add_conv(
        graph = graph,
        input = conv,
        shape = [3, 3, 256, 128],
        strides = [1, 1, 1, 1],
        padding = "SAME",
        use_batch_statistics=cfg.USE_BATCH_STATISTICS
      )
      
      detect_4_locations = add_conv(
        graph = graph,
        input = detect_4_conv,
        shape = [1, 1, 128, num_bboxes_per_cell * 4],
        strides = [1, 1, 1, 1],
        padding = "SAME",
        use_batch_statistics=cfg.USE_BATCH_STATISTICS,
        batch_normalize=False,
        add_relu = False
      )
      detect_4_locations = tf.reshape(detect_4_locations, [batch_size, -1])
    
      detect_4_confidences = add_conv(
        graph = graph,
        input = detect_4_conv,
        shape = [1, 1, 128, num_bboxes_per_cell],
        strides = [1, 1, 1, 1],
        padding = "SAME",
        use_batch_statistics=cfg.USE_BATCH_STATISTICS,
        batch_normalize=False,
        add_relu = False
      )
      detect_4_confidences = tf.reshape(detect_4_confidences, [batch_size, -1])
    
    # 3 x 3 grid cells
    with graph.name_scope("3x3"):
      detect_3_conv = add_conv(
        graph = graph,
        input = conv,
        shape = [1, 1, 256, 128],
        strides = [1, 1, 1, 1],
        padding = "SAME",
        use_batch_statistics=cfg.USE_BATCH_STATISTICS
      )
      
      detect_3_conv_1 = add_conv(
        graph = graph,
        input = detect_3_conv,
        shape = [2, 2, 128, 96],
        strides = [1, 1, 1, 1],
        padding = "VALID",
        use_batch_statistics=cfg.USE_BATCH_STATISTICS
      )
      
      detect_3_locations = add_conv(
        graph = graph,
        input = detect_3_conv_1,
        shape = [1, 1, 96, num_bboxes_per_cell * 4],
        strides = [1, 1, 1, 1],
        padding = "SAME",
        use_batch_statistics=cfg.USE_BATCH_STATISTICS,
        batch_normalize=False,
        add_relu = False
      )
      detect_3_locations = tf.reshape(detect_3_locations, [batch_size, -1])
    
      detect_3_confidences = add_conv(
        graph = graph,
        input = detect_3_conv_1,
        shape = [1, 1, 96, num_bboxes_per_cell],
        strides = [1, 1, 1, 1],
        padding = "SAME",
        use_batch_statistics=cfg.USE_BATCH_STATISTICS,
        batch_normalize=False,
        add_relu = False
      )
      detect_3_confidences = tf.reshape(detect_3_confidences, [batch_size, -1])
      
    # 2 x 2 grid cells
    with graph.name_scope("2x2"):
      detect_2_conv = add_conv(
        graph = graph,
        input = conv,
        shape = [1, 1, 256, 128],
        strides = [1, 1, 1, 1],
        padding = "SAME",
        use_batch_statistics=cfg.USE_BATCH_STATISTICS
      )
      
      detect_2_conv_1 = add_conv(
        graph = graph,
        input = detect_2_conv,
        shape = [3, 3, 128, 96],
        strides = [1, 1, 1, 1],
        padding = "VALID",
        use_batch_statistics=cfg.USE_BATCH_STATISTICS
      )
      
      #print "Detect 2 conv 1:"
      #print detect_2_conv_1.get_shape().as_list()
      
      detect_2_locations = add_conv(
        graph = graph,
        input = detect_2_conv_1,
        shape = [1, 1, 96, num_bboxes_per_cell * 4],
        strides = [1, 1, 1, 1],
        padding = "SAME",
        use_batch_statistics=cfg.USE_BATCH_STATISTICS,
        batch_normalize=False,
        add_relu = False
      )
      detect_2_locations = tf.reshape(detect_2_locations, [batch_size, -1])
      
      detect_2_confidences = add_conv(
        graph = graph,
        input = detect_2_conv_1,
        shape = [1, 1, 96, num_bboxes_per_cell],
        strides = [1, 1, 1, 1],
        padding = "SAME",
        use_batch_statistics=cfg.USE_BATCH_STATISTICS,
        batch_normalize=False,
        add_relu = False
      )
      detect_2_confidences = tf.reshape(detect_2_confidences, [batch_size, -1])
    
    # 1 x 1 grid cell
    with graph.name_scope("1x1"):
      
      detect_1_pool = add_avg_pool(
        graph=graph,
        input=inputs,
        ksize=[1, 8, 8, 1],
        strides=[1, 1, 1, 1],
        padding = "VALID",
        name="pool_3"
      )
      
      detect_1_locations = add_conv(
        graph = graph,
        input = detect_1_pool,
        shape = [1, 1, 2048, 4],
        strides = [1, 1, 1, 1],
        padding = "SAME",
        use_batch_statistics=cfg.USE_BATCH_STATISTICS,
        batch_normalize=False,
        add_relu = False
      )
      detect_1_locations = tf.reshape(detect_1_locations, [batch_size, -1])
    
      detect_1_confidences = add_conv(
        graph = graph,
        input = detect_1_pool,
        shape = [1, 1, 2048, 1],
        strides = [1, 1, 1, 1],
        padding = "SAME",
        use_batch_statistics=cfg.USE_BATCH_STATISTICS,
        batch_normalize=False,
        add_relu = False
      )
      detect_1_confidences = tf.reshape(detect_1_confidences, [batch_size, -1])
    
    # print detect_8_locations.get_shape().as_list()
    # print detect_6_locations.get_shape().as_list()
    # print detect_4_locations.get_shape().as_list()
    # print detect_3_locations.get_shape().as_list()
    # print detect_2_locations.get_shape().as_list()
    # print detect_1_locations.get_shape().as_list()
    
    # Collect all of the locations and confidences 
    locations = tf.concat(1, [detect_8_locations, detect_6_locations, detect_4_locations, detect_3_locations, detect_2_locations, detect_1_locations])
    locations = tf.reshape(locations, [batch_size, -1, 4])
    
    confidences = tf.concat(1, [detect_8_confidences, detect_6_confidences, detect_4_confidences, detect_3_confidences, detect_2_confidences, detect_1_confidences])
    confidences = tf.reshape(confidences, [batch_size, -1, 1])
    confidences = tf.sigmoid(confidences)
    
    # print "Location shape:"
    # print locations.get_shape().as_list()
    # print "Confidence shape:"
    # print confidences.get_shape().as_list()
    
    # reshape the locations and confidences into a more convenient format
    #locations = tf.reshape(locations, [-1, 4])
    #confidences = tf.reshape(confidences, [-1])
  return locations, confidences

def add_loss(graph, locations, confidences, batched_bboxes, batched_num_bboxes, bbox_priors, cfg):
  
  with graph.name_scope("loss"):
    # ground truth bounding boxes:
    # [batch_size, # of ground truth bounding boxes, 4]
    # we also need to know the number of ground truth bounding boxes for each image in the batch
    # (it can be different for each image...)
    # We could assume 1 for now.
    
    # Pass the locations, confidences, and ground truth labels to the matching function
    locations = tf.reshape(locations, [-1, 4])
    confidences = tf.reshape(confidences, [-1])
    
    # add the priors to the predicted residuals
    locations += tf.tile(bbox_priors, [cfg.BATCH_SIZE, 1])
    
    # add a small epsilon to the confidences
    confidences += small_epsilon
    
    # print "Shapes"
    # print locations.get_shape().as_list()
    # print confidences.get_shape().as_list()
    # print batched_bboxes.get_shape().as_list()
    # print batched_num_bboxes.get_shape().as_list()
    params = [locations, confidences, batched_bboxes, batched_num_bboxes, cfg.BATCH_SIZE, cfg.LOCATION_LOSS_ALPHA]
    matching, stacked_gt_bboxes = tf.py_func(compute_assignments, params, [tf.int32, tf.float32], name="bipartite_matching") 
    
    # matching: [num_predictions * batch_size] 0s and 1s for partitioning
    # stacked_gt_bboxes : [total number of gt bboxes for this batch, 4]
    
    # dynamic partition the bounding boxes and confidences into "positives" and "negatives"
    unmatched_locations, matched_locations = tf.dynamic_partition(locations, matching, 2)
    unmatched_confidences, matched_confidences = tf.dynamic_partition(confidences, matching, 2)
    
    # sum the norm from the "positive" bounding boxes 
    #loss = tf.nn.l2_loss(matched_locations - stacked_gt_bboxes)
    
    # sum the negative logs of the "positive" confidences
    #loss = loss - tf.reduce_sum(tf.log(matched_confidences)) + tf.reduce_sum(tf.log((1. - matched_confidences) + small_epsilon))
    
    # sum the negative logs of one minus the all of the confidences
    ###loss = loss - (1. / tf.cast(tf.reduce_sum(batched_num_bboxes), tf.float32) ) *  tf.reduce_sum(tf.log( 1. - confidences))
    #loss = loss -  tf.reduce_sum(tf.log( (1. - confidences) + small_epsilon))
    
    location_loss = cfg.LOCATION_LOSS_ALPHA * tf.nn.l2_loss(matched_locations - stacked_gt_bboxes)
    confidence_loss = -1. * tf.reduce_sum(tf.log(matched_confidences)) - tf.reduce_sum(tf.log((1. - unmatched_confidences) + small_epsilon))
    
    #loss = -1. * tf.reduce_sum(tf.log(matched_confidences)) - tf.reduce_sum(tf.log((1. - unmatched_confidences) + small_epsilon)) + cfg.LOCATION_LOSS_ALPHA * tf.nn.l2_loss(matched_locations - stacked_gt_bboxes)
  
  return location_loss, confidence_loss, matching

def build(graph, inputs, cfg):
  
  logits, endpoints = v3_model.build(
     inputs = inputs,
     dropout_keep_prob=0.8,
     num_classes=1001,
     is_training=cfg.USE_BATCH_STATISTICS,
     scope=''
  )
  
  return endpoints['mixed_8x8x2048b']