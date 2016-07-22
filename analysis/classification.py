"""
Utilities for analyzing classification tests.
"""

import argparse
import cPickle as pickle
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf


"""
Dump files that can be uploaded to Google Sheets for analysis. 
"""

def process(classification_tfrecords, num_classes, output_dir, class_names=None, image_urls=None):
  """
  classification_records : tf record file containing the classification results
  num_classes : 
  output_dir : 
  class_names : a dictionary mapping from class ids to human readable text
  image_urls : a dictionary mapping from image ids to a url for the image
  """
  
  if class_names == None:
    class_names = {}

  # Lets compute the TP, FP, and FN for each class
  # We'll also store mistake information for the images so that we can visualize them
  class_stats = {}
  mistake_data = {} # This is false positive mistakes
  for class_id in range(num_classes):
    class_stats[class_id] = {'tp' : 0., 'fp' : 0., 'fn' : 0., 'support' : 0}
    mistake_data[class_id] = [] # we'll store the instance_id, pred label 
  
  # We'll track top-k raw image classification accuracy
  top_k_correct_count = np.zeros(num_classes)
  
  # Confusion matrix data
  confusion_matrix = np.zeros((num_classes, num_classes))
  
  # We'll also track raw accuracy
  raw_num_correct = 0
  
  # TensorFlow operations to read in the classification results
  filename_queue = tf.train.string_input_producer(
    classification_records,
    num_epochs=1
  )
  
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)

  # Parse an Example to access the Features
  features = tf.parse_single_example(
    serialized_example,
    features = {
      'image/pred/class/logits' : _float_feature(logits),
      'image/pred/class/label' :  _int64_feature(pred_label),
      'image/class/label': _int64_feature(gt_label),
      'image/id': _bytes_feature(str(id))
    }
  )

  tf_image_id = features['image/id']
  tf_gt_label = tf.cast(features['image/class/label'], tf.int32)
  tf_pred_label = tf.cast(features['image/pred/class/label'], tf.int32)
  tf_logits = features['image/pred/class/logits']
  
  fetches = [tf_image_id, tf_gt_label, tf_pred_label, tf_logits]
  
  # Due to the batching of images, it could be the case that an image was classified twice
  # so lets keep track of the instances we have already processed.
  processed_instances = set()
  num_instances = 0
  coord = tf.train.Coordinator()
  with tf.Session() as sess:
    
    tf.initialize_all_variables().run()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    try:
    
      while not coord.should_stop():
      
        # classification result
        image_id, gt_label, pred_label, logits = sess.run(fetches)
      
        if image_id in processed_instances:
          continue
        else:
          processed_instances.add(image_id)
          num_instances += 1
      
        class_stats[gt_label]['support'] += 1
      
        # Correct classification
        if gt_label == pred_label:
          
          raw_num_correct += 1
          
          class_stats[gt_label]['tp'] += 1
        
          # update top-k
          for i in range(num_classes):
            top_k_correct_count[i] += 1
            
        # Incorrect classification  
        else:
          class_stats[gt_label]['fn'] += 1
          class_stats[pred_label]['fp'] += 1
        
          # Save the mistake info for visualization purposes
          mistake_data[gt_label].append([image_id, pred_label])
        
          # Figure out where this landed in the top-k metric
          for i, class_id in enumerate(np.argsort(logits)[::-1]):
            if class_id == gt_label:
              for j in range(i, num_classes):
                top_k_correct_count[j] += 1
              break
            
        # Update the confusion matrix
        confusion_matrix[gt_label][pred_label] += 1 

    except tf.errors.OutOfRangeError as e:
      pass

    coord.request_stop()
    coord.join(threads)
    
  class_data = {}
  for class_id, stats in class_stats.iteritems():
    
    # Compute the f1 score for the category
    num = 2. * stats['tp']
    denom = 2. * stats['tp'] + stats['fn'] + stats['fp']
    f1 = num  / denom
    
    # process the mistakes for this category. We want to be able to render the images. 
    mistake_images = []
    if image_urls != None:
      for image_id, pred_class_id in mistake_data[class_id]:
      
        image_url = image_urls.get(image_id, "")
      
        if class_names != None and pred_class_id in class_names:
          mistake_images.append((image_url, class_names.get(pred_class_id, pred_class_id), image_id))
    
    class_data[class_id] = {
      'classification_id' : class_id,
      'test_support' : stats['support'],
      'f1_score' : f1,
      'tp' : stats['tp'],
      'fp' : stats['fp'],
      'fn' : stats['fn'],
      'name' : class_names.get(class_id, class_id),
      'mistakes' : mistake_images
    }
  
  report_data = class_data.values()
  
  # Create the output directory if it doesn't exist. 
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  
  #####################################
  # Dump a general report with some basic stats
  with open(os.path.join(output_dir, 'general_report.tsv'), 'w') as f: 
    print >> f, "\t".join(['Num Classes', 'Num Instances', 'Raw Accuracy'])
    print >> f, "\t".join([
      str(num_classes),
      str(num_instances),
      str(float(raw_num_correct) / num_instances) 
    ])
  
  #####################################
  # Dump a classification report per category
  with open(os.path.join(output_dir, 'classification_report.tsv'), 'w') as f:
    
    print >> f, "\t".join(['Class Id', 'Name', 'F1', 'TP', 'FP', 'FN', 'Test Support'])
    
    for data in report_data:
      print >> f, "\t".join(map(str, [
        data['classification_id'],
        data['name'],
        data['f1_score'],
        data['tp'],
        data['fp'],
        data['fn'],
        data['test_support']
      ]))
  
  #####################################  
  # Dump the mistakes
  if image_urls != None:
    mistake_dir = os.path.join(output_dir, "fp_mistakes")
    if not os.path.exists(mistake_dir):
      os.makedirs(mistake_dir)
    # dump all mistakes to one file
    with open(os.path.join(mistake_dir, '__all.tsv'), 'w') as f:
    
      # Headers
      print >> f, "\t".join(["Image", "GT Class", "Pred Class", "Instance ID"])
    
      for data in report_data:
      
        for image_url, pred_category_name, instance_id in data['mistakes']:
          print >> f, "\t".join([
            "=image(\"%s\")" % (image_url,),
            data['name'],
            pred_category_name,
            instance_id
          ])
      
        print >> f, "" # Leave a line between categories. 
  
    # now go through and make individual files for the individual categories
    for data in report_data:
    
      with open(os.path.join(mistake_dir, '%s.tsv' % (data['name'],)), 'w') as f:
      
        # Headers
        print >> f, "\t".join(["Image", "GT Class", "Pred Class", "Instance ID"])
      
        for image_url, pred_category_name, instance_id in data['mistakes']:
          print >> f, "\t".join([
            "=image(\"%s\")" % (image_url,),
            data['name'],
            pred_category_name,
            instance_id
          ])
  
  #####################################
  # Dump the top-k data
  with open(os.path.join(output_dir, 'top-k.tsv'), 'w') as f:
    # Header
    print >> f, "\t".join(["k", "Accuracy"])
    for i, correct_count in enumerate(top_k_correct_count):
      print >> f, "%d\t%0.4f" % (i+1, float(correct_count) / num_instances)
  
  #####################################
  # Dump the confusion matrix
  # Not sure what I want to do here....
  with open(os.path.join(output_dir, 'confusion_matrix.tsv'), 'w') as f:
    print >> f, "\t".join([""] + [x['name'] for x in report_data])
    for i in range(num_classes):
      print >> f, "\t".join([report_data[i]['name']] + map(str, confusion_matrix[i]))
  

if __name__ == '__main__':
  classification_records = [sys.argv[1]]
  category_labels_file = sys.argv[2]
  output_dir = sys.argv[3]
  process(classification_records, category_labels_file, output_dir)
  
  