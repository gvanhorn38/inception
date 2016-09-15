"""
These utility functions are meant for computing basic statistics in a set of tfrecord 
files. They can be used to sanity check the training and testing files. 
"""
import argparse
import tensorflow as tf

def class_stats(tfrecords):
  """
  Sum the number of images and compute the number of images available for each class. 
  """
  
  filename_queue = tf.train.string_input_producer(
    tfrecords,
    num_epochs=1
  )

  # Construct a Reader to read examples from the .tfrecords file
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  
  features = tf.parse_single_example(
    serialized_example,
    features = {
      'image/class/label' : tf.FixedLenFeature([], tf.int64)
    }
  )
  
  label = tf.cast(features['image/class/label'], tf.int32)
  
  image_count = 0
  class_image_count = {}
  
  coord = tf.train.Coordinator()
  with tf.Session() as sess:

    tf.initialize_all_variables().run()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
      
      while not coord.should_stop():
        
        outputs = sess.run([label])
        
        class_label = outputs[0]
        if class_label not in class_image_count:
          class_image_count[class_label] = 0
        class_image_count[class_label] += 1
        image_count += 1
        
        
    except tf.errors.OutOfRangeError as e:
      pass
  
  # Basic info
  print "Found %d images" % (image_count,)
  print "Found %d classes" % (len(class_image_count),)
  
  class_labels = class_image_count.keys()
  class_labels.sort()
  
  # Print out the per class image counts
  print "Class Index | Image Count"
  for class_label in class_labels:
    print "{0:11d} | {1:6d} ".format(class_label, class_image_count[class_label]) 
  
  # Can we detect if there any missing classes? 
  max_class_index = max(class_labels)
  
  # We expect class id for each value in the range [0, max_class_id]
  # So lets see if we are missing any of these values
  missing_values = list(set(range(max_class_index+1)).difference(class_labels))
  if len(missing_values) > 0:
    print "WARNING: expected %d classes but only found %d classes." % (max_class_index, len(class_labels))
    missing_values.sort()
    for index in missing_values:
      print "Missing class %d" % (index,) 
    
def bbox_stats(tfrecords):
  """
  Check that each example has valid bounding box data. 
  """
  
  filename_queue = tf.train.string_input_producer(
    tfrecords,
    num_epochs=1
  )

  # Construct a Reader to read examples from the .tfrecords file
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  
  # Parse an Example to access the Features
  features = tf.parse_single_example(
    serialized_example,
    features = {
      'image/class/label' : tf.FixedLenFeature([], tf.int64),
      'image/id' : tf.FixedLenFeature([], tf.string),
      'image/object/bbox/xmin' : tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/ymin' : tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/xmax' : tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/ymax' : tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/label' : tf.VarLenFeature(dtype=tf.int64),
      'image/object/id' : tf.VarLenFeature(dtype=tf.string),
    }
  )

  #label = tf.cast(features['image/class/label'], tf.int32)
  image_id = features['image/id']
  
  xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
  ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
  xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
  ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)
  label = tf.expand_dims(tf.cast(features['image/object/bbox/label'].values, tf.float32), 0)
  annot_id = tf.expand_dims(features['image/object/id'].values, 0)

  image_count = 0
  image_missing_bbox_count = 0
  flat_bbox_count = 0
  
  coord = tf.train.Coordinator()
  with tf.Session() as sess:

    tf.initialize_all_variables().run()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
      
      while not coord.should_stop():
        
        #outputs = sess.run([xmin, xmax, ymin, ymax, label])
        outputs = sess.run([xmin, xmax, ymin, ymax, label, image_id, annot_id])
        #outputs = sess.run([label])
        
        image_xmin = outputs[0]
        image_xmax = outputs[1]
        image_ymin = outputs[2]
        image_ymax = outputs[3]
        image_label = outputs[4]
        id = outputs[5]
        annotation_id = outputs[6]

        image_count += 1
        if(len(image_xmin) == 0):
          image_missing_bbox_count += 1
          print "MISSING bbox: image_id %s, box_id %s, Class %s, xmin %s, xmax %s, ymin %s, ymax %s" % (id, annotation_id, image_label, image_xmin, image_xmax, image_ymin, image_ymax)
        elif image_xmin[0] >= image_xmax[0] or image_ymin[0] >= image_ymax[0]:
          flat_bbox_count += 1
          print "bbox ERROR: image_id %s, box_id %s, Class %s, xmin %s, xmax %s, ymin %s, ymax %s" % (id, annotation_id, image_label, image_xmin, image_xmax, image_ymin, image_ymax)

        if image_count % 1000 == 0:
          print "Processed %d images" % image_count

        
    except tf.errors.OutOfRangeError as e:
      pass
  
  # Basic info
  print "Found %d images" % (image_count,)
  print "Found %d missing bboxes" % (image_missing_bbox_count,)
  print "Found %d 2d bboxes" % (flat_bbox_count,)
  

def parse_args():

    parser = argparse.ArgumentParser(description='Basic statistics on tfrecord files')
  
    parser.add_argument('--stat', dest='stat_type',
                        choices=['class_stats', 'bbox_stats'],
                        required=True)
    
    parser.add_argument('--tfrecords', dest='tfrecords',
                        help='paths to tfrecords files', type=str,
                        nargs='+', required=True)
    

    args = parser.parse_args()
    
    return args

if __name__ == '__main__':

    args = parse_args()
    
    if args.stat_type == 'class_stats':
      class_stats(args.tfrecords)
    elif args.stat_type == 'bbox_stats':
      bbox_stats(args.tfrecords)

