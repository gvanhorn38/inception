import tensorflow as tf
import tensorflow.contrib.slim as slim

def build(inputs,
                 dropout_keep_prob=0.8,
                 num_classes=1000,
                 is_training=True,
                 scope=''):
  
  end_points = {}
  with tf.op_scope([inputs], scope, 'inception_v3'):
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                          is_training=is_training):
      with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='VALID'):
        # 299 x 299 x 3
        end_points['conv0'] = slim.conv2d(inputs, 32, [3, 3], stride=2,
                                         scope='conv0')
        # 149 x 149 x 32
        end_points['conv1'] = slim.conv2d(end_points['conv0'], 32, [3, 3],
                                         scope='conv1')
        # 147 x 147 x 32
        end_points['conv2'] = slim.conv2d(end_points['conv1'], 64, [3, 3],
                                         padding='SAME', scope='conv2')
        # 147 x 147 x 64
        end_points['pool1'] = slim.max_pool2d(end_points['conv2'], [3, 3],
                                           stride=2, scope='pool1')
        # 73 x 73 x 64
        end_points['conv3'] = slim.conv2d(end_points['pool1'], 80, [1, 1],
                                         scope='conv3')
        # 73 x 73 x 80.
        end_points['conv4'] = slim.conv2d(end_points['conv3'], 192, [3, 3],
                                         scope='conv4')
        # 71 x 71 x 192.
        end_points['pool2'] = slim.max_pool2d(end_points['conv4'], [3, 3],
                                           stride=2, scope='pool2')
        # 35 x 35 x 192.
        net = end_points['pool2']
      
      # Inception blocks
      with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='SAME'):
        # mixed: 35 x 35 x 256.
        with tf.variable_scope('mixed_35x35x256a'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = slim.conv2d(net, 64, [1, 1])
          with tf.variable_scope('branch5x5'):
            branch5x5 = slim.conv2d(net, 48, [1, 1])
            branch5x5 = slim.conv2d(branch5x5, 64, [5, 5])
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = slim.conv2d(net, 64, [1, 1])
            branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
            branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
          with tf.variable_scope('branch_pool'):
            branch_pool = slim.avg_pool2d(net, [3, 3])
            branch_pool = slim.conv2d(branch_pool, 32, [1, 1])
          net = tf.concat(3, [branch1x1, branch5x5, branch3x3dbl, branch_pool])
          end_points['mixed_35x35x256a'] = net
        # mixed_1: 35 x 35 x 288.
        with tf.variable_scope('mixed_35x35x288a'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = slim.conv2d(net, 64, [1, 1])
          with tf.variable_scope('branch5x5'):
            branch5x5 = slim.conv2d(net, 48, [1, 1])
            branch5x5 = slim.conv2d(branch5x5, 64, [5, 5])
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = slim.conv2d(net, 64, [1, 1])
            branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
            branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
          with tf.variable_scope('branch_pool'):
            branch_pool = slim.avg_pool2d(net, [3, 3])
            branch_pool = slim.conv2d(branch_pool, 64, [1, 1])
          net = tf.concat(3, [branch1x1, branch5x5, branch3x3dbl, branch_pool])
          end_points['mixed_35x35x288a'] = net
        # mixed_2: 35 x 35 x 288.
        with tf.variable_scope('mixed_35x35x288b'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = slim.conv2d(net, 64, [1, 1])
          with tf.variable_scope('branch5x5'):
            branch5x5 = slim.conv2d(net, 48, [1, 1])
            branch5x5 = slim.conv2d(branch5x5, 64, [5, 5])
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = slim.conv2d(net, 64, [1, 1])
            branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
            branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
          with tf.variable_scope('branch_pool'):
            branch_pool = slim.avg_pool2d(net, [3, 3])
            branch_pool = slim.conv2d(branch_pool, 64, [1, 1])
          net = tf.concat(3, [branch1x1, branch5x5, branch3x3dbl, branch_pool])
          end_points['mixed_35x35x288b'] = net
        # mixed_3: 17 x 17 x 768.
        with tf.variable_scope('mixed_17x17x768a'):
          with tf.variable_scope('branch3x3'):
            branch3x3 = slim.conv2d(net, 384, [3, 3], stride=2, padding='VALID')
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = slim.conv2d(net, 64, [1, 1])
            branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
            branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3],
                                      stride=2, padding='VALID')
          with tf.variable_scope('branch_pool'):
            branch_pool = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID')
          net = tf.concat(3, [branch3x3, branch3x3dbl, branch_pool])
          end_points['mixed_17x17x768a'] = net
        # mixed4: 17 x 17 x 768.
        with tf.variable_scope('mixed_17x17x768b'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = slim.conv2d(net, 192, [1, 1])
          with tf.variable_scope('branch7x7'):
            branch7x7 = slim.conv2d(net, 128, [1, 1])
            branch7x7 = slim.conv2d(branch7x7, 128, [1, 7])
            branch7x7 = slim.conv2d(branch7x7, 192, [7, 1])
          with tf.variable_scope('branch7x7dbl'):
            branch7x7dbl = slim.conv2d(net, 128, [1, 1])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 128, [7, 1])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 128, [1, 7])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 128, [7, 1])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [1, 7])
          with tf.variable_scope('branch_pool'):
            branch_pool = slim.avg_pool2d(net, [3, 3])
            branch_pool = slim.conv2d(branch_pool, 192, [1, 1])
          net = tf.concat(3, [branch1x1, branch7x7, branch7x7dbl, branch_pool])
          end_points['mixed_17x17x768b'] = net
        # mixed_5: 17 x 17 x 768.
        with tf.variable_scope('mixed_17x17x768c'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = slim.conv2d(net, 192, [1, 1])
          with tf.variable_scope('branch7x7'):
            branch7x7 = slim.conv2d(net, 160, [1, 1])
            branch7x7 = slim.conv2d(branch7x7, 160, [1, 7])
            branch7x7 = slim.conv2d(branch7x7, 192, [7, 1])
          with tf.variable_scope('branch7x7dbl'):
            branch7x7dbl = slim.conv2d(net, 160, [1, 1])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 160, [7, 1])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 160, [1, 7])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 160, [7, 1])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [1, 7])
          with tf.variable_scope('branch_pool'):
            branch_pool = slim.avg_pool2d(net, [3, 3])
            branch_pool = slim.conv2d(branch_pool, 192, [1, 1])
          net = tf.concat(3, [branch1x1, branch7x7, branch7x7dbl, branch_pool])
          end_points['mixed_17x17x768c'] = net
        # mixed_6: 17 x 17 x 768.
        with tf.variable_scope('mixed_17x17x768d'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = slim.conv2d(net, 192, [1, 1])
          with tf.variable_scope('branch7x7'):
            branch7x7 = slim.conv2d(net, 160, [1, 1])
            branch7x7 = slim.conv2d(branch7x7, 160, [1, 7])
            branch7x7 = slim.conv2d(branch7x7, 192, [7, 1])
          with tf.variable_scope('branch7x7dbl'):
            branch7x7dbl = slim.conv2d(net, 160, [1, 1])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 160, [7, 1])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 160, [1, 7])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 160, [7, 1])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [1, 7])
          with tf.variable_scope('branch_pool'):
            branch_pool = slim.avg_pool2d(net, [3, 3])
            branch_pool = slim.conv2d(branch_pool, 192, [1, 1])
          net = tf.concat(3, [branch1x1, branch7x7, branch7x7dbl, branch_pool])
          end_points['mixed_17x17x768d'] = net
        # mixed_7: 17 x 17 x 768.
        with tf.variable_scope('mixed_17x17x768e'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = slim.conv2d(net, 192, [1, 1])
          with tf.variable_scope('branch7x7'):
            branch7x7 = slim.conv2d(net, 192, [1, 1])
            branch7x7 = slim.conv2d(branch7x7, 192, [1, 7])
            branch7x7 = slim.conv2d(branch7x7, 192, [7, 1])
          with tf.variable_scope('branch7x7dbl'):
            branch7x7dbl = slim.conv2d(net, 192, [1, 1])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [7, 1])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [1, 7])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [7, 1])
            branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [1, 7])
          with tf.variable_scope('branch_pool'):
            branch_pool = slim.avg_pool2d(net, [3, 3])
            branch_pool = slim.conv2d(branch_pool, 192, [1, 1])
          net = tf.concat(3, [branch1x1, branch7x7, branch7x7dbl, branch_pool])
          end_points['mixed_17x17x768e'] = net
        
        # Auxiliary Head logits
        aux_logits = tf.identity(end_points['mixed_17x17x768e'])
        with tf.variable_scope('aux_logits'):
          aux_logits = slim.avg_pool2d(aux_logits, [5, 5], stride=3,
                                    padding='VALID')
          aux_logits = slim.conv2d(aux_logits, 128, [1, 1], scope='proj')
          # Shape of feature map before the final layer.
          shape = aux_logits.get_shape()
          aux_logits = slim.conv2d(aux_logits, 768, shape[1:3],
                                  padding='VALID')
          aux_logits = slim.flatten(aux_logits)
          aux_logits = slim.fully_connected(aux_logits, num_classes, activation_fn=None)
          end_points['aux_logits'] = aux_logits
        # mixed_8: 8 x 8 x 1280.
        # Note that the scope below is not changed to not void previous
        # checkpoints.
        # (TODO) Fix the scope when appropriate.
        with tf.variable_scope('mixed_17x17x1280a'):
          with tf.variable_scope('branch3x3'):
            branch3x3 = slim.conv2d(net, 192, [1, 1])
            branch3x3 = slim.conv2d(branch3x3, 320, [3, 3], stride=2,
                                   padding='VALID')
          with tf.variable_scope('branch7x7x3'):
            branch7x7x3 = slim.conv2d(net, 192, [1, 1])
            branch7x7x3 = slim.conv2d(branch7x7x3, 192, [1, 7])
            branch7x7x3 = slim.conv2d(branch7x7x3, 192, [7, 1])
            branch7x7x3 = slim.conv2d(branch7x7x3, 192, [3, 3],
                                     stride=2, padding='VALID')
          with tf.variable_scope('branch_pool'):
            branch_pool = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID')
          net = tf.concat(3, [branch3x3, branch7x7x3, branch_pool])
          end_points['mixed_17x17x1280a'] = net
        # mixed_9: 8 x 8 x 2048.
        with tf.variable_scope('mixed_8x8x2048a'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = slim.conv2d(net, 320, [1, 1])
          with tf.variable_scope('branch3x3'):
            branch3x3 = slim.conv2d(net, 384, [1, 1])
            branch3x3 = tf.concat(3, [slim.conv2d(branch3x3, 384, [1, 3]),
                                      slim.conv2d(branch3x3, 384, [3, 1])])
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = slim.conv2d(net, 448, [1, 1])
            branch3x3dbl = slim.conv2d(branch3x3dbl, 384, [3, 3])
            branch3x3dbl = tf.concat(3, [slim.conv2d(branch3x3dbl, 384, [1, 3]),
                                         slim.conv2d(branch3x3dbl, 384, [3, 1])])
          with tf.variable_scope('branch_pool'):
            branch_pool = slim.avg_pool2d(net, [3, 3])
            branch_pool = slim.conv2d(branch_pool, 192, [1, 1])
          net = tf.concat(3, [branch1x1, branch3x3, branch3x3dbl, branch_pool])
          end_points['mixed_8x8x2048a'] = net
        # mixed_10: 8 x 8 x 2048.
        with tf.variable_scope('mixed_8x8x2048b'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = slim.conv2d(net, 320, [1, 1])
          with tf.variable_scope('branch3x3'):
            branch3x3 = slim.conv2d(net, 384, [1, 1])
            branch3x3 = tf.concat(3, [slim.conv2d(branch3x3, 384, [1, 3]),
                                      slim.conv2d(branch3x3, 384, [3, 1])])
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = slim.conv2d(net, 448, [1, 1])
            branch3x3dbl = slim.conv2d(branch3x3dbl, 384, [3, 3])
            branch3x3dbl = tf.concat(3, [slim.conv2d(branch3x3dbl, 384, [1, 3]),
                                         slim.conv2d(branch3x3dbl, 384, [3, 1])])
          with tf.variable_scope('branch_pool'):
            branch_pool = slim.avg_pool2d(net, [3, 3])
            branch_pool = slim.conv2d(branch_pool, 192, [1, 1])
          net = tf.concat(3, [branch1x1, branch3x3, branch3x3dbl, branch_pool])
          end_points['mixed_8x8x2048b'] = net
        # Final pooling and prediction
        with tf.variable_scope('logits'):
          shape = net.get_shape()
          net = slim.avg_pool2d(net, shape[1:3], padding='VALID', scope='pool')
          # 1 x 1 x 2048
          net = slim.dropout(net, dropout_keep_prob, scope='dropout')
          net = slim.flatten(net, scope='flatten')
          # 2048
          logits = slim.fully_connected(net, num_classes, activation_fn=None, scope='logits')
          # 1000
          end_points['logits'] = logits
          end_points['predictions'] = tf.nn.softmax(logits, name='predictions')
      
      return logits, end_points

