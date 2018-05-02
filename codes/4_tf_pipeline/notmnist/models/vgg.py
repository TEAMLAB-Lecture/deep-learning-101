import tensorflow as tf
slim = tf.contrib.slim

class SimpleVGGNet(object):
    def __init__(self):
        self.n_input = 128*128*3
        self.n_classes = 120
        self.learning_rate = 0.001
        self.optimizer = None
        dropout_keep_prob = 0.5

        self.image_batch = tf.placeholder(tf.float32, shape=[None, 128,128,3])
        self.label_batch = tf.placeholder(tf.float32, shape=[None, self.n_classes])
        self.is_training = tf.placeholder(tf.bool)

        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d]):
          self.net = slim.repeat(self.image_batch, 2, slim.conv2d, 64, [3, 3], scope='conv1')
          self.net = slim.max_pool2d(self.net, [2, 2], scope='pool1')
          self.net = slim.repeat(self.net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
          self.net = slim.max_pool2d(self.net, [2, 2], scope='pool2')
          self.net = slim.repeat(self.net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
          self.net = slim.max_pool2d(self.net, [2, 2], scope='pool3')
          self.net = slim.repeat(self.net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
          self.net = slim.max_pool2d(self.net, [2, 2], scope='pool4')
          self.net = slim.repeat(self.net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
          self.net = slim.max_pool2d(self.net, [2, 2], scope='pool5')
          # Use conv2d instead of fully_connected layers.
          self.net = slim.conv2d(self.net, 4096, [7, 7], padding="SAME", scope='fc6')
          self.net = slim.dropout(self.net, dropout_keep_prob, is_training=self.is_training,
                             scope='dropout6')
          self.net = slim.conv2d(self.net, 4096, [1, 1], scope='fc7')
          self.net = slim.dropout(self.net, 0.5, is_training=self.is_training,
                             scope='dropout7')
          self.net = slim.conv2d(self.net, self.n_classes , [1, 1],
                            activation_fn=None,
                            normalizer_fn=None,
                            scope='fc8')
        self.net = slim.flatten(self.net, scope='flatten3')
        self.net = slim.fully_connected(self.net, self.n_classes, activation_fn=None, scope='fc5')
        self.pred = self.net

        slim.losses.softmax_cross_entropy(
            self.pred,
            self.label_batch)

        self.cost = slim.losses.get_total_loss()
        tf.summary.scalar('loss', self.cost)

        self.optimizer = tf.train.AdamOptimizer(0.01, 0.9)

        # create train op
        self.train_op = self.optimizer.minimize(self.cost)
