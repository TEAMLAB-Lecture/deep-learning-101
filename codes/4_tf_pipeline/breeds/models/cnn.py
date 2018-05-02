import tensorflow as tf
slim = tf.contrib.slim

class SimpleLeNet(object):
    def __init__(self):
        self.n_hidden_1 = 2048
        self.n_hidden_2 = 1024
        self.n_input = 128*128*3
        self.n_classes = 120
        self.learning_rate = 0.001
        self.optimizer = None

        self.image_batch = tf.placeholder(tf.float32, shape=[None, 128,128,3])
        self.label_batch = tf.placeholder(tf.float32, shape=[None, self.n_classes])
        self.is_training = tf.placeholder(tf.bool)

        float_image_batch = tf.image.convert_image_dtype(self.image_batch, tf.float32)

        conv2d_layer_one = tf.contrib.layers.convolution2d(
            float_image_batch,
            num_outputs=32,     # The number of filters to generate
            kernel_size=(5,5),          # It's only the filter height and width.
            activation_fn=tf.nn.relu,
            stride=(2, 2),
            trainable=True)
        pool_layer_one = tf.nn.max_pool(conv2d_layer_one,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME')

        # Note, the first and last dimension of the convolution output hasn't changed but the
        # middle two dimensions have.
        conv2d_layer_one.get_shape(), pool_layer_one.get_shape()

        conv2d_layer_two = tf.contrib.layers.convolution2d(
            pool_layer_one,
            num_outputs=64,        # More output channels means an increase in the number of filters
            kernel_size=(5,5),
            activation_fn=tf.nn.relu,
            stride=(1, 1),
            trainable=True)

        pool_layer_two = tf.nn.max_pool(conv2d_layer_two,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME')

        pool_layer_two = slim.flatten(pool_layer_two, scope='flatten3')
        print(pool_layer_two)

        pool2_flat = tf.reshape(pool_layer_two, [-1, 16384])

        dense = tf.layers.dense(inputs=pool2_flat, units=128, activation=tf.nn.relu)

        # The weight_init parameter can also accept a callable, a lambda is used here  returning a truncated normal
        # with a stddev specified.
        hidden_layer_three = tf.contrib.layers.fully_connected(
            inputs=dense, num_outputs= 128
        )

        # Dropout some of the neurons, reducing their importance in the model
        if self.is_training == True:
            hidden_layer_three = tf.nn.dropout(hidden_layer_three, 0.1)
        else:
            hidden_layer_three = hidden_layer_three

        # The output of this are all the connections between the previous layers and the 120 different dog breeds
        # available to train on.
        final_fully_connected = tf.contrib.layers.fully_connected(
            hidden_layer_three,
            120
        )

        self.loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=final_fully_connected, labels=self.label_batch)
        self.cost = tf.reduce_mean(self.loss)
        tf.summary.scalar('loss', self.cost)


        train_prediction = tf.nn.softmax(final_fully_connected)
        self.pred = train_prediction

        self.optimizer = tf.train.AdamOptimizer(0.001, 0.9)

        # create train op
        self.train_op = self.optimizer.minimize(self.cost)
