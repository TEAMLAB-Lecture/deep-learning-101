import tensorflow as tf

class SimpleModel(object):
    def __init__(self):
        self.n_hidden_1 = 2048
        self.n_hidden_2 = 1024
        self.n_input = 100*100
        self.n_classes = 120
        self.learning_rate = 0.001
        self.optimizer = None

        with tf.name_scope("simple_mlp"):

            self.image_batch = tf.placeholder(tf.float32, shape=[None, 100,100,1])
            self.label_batch = tf.placeholder(tf.float32, shape=[None, self.n_classes])

            self.image = tf.reshape(self.image_batch, [-1, self.n_input])

            with tf.name_scope("hidden_layer_1"):
                self.weight_1 = tf.get_variable(
                    "weight_1",
                    shape=[self.n_input,self.n_hidden_1],
                    initializer=tf.contrib.layers.xavier_initializer()
                )
                self.bias_1 = tf.get_variable(
                    "bias_1",
                    shape=[self.n_hidden_1],
                    initializer=tf.contrib.layers.xavier_initializer()
                )
                self.layer_1 = tf.add(tf.matmul(self.image, self.weight_1),
                    self.bias_1)
                self.layer_1 = tf.nn.relu(self.layer_1)

            with tf.name_scope("hidden_layer_2"):
                self.weight_2 = tf.get_variable(
                    "weight_2",
                    shape=[self.n_hidden_1,self.n_hidden_2],
                    initializer=tf.contrib.layers.xavier_initializer()
                )
                self.bias_2 = tf.get_variable(
                    "bias_2",
                    shape=[self.n_hidden_2],
                    initializer=tf.contrib.layers.xavier_initializer()
                )
                self.layer_2 = tf.add(tf.matmul(self.layer_1 , self.weight_2),
                    self.bias_2)
                self.layer_2 = tf.nn.relu(self.layer_2)

            with tf.name_scope("output_layer"):
                self.weight_output = tf.get_variable(
                    "weight_output",
                    shape=[self.n_hidden_2,self.n_classes],
                    initializer=tf.contrib.layers.xavier_initializer()
                )
                self.bias_output = tf.get_variable(
                    "bias_output",
                    shape=[self.n_classes],
                    initializer=tf.contrib.layers.xavier_initializer()
                )
                self.layer_output = tf.add(tf.matmul(self.layer_2 , self.weight_output),
                    self.bias_output)
                self.pred = self.layer_output

            with tf.name_scope("cost_fucntion"):
                self.cost = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.pred, labels=self.label_batch))
            with tf.name_scope("optimizing"):
                self.test = tf.placeholder(tf.bool)
                self.optimizer = tf.train.AdamOptimizer(
                        learning_rate=self.learning_rate,
                        ).minimize(self.cost)
