import tensorflow as tf


class MLP(object):
    def __init__(self, X, n_hiddens, n_in, n_out):
        self.X = X
        for i, n_hidden in enumerate(n_hiddens):
            if i == 0:
                input = X
                input_dim = n_in
            else:
                input = output
                input_dim = n_hiddens[i-1]

            layer_name = "layer_"+str(i+1)

            with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
                W = self._weight_variable(layer_name, [input_dim, n_hidden])
                b = self._bias_variable(layer_name, [n_hidden])

                output = tf.nn.sigmoid(tf.matmul(input, W) + b)

        layer_name = "output"
        with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
            W = self._weight_variable(layer_name, [n_hiddens[-1], n_out])
            b = self._bias_variable(layer_name, [n_out])
            self.hypothesis = tf.matmul(output, W) + b

    def _weight_variable(self, layer_name, shape):
        return tf.get_variable(
            layer_name + "_w", shape=shape,
            initializer=tf.random_normal_initializer)

    def _bias_variable(self, layer_name, shape):
        return tf.get_variable(
            layer_name + "_bias", shape=shape,
            initializer=tf.random_normal_initializer)
