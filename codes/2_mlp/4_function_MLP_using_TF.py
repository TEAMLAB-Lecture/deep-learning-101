import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def inference(X, n_in, n_hiddens, n_out):
    def weight_variable(layer_name, shape):
        return tf.get_variable(
            layer_name + "_w", shape=shape,
            initializer=tf.random_normal_initializer())

    def bias_variable(layer_name, shape):
        return tf.get_variable(
            layer_name + "_bias", shape=shape,
            initializer=tf.random_normal_initializer())

    for i, n_hidden in enumerate(n_hiddens):
        if i == 0:
            input = X
            input_dim = n_in
        else:
            input = output
            input_dim = n_hiddens[i-1]

        layer_name = "layer_"+str(i+1)

        with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
            W = weight_variable(layer_name, [input_dim, n_hidden])
            b = bias_variable(layer_name, [n_hidden])

            output = tf.nn.relu(tf.matmul(input, W) + b)

    layer_name = "output"
    with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
        W = weight_variable(layer_name, [n_hiddens[-1], n_out])
        b = bias_variable(layer_name, [n_out])
        hypothesis = tf.matmul(output, W) + b
    return hypothesis


def loss(hypothesis, Y):
    with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=hypothesis, labels=Y), name="softmax_cross_entropy")
    return loss


def training(loss, learning_rate):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_step = optimizer.minimize(loss)
    return train_step

if __name__ == "__main__":
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    X = tf.placeholder(tf.float32, [None, 784])
    Y = tf.placeholder(tf.float32, [None, 10])

    learning_rate = 0.001
    training_epochs = 20
    batch_size = 128

    n_in = 784
    n_hiddens = [256, 64]
    n_out = 10

    hypothesis = inference(X, n_in, n_hiddens=n_hiddens, n_out=n_out)
    loss = loss(hypothesis, Y)
    train_step = training(loss, learning_rate=learning_rate)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed_dict = {X: batch_xs, Y: batch_ys}
            c, _ = sess.run([loss, train_step], feed_dict=feed_dict)
            avg_cost += c / total_batch


        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(
            avg_cost))

        correct_prediction = tf.equal(
                tf.argmax(hypothesis, axis=1), tf.argmax(Y, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('Accuracy:', sess.run(accuracy, feed_dict={
              X: mnist.test.images, Y: mnist.test.labels}))


    print('Learning Finished!')


    # 3. Learning process
    # 4. Check performance
