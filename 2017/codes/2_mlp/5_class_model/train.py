import tensorflow as tf
from model import MLP
from tensorflow.examples.tutorials.mnist import input_data


def loss_function(hypothesis, Y):
    with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=hypothesis, labels=Y), name="softmax_cross_entropy")
    return loss


def training(loss, learning_rate):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_step = optimizer.minimize(loss)
    return train_step


def train(config_dict):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    n_in = 784
    n_hiddens = [128, 256, 64]
    n_out = 10

    mlp_model = MLP(X, n_hiddens=n_hiddens, n_in=n_in, n_out=n_out)
    loss = loss_function(mlp_model.hypothesis, Y)
    train_step = training(loss, learning_rate=config_dict["learning_rate"])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(config_dict["training_epochs"]):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / config_dict["batch_size"])


        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(
                config_dict["batch_size"])
            feed_dict = {X: batch_xs, Y: batch_ys}
            c, _ = sess.run([loss, train_step], feed_dict=feed_dict)
            avg_cost += c / total_batch


        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(
            avg_cost))

        correct_prediction = tf.equal(
                tf.argmax(mlp_model.hypothesis, axis=1), tf.argmax(Y, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('Accuracy:', sess.run(accuracy, feed_dict={
              X: mnist.test.images, Y: mnist.test.labels}))

if __name__ == "__main__":
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    X = tf.placeholder(tf.float32, [None, 784])
    Y = tf.placeholder(tf.float32, [None, 10])

    config_dict = {
        "learning_rate": 0.001,
        "training_epochs": 20,
        "batch_size": 128
        }

    train(config_dict)
