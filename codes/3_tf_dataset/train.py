import tensorflow as tf
from model import MLP
import tensorflow as tf
import numpy as np


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

    X = tf.placeholder(tf.float32, [None, 28*28])
    Y = tf.placeholder(tf.float32, [None, 10])


    n_in = 28*28
    n_hiddens = config_dict["n_hiddens"]
    n_out = 10

    mlp_model = MLP(X, n_hiddens=n_hiddens, n_in=n_in, n_out=n_out)
    loss = loss_function(mlp_model.hypothesis, Y)
    train_step = training(loss, learning_rate=config_dict["learning_rate"])


    data = np.load("notMNIST_large.npy")
    features = data["features"]
    labels = data["labels"]

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(config_dict["batch_size"])

    iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                            dataset.output_shapes)
    next_element = iterator.get_next()
    init_op = iterator.make_initializer(dataset)


    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(config_dict["training_epochs"]):
            # initialize the iterator on the training data
            sess.run(init_op)

            # get each element of the training dataset until the end is reached
            cost = 0
            total_batch = 0
            while True:
                try:
                    batch = sess.run(next_element)
                    batch_xs = batch[0]
                    batch_ys = batch[1]

                    feed_dict = {X: batch_xs, Y: batch_ys}
                    c, _ = sess.run([loss, train_step], feed_dict=feed_dict)
                    cost += c
                    total_batch += 1

                except tf.errors.OutOfRangeError:
                    print("End of training dataset.")
                    break

            avg_cost = cost / total_batch
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(
                    avg_cost))

            correct_prediction = tf.equal(
                    tf.argmax(mlp_model.hypothesis, axis=1), tf.argmax(Y, axis=1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print('Accuracy:', sess.run(accuracy, feed_dict={
                  X: features, Y: labels}))

if __name__ == "__main__":
    config_dict = {
        "learning_rate": 0.001,
        "n_hiddens": [256, 128],
        "training_epochs": 20,
        "batch_size": 256
        }

    train(config_dict)
