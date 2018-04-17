import tensorflow as tf
from model import MLP
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
    X = tf.placeholder(tf.float32, [None, 28*28])
    Y = tf.placeholder(tf.float32, [None, 10])

    n_in = 28*28
    n_hiddens = config_dict["n_hiddens"]
    n_out = 10

    mlp_model = MLP(X, n_hiddens=n_hiddens, n_in=n_in, n_out=n_out)
    loss = loss_function(mlp_model.hypothesis, Y)
    train_step = training(loss, learning_rate=config_dict["learning_rate"])

    train_data = np.load("notMNIST_large_train.npy")
    train_features = train_data["features"]
    train_labels = train_data["labels"]

    train_dataset = tf.data.Dataset.from_tensor_slices(
                                        (train_features, train_labels))
    train_dataset = train_dataset.shuffle(buffer_size=10000)
    train_dataset = train_dataset.batch(config_dict["batch_size"])

    iterator = tf.data.Iterator.from_structure(
                            train_dataset.output_types,
                            train_dataset.output_shapes)
    next_element = iterator.get_next()
    train_init_op = iterator.make_initializer(train_dataset)

    test_data = np.load("notMNIST_large_test.npy")
    test_features = test_data["features"]
    test_labels = test_data["labels"]
    test_dataset = tf.data.Dataset.from_tensor_slices(
                                        (test_features, test_labels))

    test_dataset = test_dataset.batch(test_labels.shape[0])

    test_iterator = tf.data.Iterator.from_structure(
                            test_dataset.output_types,
                            test_dataset.output_shapes)
    test_next_element = test_iterator.get_next()
    test_init_op = test_iterator.make_initializer(test_dataset)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(config_dict["training_epochs"]):
            # initialize the iterator on the training data
            sess.run(train_init_op)

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

            sess.run(test_init_op)
            while True:
                try:
                    batch = sess.run(test_next_element)
                    batch_x = batch[0]
                    batch_y = batch[1]

                    correct_prediction = tf.equal(
                            tf.argmax(mlp_model.hypothesis, axis=1),
                            tf.argmax(Y, axis=1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    print('Accuracy:', sess.run(accuracy, feed_dict={
                          X: batch_x, Y: batch_y}))
                except tf.errors.OutOfRangeError:
                    break


if __name__ == "__main__":
    config_dict = {
        "learning_rate": 0.001,
        "n_hiddens": [1024, 256],
        "training_epochs": 300,
        "batch_size": 256
        }

    train(config_dict)
