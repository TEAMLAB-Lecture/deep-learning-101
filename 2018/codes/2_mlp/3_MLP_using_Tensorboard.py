import tensorflow as tf
import random
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(777)  # reproducibility
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print ("What does the data of MNIST look like?")
trainimg   = mnist.train.images
trainlabel = mnist.train.labels
testimg    = mnist.test.images
testlabel  = mnist.test.labels
print (" type of 'trainimg' is %s"    % (type(trainimg)))
print (" type of 'trainlabel' is %s"  % (type(trainlabel)))
print (" type of 'testimg' is %s"     % (type(testimg)))
print (" type of 'testlabel' is %s"   % (type(testlabel)))
print (" shape of 'trainimg' is %s"   % (trainimg.shape,))
print (" shape of 'trainlabel' is %s" % (trainlabel.shape,))
print (" shape of 'testimg' is %s"    % (testimg.shape,))
print (" shape of 'testlabel' is %s"  % (testlabel.shape,))


# ## parameters
learning_rate = 0.001
training_epochs = 20
batch_size = 100

tf.summary.FileWriterCache.clear()


# ## input place holders
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# ## weights & bias for nn layers
with tf.variable_scope("layer_1", reuse=tf.AUTO_REUSE):
    W1 = tf.get_variable(
        "w1", shape=[784, 128], initializer=tf.random_normal_initializer())
    b1 = tf.get_variable(
        "b1", shape=[128], initializer=tf.random_normal_initializer())
    L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

with tf.variable_scope("layer_2", reuse=tf.AUTO_REUSE):
    W2 = tf.get_variable(
        "w2", shape=[128, 64], initializer=tf.random_normal_initializer())
    b2 = tf.get_variable(
        "b2", shape=[64], initializer=tf.random_normal_initializer())
    L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)


with tf.variable_scope("layer_3", reuse=tf.AUTO_REUSE):
    W3 = tf.get_variable(
        "w3", shape=[64, 10], initializer=tf.random_normal_initializer())
    b3 = tf.get_variable(
        "b3", shape=[10], initializer=tf.random_normal_initializer())
    hypothesis = tf.matmul(L2, W3) + b3
    y_historgram = tf.summary.histogram("activation",hypothesis)



# ## define cost/loss & optimizer
with tf.variable_scope("cost", reuse=tf.AUTO_REUSE):
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y), name ="softmax_cross_entropy")
with tf.variable_scope("train", reuse=tf.AUTO_REUSE):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.name_scope("accuracy"):
    # Test trained model
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(hypothesis, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_scalar = tf.summary.scalar("accuracy",accuracy)

saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# deprecated
# writer = tf.train.SummaryWriter(
#     "./MLP_tensorflow", graph=tf.get_default_graph())
# merged = tf.merged_all_summeries()
# merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('board_beginner')  # create writer
writer.add_graph(sess.graph)

step = 0
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
        step += 1

        if step % 200 == 0:
            sum2 = sess.run(accuracy_scalar, feed_dict=feed_dict)
            sum3 = sess.run(y_historgram, feed_dict=feed_dict)
            writer.add_summary(sum2,step)
            writer.add_summary(sum3,step)


    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    correct_prediction = tf.equal(tf.argmax(hypothesis, axis=1), tf.argmax(Y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy:', sess.run(accuracy, feed_dict={
          X: mnist.test.images, Y: mnist.test.labels}))


print('Learning Finished!')



## Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: mnist.test.images, Y: mnist.test.labels}))

# Get one and predict
r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(
    tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

tf.summary.merge_all()
writer.close()
