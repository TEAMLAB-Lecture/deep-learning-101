
# Copyright 2017 TEMALAB. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

# import models.cnn.SimpleAlexNet
from models.mlp import SimpleModel
from models.cnn import SimpleLeNet
from models.vgg import SimpleVGGNet

flags = tf.app.flags
flags.DEFINE_string("model", "mlp.SimpleLeNet", "The directory of dog images [Images]")

def read_file_format(filename_queue):
    reader = tf.TFRecordReader()
    key, serialized = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized,
        features={
            'label': tf.FixedLenFeature([], tf.string),
            'images': tf.FixedLenFeature([], tf.string),
        })

      # Convert from a string to a vector of uint8 that is record_bytes long.
    record_image = tf.decode_raw(features['images'], tf.uint8)
    record_label = tf.decode_raw(features['label'], tf.uint8)

    image = tf.reshape(record_image, [128,128, 3])
    label = tf.reshape(record_label, [120])
    return image, label

def input_pipeline(filenames, batch_size=128, num_epochs=100,
        min_after_dequeue=512):
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs, shuffle=True)
    image, label = read_file_format(filename_queue)
    # min_after_dequeue
    # - 무작위로 샘플링 할 버퍼의 크기를 정의
    # - 크면 shuffling이 더 좋지만 느리게 시작되고 메모리가 많이 사용됨
    # capacity
    # min_after_dequeue보다 커야하며 더 큰 금액은 프리 페치 할 최대 값을 결정합니다.
    # 추천: min_after_dequeue + (num_threads + 약간의 여유값) * batch_size

    capacity = min_after_dequeue + 4 * batch_size
    image_batch, label_batch = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size,
            capacity=capacity, min_after_dequeue=min_after_dequeue)

    return  image_batch, label_batch

def train(image_batch, label_batch):
    net = SimpleVGGNet()
    with tf.Session() as sess:
        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer())
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        counter = 0

        try:
            while not coord.should_stop():
                # Run training steps or whatever

                _, c = sess.run([net.train_op, net.cost], feed_dict={
                   net.image_batch:sess.run(image_batch),
                   net.label_batch:sess.run(label_batch),
                   net.is_training:True
                   })

                print("Total cost :", c)
                counter += 1
                if counter % 5 == 0:
                    label,cost, pred = sess.run([net.label_batch, net.cost, net.pred], feed_dict={
                       net.image_batch:sess.run(image_batch),
                       net.label_batch:sess.run(label_batch),
                       net.is_training:False
                       })
                    correct_prediction = sess.run(tf.equal(
                        tf.argmax(pred, 1),
                        tf.argmax(label, 1)))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                    import numpy as np
                    print("Total cost :", np.argmax(pred, axis=1))
                    print("Total cost :", np.argmax(label, axis=1))
                    print("Total cost :", cost)
                    print("Accuracy ",counter, " :", sess.run(accuracy))

        except tf.errors.OutOfRangeError as e:
            print('Done training -- epoch limit reached')
            coord.request_stop(e)
        finally:
            print(counter)
            coord.request_stop()
            coord.join(threads)

def main(_):
    import os
    filenames = []
    TFRECORDS_DIR = "./tfrecords/cropping/train/"
    for file_name in os.listdir(TFRECORDS_DIR):
        filenames.append(os.path.join(TFRECORDS_DIR,file_name))

    image_batch, label_batch = input_pipeline(filenames, num_epochs=10000)
    train(image_batch, label_batch)


if __name__ == "__main__":
  tf.app.run()
