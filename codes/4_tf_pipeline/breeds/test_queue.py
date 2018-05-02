import tensorflow as tf

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

    image = tf.reshape(record_image, [256, 256, 1])
    label = tf.reshape(record_label, [120])
    return image, label

def input_pipeline(filenames, batch_size=128, num_epochs=None,
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

# 이 값은 입력 받아야 할 듯
filenames=[]

import os
for file_name in os.listdir("./tfrecords/train/"):
    filenames.append(os.path.join("./tfrecords/train/",file_name))

image_batch, label_batch = input_pipeline(filenames,num_epochs=10000)

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
            example,label = sess.run([image_batch,label_batch])
            print(example.shape)
            # print(example)
            print(label.shape)
            print(label)

    except tf.errors.OutOfRangeError as e:
        print('Done training -- epoch limit reached')
        coord.request_stop(e)
    finally:
        print(counter)
        coord.request_stop()
        coord.join(threads)
