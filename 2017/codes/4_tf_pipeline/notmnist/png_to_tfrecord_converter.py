import os
import numpy as np
import tensorflow as tf

import image_preprocessing_util as iputil

import pprint
pp = pprint.PrettyPrinter()

from scipy.misc import imsave

TRAIN  = "train"
TEST  = "test"

flags = tf.app.flags
flags.DEFINE_string("image_dir", "Images", "The directory of dog images [Images]")
flags.DEFINE_string("output_dir", "tfrecords", "The directory of tfrecord_output [tfrecords]")
flags.DEFINE_float("test_ratio", "0.2", "The ratio of test image data set [0.8]")

FLAGS = flags.FLAGS

def get_total_data():
    IMAGE_DIR = FLAGS.image_dir

    bleeds = os.listdir(IMAGE_DIR)
    image_set = {}

    for bleed in bleeds:
        image_dir = os.path.join(IMAGE_DIR, bleed)
        image_set[bleed] = os.listdir(image_dir)

        total_data = []
        for bleed in image_set.keys():
            total_data.extend([ [filename,bleed] for filename in image_set[bleed]])
        total_data = np.array(total_data)
    return total_data

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _get_target_dir():
    TAGET_DIR = "general"

    return TAGET_DIR

def get_splitted_data(total_data):
    """
    Returns:
    """

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
         total_data[:,0], total_data[:,1], test_size=FLAGS.test_ratio, random_state=1)
    return X_train, X_test, y_train, y_test

def generate_patches():
    with open('testfile.txt', 'r') as f:
        for patch in f.readlines():
            yield patch[:-1]

def persistence_image_data_to_tfrecords(
    x_data, y_data, data_type, split_index=128):

    TAGET_DIR = _get_target_dir()
    OUTPUT_DIR = os.path.join(FLAGS.output_dir,TAGET_DIR,data_type)
    IMAGE_DIR = FLAGS.image_dir

    if not(os.path.exists(OUTPUT_DIR)):
        os.makedirs(OUTPUT_DIR)
        print("Directory create : {0}".format(OUTPUT_DIR,))

    writer = None
    sess = None
    current_index = 0

    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(y_data)
    y_data_size = len(le.classes_)

    # https://stackoverflow.com/questions/45427637/is-there-a-more-simple-way-to-handle-batch-inputs-from-tfrecords

    for images_filename, y_label in zip(x_data, y_data):
        if not(images_filename[-3:] == "png"):
            print("Error - ", images_filename)
            continue
        if current_index % split_index == 0:
            if writer:
                writer.close()
            if sess:
                sess.close()
            tf.reset_default_graph()
            graph = tf.get_default_graph()
            sess = tf.Session(graph=graph)
            sess.run(tf.global_variables_initializer())

            record_filename = "{output_dir}/{data_type}-{current_index}.tfrecords".format(
                output_dir=OUTPUT_DIR, data_type=data_type, current_index=current_index
            )
            print("=============>" , record_filename)
            print("current index : {0}".format(current_index,))
            writer = tf.python_io.TFRecordWriter(record_filename)

        file_full_path = os.path.join(
                IMAGE_DIR, y_label,  images_filename)
        try:
            image_file = tf.read_file(file_full_path)
            image = tf.image.decode_png(image_file)
        except tf.errors.InvalidArgumentError as e:
            print(e)
            print("Error : ", images_filename)
            continue

        image_list = [image]

        for image in image_list:
            try:
                image_bytes = sess.run(
                    tf.cast(image, tf.uint8))
                image_bytes = image_bytes.tobytes()

                y_data_label = le.transform([y_label])
                lbl_one_hot = tf.one_hot(y_data_label[0], y_data_size, 1, 0)
                image_label = sess.run(tf.cast(lbl_one_hot, tf.uint8))
                image_label = image_label.tobytes()


                feature = {'label': _bytes_feature(image_label),
                            'image': _bytes_feature(image_bytes)}

                example = tf.train.Example(
                        features = tf.train.Features(
                                            feature=feature))

                writer.write(example.SerializeToString())
                current_index += 1
            except tf.errors.InvalidArgumentError as e:
                print(e)
                print("Error : ", images_filename)
                continue

    writer.close()

def main(_):
    print ('Converting PNG to tfrecord datatype')
    print ('Argument setup')
    pp.pprint(flags.FLAGS.__flags)
    print ('---------------------------------')

    total_data = get_total_data()
    number_of_data_types = len(np.unique(total_data[:, 1]))
    print("The number of data : {0}".format(total_data.shape[0],))
    print("The number of bleeds : {0}".format(number_of_data_types,))

    print('---------------------------------')
    X_train, X_test, y_train, y_test = get_splitted_data(total_data)
    print("Train / Test ratio : {0:.2f} / {1:.2f}".format( 1-FLAGS.test_ratio, FLAGS.test_ratio ))
    print("Number of train data set : {0}".format(len(X_train)))
    print("Number of test data set : {0}".format(len(X_test)))


    print('---------------------------------')

    persistence_image_data_to_tfrecords(X_train, y_train, data_type=TRAIN, split_index=128)
    persistence_image_data_to_tfrecords(X_test, y_test, data_type=TEST, split_index=128)


if __name__ =="__main__":
    tf.app.run()
