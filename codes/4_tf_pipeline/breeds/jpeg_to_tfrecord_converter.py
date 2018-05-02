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
flags.DEFINE_boolean("cropping", "True", "The boolean vairable of dog faces cropping [True]")
flags.DEFINE_integer("image_height", "128", "The boolean vairable of dog faces cropping [128]")
flags.DEFINE_integer("image_width", "128", "The boolean vairable of dog faces cropping [128]")
flags.DEFINE_boolean("image_adjusted", "False", "The boolean vairable expressing whether or not to reduce the image without distorting the image according to the face size of the dog [False]")
flags.DEFINE_boolean("image_augumentation", "False", "The boolean vairable of generating image data added with random distortion, upside-downside, side-to-side reversal, etc. [False]")
flags.DEFINE_float("test_ratio", "0.2", "The ratio of test image data set [0.8]")

FLAGS = flags.FLAGS

def get_total_data():
    """ image가 저장된 폴더로 부터 모든 JPEG를 가져와서 [파일명, 개품중(Bleed)]의 형태의 Numpy Array를 생성함
    image폴더의 기본 저장형태는 "Images\개품종명\파일명" 형태로 저장되어 있다고 가정함

    Returns:
        Numpy.ndarray: [[파일명, name_of_dog_bleed]]
        [['n02085620_10074.jpg', 'n02085620-Chihuahua'],
        ['n02085620_10131.jpg', 'n02085620-Chihuahua'],
        ['n02085620_10621.jpg', 'n02085620-Chihuahua']

    """
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

def _get_target_dir():
    if FLAGS.image_augumentation:
        TAGET_DIR = "augumentation"
    elif FLAGS.image_adjusted:
        TAGET_DIR = "adjusted"
    elif FLAGS.cropping:
        TAGET_DIR = "cropping"
    else:
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

def persistence_image_data_to_tfrecords(x_data, y_data, data_type,
        split_index=128):
    """
    Returns:
    """

    TAGET_DIR = _get_target_dir()
    OUTPUT_DIR = os.path.join(FLAGS.output_dir,TAGET_DIR,data_type)
    IMAGE_DIR = FLAGS.image_dir

    if not(os.path.exists(OUTPUT_DIR)):
        os.makedirs(OUTPUT_DIR)
        print("Directory create : {0}".format(OUTPUT_DIR,))

    if not(os.path.exists("./resized_image")):
        os.makedirs("./resized_image")

    writer = None
    sess = None
    current_index = 0

    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(y_data)
    y_data_size = len(le.classes_)

    for images_filename,breed in zip(x_data,y_data):
        if not(images_filename[-3:] == "jpg"):
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

        file_full_path = os.path.join(IMAGE_DIR, breed,  images_filename)
        image_file = tf.read_file(file_full_path)
        try:
            # print(file_full_path)
            image = tf.image.decode_jpeg(image_file)
        except InvalidArgumentError as e:
            print(e)
            print("Error : ", images_filename)
            continue

        image_list = [image]
        if FLAGS.cropping:
            image_list = []
            size_info = iputil.get_orginal_size_info(breed, images_filename)
            target_image = tf.image.resize_images(
                image, [size_info["width"], size_info["height"]])
            bounding_info = iputil.get_bounding_size_info(breed, images_filename)
            for box in bounding_info:
                image = tf.image.crop_to_bounding_box(
                    target_image,
                    box[0],box[1],box[2],box[3])
                image_list.append(image)

        for image in image_list:
            # grayscale_image = tf.image.rgb_to_grayscale(image)
            grayscale_image = image
            resized_image = tf.image.resize_images(
                grayscale_image, [FLAGS.image_width, FLAGS.image_height])

            image_bytes = sess.run(tf.cast(resized_image, tf.uint8)).tobytes()
            y_data_label = le.transform([breed])

            lbl_one_hot = tf.one_hot(y_data_label[0], y_data_size, 1.0, 0.0)
            image_label = sess.run(tf.cast(lbl_one_hot, tf.uint8)).tobytes()


            imsave("./resized_image/" +images_filename+"_"+ str(current_index) +".jpeg", sess.run(resized_image))



            example = tf.train.Example(features = tf.train.Features(
                                        feature={'label':
                                                  tf.train.Feature(bytes_list=tf.train.BytesList(
                                                      value=[image_label])),
                                                  "images":
                                                  tf.train.Feature(bytes_list=tf.train.BytesList(
                                                      value=[image_bytes]))
                                                 }
                                      ))
            writer.write(example.SerializeToString())
            current_index += 1
    writer.close()

def main(_):
    print ('Converting JPEG to tfrecord datatype')
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
    persistence_image_data_to_tfrecords(X_train, y_train, data_type=TRAIN)
    persistence_image_data_to_tfrecords(X_test, y_test, data_type=TEST)

    #TODO - Google Detection API 써서 실험 먼저 해보기
    #TODO - Test에도 공동적용해야할 내용 ==> data resize + adjustable
    #TODO - 데이터 리사이즈 ==> adjustable에 맞춰 처리함
    #TODO - Training 데이터의 data augumentation을 어떻게 할 것인가?
    #TODO - config.py 파일을 따로 만들던가, 따로 config 정보를 저장하는 공간을 만들자


if __name__ =="__main__":
    tf.app.run()
