

from __future__ import print_function

from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation


batch_size = 128
num_classes = 10
epochs = 20


(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_axis, y_axis = x_train[0].shape

# import matplotlib.pyplot as plt
# plt.gray()
# plt.imshow(x_train[0])
# plt.show()


x_train = x_train.reshape(60000, x_axis*y_axis)
x_test = x_test.reshape(10000, x_axis*y_axis)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


x_train /= 255 #Normalization
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')



x_train[0]



import keras
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(784,)))
# model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer="sgd",
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)


print('Test loss:', score[0])
print('Test accuracy:', score[1])
