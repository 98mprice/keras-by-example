# title: classify images of handwritten numbers 0 - 9
# author: @98mprice
# based on example by @fchollet https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

import os

from keras.datasets import mnist
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import Sequential
from keras.utils import to_categorical

import numpy as np

# loading dataset
'''
    mnist input data takes the form of:
    x_train, x_test: uint8 array of grayscale image data with shape (num_samples, 28, 28)
    y_train, y_test: uint8 array of digit labels (integers in range 0-9) with shape (num_samples,)
'''
(x_train, y_train), (x_validation, y_validation) = mnist.load_data()
'''
    you should have seperate training, validation & test sets, but to keep this example simple
    we're going to concatenate them and use keras' built in split() function to create
    a validation set
'''
x = np.concatenate((x_train, x_validation))
y = np.concatenate((y_train, y_validation))

# constants
batch_size = 128
num_classes = 10
epochs = 5
lrate = 0.001
img_x, img_y = 28, 28

# preprocessing
x = x.reshape(x.shape[0], img_x, img_y, 1)
x = x.astype('float32')
x /= 255

y = to_categorical(y, num_classes)

# model architecture
model = Sequential()
'''
    rectifier maps linearly for all positive numbers,
    and maps all negative numbers to 0
    i.e. f(x)=max(0,x)

'''
model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                 activation='relu',
                 input_shape=(img_x, img_y, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))

'''
    softmax returns an normalized array of size num_classes (which consequently sums to 1)
    therefore it's better suited to multi-class classification
'''
model.add(Dense(num_classes, activation='softmax'))

decay = lrate/epochs
'''
    categorical_crossentropy should be used for multi-class classification
'''
model.compile(loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])

model.summary()

# loading pre-existing model, if present
if os.path.exists("model_weights.h5"):
    model.load('model_weights.h5')

# train
model.fit(x, y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          shuffle=True,
          validation_split=0.3)

# evaluate
'''
    if you have a seperate test set, you should use it here
'''
if 'x_test' in globals():
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

# saving model
model.save('model_weights.h5')
