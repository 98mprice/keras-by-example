# title: classify images of cats vs dogs
# author: @98mprice
# based on example by @fchollet https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

import os

from keras.datasets import mnist
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import Sequential
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

import numpy as np

# constants
batch_size = 128
epochs = 10
lrate = 0.001
img_x, img_y = 150, 150

# loading dataset
datagen = ImageDataGenerator()
training_set = datagen.flow_from_directory('/Volumes/Untitled/dogsvscats/all/train',
                                target_size=(img_x, img_y),
                                batch_size=batch_size,
                                class_mode='binary')

# model architecture
model = Sequential()
'''
    rectifier maps linearly for all positive numbers,
    and maps all negative numbers to 0
    i.e. f(x)=max(0,x)

'''
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=(img_x, img_y, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))

'''
    sigmoid maps values to a single probability between 0 - 1,
    making it better suited for single class classification problems
    note: it may also be suitable for multi-class classification, depending on your problem
'''
model.add(Dense(activation='sigmoid'))

decay = lrate/epochs
'''
    binary classification should be used for binary classification
    note: it should also be used for multi-label classification (one-hot encoding)
'''
model.compile(loss='binary_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])

model.summary()

# loading pre-existing model, if present
if os.path.exists("model_weights.h5"):
    model.load('model_weights.h5')

# train
model.fit_generator(training_set,
          epochs=epochs,
          verbose=1,
          shuffle=True,
          validation_steps = 2000)

# evaluate
'''
    if you have a seperate test set, you should use it here
'''
if x_test in globals():
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

# saving model
model.save('model_weights.h5')
