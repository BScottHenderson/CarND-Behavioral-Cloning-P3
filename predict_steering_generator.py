# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 11:11:50 2018

@author: henders
"""

import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D, MaxPooling2D


# Input image shape:
IMAGE_HEIGHT   = 160
IMAGE_WIDTH    = 320
IMAGE_CHANNELS = 3

# Cropping
CROP_TOP    = 70
CROP_BOTTOM = 25

# Cropped image shape:
CROPPED_IMAGE_HEIGHT = IMAGE_HEIGHT - CROP_TOP - CROP_BOTTOM
CROPPED_IMAGE_WIDTH  = IMAGE_WIDTH

# Could use trigonometry to find exact adjustment factor if we had enough
# data regarding the relative position and angle of the three cameras.
STEERING_CORRECTION = 0.2

# Data batch size for training
BATCH_SIZE = 32


def read_image(source_path):
    file_name = source_path.split('/')[-1]
    current_path = './data/IMG/' + file_name
    return cv2.imread(current_path)


def data_generator(lines, batch_size=32):
    line_count = len(lines)
    while 1:  # Loop forever so the generator never terminates
        shuffle(lines)
        for offset in range(0, line_count, batch_size):
            batch_lines = lines[offset:offset + batch_size]

            images = []
            angles = []
            for batch_line in batch_lines:
                images.append(read_image(line[0]))  # center image
                images.append(read_image(line[1]))  # left image
                images.append(read_image(line[2]))  # right image

                steering = float(line[3])
                angles.append(steering)  # center image
                angles.append(steering + STEERING_CORRECTION)  # left image
                angles.append(steering - STEERING_CORRECTION)  # right image

            # Crop the top and bottom of each image. The top of each image
            # typically shows sky, trees, hills and other data not relevant
            # to steering while the bottom of each image shows the hood of
            # the car which is also not relevant to steering.
            # The purpose of this step is to reduce the amount of data used
            # for training to improve training speed.
            cropped_images = []
            for image in images:
                # Use numpy slicing to crop the image.
                # If (x1, y1) is the top/left corner and (x2, y2) is the
                # bottom/right corner, then:
                #   cropped_image = image[y1:y2, x1:x2]
                cropped_image = image[CROP_TOP:IMAGE_HEIGHT - CROP_BOTTOM, 0:IMAGE_WIDTH]
                cropped_images.append(cropped_image)

            # Add augmented images. The purpose of this step is to attempt
            # to avoid overfitting the model and make it more general.
            augmented_images, augmented_angles = [], []
            for image, measurement in zip(cropped_images, angles):
                augmented_images.append(image)
                augmented_angles.append(measurement)
                # Mitigate the tendency of the model to turn left - since the
                # training track is basically an oval - by flipping the image
                # horizontally and changing the sign for the steering angle
                # measurement.
                augmented_images.append(cv2.flip(image, 1))  # flip around the y axis
                augmented_angles.append(-measurement)

            # Yield numpy arrays since that's what keras expects.
            X = np.array(augmented_images)
            y = np.array(augmented_angles)
            yield shuffle(X, y)


def LeNet(model):
    model.add(Convolution2D(6, (5, 5), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, (5, 5), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model


def NVIDIA(model):
    model.add(Convolution2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


print('Read driving log ...')
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # skip the header line
    for line in reader:
        lines.append(line)

train_lines, validation_lines = train_test_split(lines, test_size=0.2)

# Compile and train the model using the generator function
train_generator = data_generator(train_lines, batch_size=BATCH_SIZE)
validation_generator = data_generator(validation_lines, batch_size=BATCH_SIZE)

cropped_image_shape = (CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH, IMAGE_CHANNELS)

print('Train the model ...')
model = Sequential()
# data preprocessing: normalization and mean centering (shift mean to 0.0)
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=cropped_image_shape))
# Preprocess incoming data, centered around zero with small standard deviation
# model.add(Lambda(lambda x: (x / 127.5) - 1., input_shape=cropped_image_shape))

# model = LeNet(model)
model = NVIDIA(model)

# Use 'mse' (mean squared error) rather than 'cross_entropy' because this is
# a regression network rather than a classification network.
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_lines) / BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=len(validation_lines) / BATCH_SIZE,
        epochs=5, verbose=1)

model.save('model.h5')

# print the keys contained in the history object
print(history_object.history.keys())

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('Model Mean Squared Error Loss')
plt.ylabel('Mean Squared Error Loss')
plt.xlabel('Epoch')
plt.legend(['Training Set', 'Validation Set'], loc='upper right')
plt.show()
