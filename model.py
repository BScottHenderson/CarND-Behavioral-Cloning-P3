# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 11:11:50 2018

@author: henders

Self-Driving Car Engineer Nanodegree

Project: Behavioral Cloning
"""

import sys
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda  # , Cropping2D
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dropout
from keras import regularizers

from image_augmentation import add_brightness, add_shadow, add_snow, add_rain, add_fog


#
# Hyperparameters
#

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

# Training parameters
DROPOUT           = 0.5
L2_REGULARIZATION = 0.1
VALIDATION_SPLIT  = 0.2
EPOCHS            = 5


#
# Local functions
#

def test_weather_augmentation(image):
    cv2.imwrite('./images/image.png', image)
    image_bright = add_brightness(image)
    cv2.imwrite('./images/image_0_bright.png', image_bright)
    image_shadow = add_shadow(image)
    cv2.imwrite('./images/image_1_shadow.png', image_shadow)
    image_snow = add_snow(image)
    cv2.imwrite('./images/image_2_snow.png', image_snow)
    image_rain = add_rain(image)
    cv2.imwrite('./images/image_3_rain.png', image_rain)
    image_fog = add_fog(image)
    cv2.imwrite('./images/image_4_fog.png', image_fog)
    image_torrential_rain = add_fog(add_rain(image))
    cv2.imwrite('./images/image_5_torrential_rain.png', image_torrential_rain)


def read_image(source_path):
    file_name = source_path.split('/')[-1]
    image_path = './data/IMG/' + file_name
    image = cv2.imread(image_path)
    return image


def read_and_crop_image(source_path):
    image = read_image(source_path)
    # Crop the top and bottom of each image. The top of each image
    # typically shows sky, trees, hills and other data not relevant
    # to steering while the bottom of each image shows the hood of
    # the car which is also not relevant to steering.
    # The purpose of this step is to reduce the amount of data used
    # for training to improve training speed.
    # Use numpy slicing to crop the image.
    # If (x1, y1) is the top/left corner and (x2, y2) is the
    # bottom/right corner, then:
    #   cropped_image = image[y1:y2, x1:x2]
    cropped_image = image[CROP_TOP:IMAGE_HEIGHT - CROP_BOTTOM, 0:IMAGE_WIDTH]
    return cropped_image


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


def NVIDIA2(model):
    model.add(Convolution2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Flatten())
    model.add(Dense(100, kernel_regularizer=regularizers.l2(L2_REGULARIZATION)))
    model.add(Dense(50, kernel_regularizer=regularizers.l2(L2_REGULARIZATION)))
    model.add(Dense(10, kernel_regularizer=regularizers.l2(L2_REGULARIZATION)))
    model.add(Dense(1, kernel_regularizer=regularizers.l2(L2_REGULARIZATION)))
    return model


#
# Main
#

def main(name=None):

    print('Name: {}'.format(name))

    print('Read driving log ...')
    lines = []
    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    print('Load image files ...')
    images = []
    measurements = []
    iterlines = iter(lines)
    next(iterlines)  # skip the first line
    for line in iterlines:
        images.append(read_and_crop_image(line[0]))  # center image
        images.append(read_and_crop_image(line[1]))  # left image
        images.append(read_and_crop_image(line[2]))  # right image

        steering = float(line[3])
        measurements.append(steering)  # center image
        measurements.append(steering + STEERING_CORRECTION)  # left image
        measurements.append(steering - STEERING_CORRECTION)  # right image

    print('Add augmented images ...')
    augmented_images, augmented_measurements = [], []
    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        # Mitigate the tendency of the model to turn left - since the
        # training track is basically an oval - by flipping the image
        # horizontally and changing the sign for the steering angle
        # measurement.
        augmented_images.append(cv2.flip(image, 1))  # flip around the y axis
        augmented_measurements.append(-measurement)
        # Add some weather - help prevent overfitting.
        augmented_images.append(add_brightness(image))
        augmented_measurements.append(measurement)
        augmented_images.append(add_shadow(image))
        augmented_measurements.append(measurement)
#        augmented_images.append(add_snow(image))
#        augmented_measurements.append(measurement)
#        augmented_images.append(add_rain(image))
#        augmented_measurements.append(measurement)
#        augmented_images.append(add_fog(image))
#        augmented_measurements.append(measurement)

    print('Train the model ...')
    X_train = np.array(augmented_images)
    y_train = np.array(augmented_measurements)

#    image_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
    cropped_image_shape = (CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH, IMAGE_CHANNELS)

    model = Sequential()
    # Preprocess incoming data, centered around zero with small standard deviation
    # data preprocessing: normalization and mean centering (shift mean to 0.0)
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=cropped_image_shape))

    # LeNet
#    model = LeNet(model)
#    model = NVIDIA(model)
    model = NVIDIA2(model)

    # Use 'mse' (mean squared error) rather than 'cross_entropy' because this is
    # a regression network rather than a classification network.
    # Use 'mae' (mean absolute error) rather than 'mse' to reduce the penalty
    # for wrong answers and make the model "take risks" so it will learn better.
    model.compile(loss='mae', optimizer='adam')
    history_object = model.fit(X_train, y_train,
                               validation_split=VALIDATION_SPLIT, shuffle=True,
                               epochs=EPOCHS)

    model.save('model.h5')

    # Plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    # plt.title('Model Mean Squared Error Loss')
    # plt.ylabel('Mean Squared Error Loss')
    plt.title('Model Mean Absolute Error Loss')
    plt.ylabel('Mean Absolute Error Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training Set', 'Validation Set'], loc='upper right')
    plt.savefig('Loss.png')
    plt.show()


if __name__ == '__main__':
    main(*sys.argv)
