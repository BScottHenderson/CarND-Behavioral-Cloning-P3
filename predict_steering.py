# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 11:11:50 2018

@author: henders

Self-Driving Car Engineer Nanodegree

Project: Behavioral Cloning
"""

import sys
import os
import csv
import random
from tqdm import tqdm
import cv2
import click
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Lambda  # , Cropping2D
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers

from image_augmentation import add_brightness, add_shadow, add_snow, add_rain, add_fog


DATA_DIR = './data/'
TEST_IMAGE_DIR = './test_images'


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

def test_weather_augmentation(img):
    cv2.imwrite(os.path.join(TEST_IMAGE_DIR, 'image.png'), img)
    cv2.imwrite(os.path.join(TEST_IMAGE_DIR, 'image_0_bright.png'), add_brightness(img))
    cv2.imwrite(os.path.join(TEST_IMAGE_DIR, 'image_1_shadow.png'), add_shadow(img))
    cv2.imwrite(os.path.join(TEST_IMAGE_DIR, 'image_2_snow.png'), add_snow(img))
    cv2.imwrite(os.path.join(TEST_IMAGE_DIR, 'image_3_rain.png'), add_rain(img))
    cv2.imwrite(os.path.join(TEST_IMAGE_DIR, 'image_4_fog.png'), add_fog(img))
    cv2.imwrite(os.path.join(TEST_IMAGE_DIR, 'image_5_torrential_rain.png'), add_fog(add_rain(img)))


def read_and_crop_image(file_name):
    """
    Crop the top and bottom of each image. The top of each image typically shows sky, trees,
    hills and other data not relevant to steering while the bottom of each image shows the
    hood of the car which is also not relevant to steering.

    The purpose of this step is to reduce the amount of data used for training to improve
    training speed.
    """
    img = cv2.imread(os.path.join(DATA_DIR, file_name))
    # Use numpy slicing to crop the image.
    # If (x1, y1) is the top/left corner and (x2, y2) is the
    # bottom/right corner, then:
    #   cropped_img = img[y1:y2, x1:x2]
    cropped_img = img[CROP_TOP:IMAGE_HEIGHT - CROP_BOTTOM, 0:IMAGE_WIDTH]
    return cropped_img


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


def NVIDIA(model, dropout=False):
    model.add(Convolution2D(24, (5, 5), strides=(2, 2), activation='relu'))
    if dropout:
        model.add(Dropout(DROPOUT))
    model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation='relu'))
    if dropout:
        model.add(Dropout(DROPOUT))
    model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation='relu'))
    if dropout:
        model.add(Dropout(DROPOUT))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    if dropout:
        model.add(Dropout(DROPOUT))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    if dropout:
        model.add(Dropout(DROPOUT))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


#
# Main
#

@click.command()
@click.option('--test-augmentation', '-t', is_flag=True, default=False)
@click.option('--model-type', type=click.Choice(['LeNet', 'NVIDIA']), required=True)
@click.option('--dropout', '-d', is_flag=True, default=False)
@click.option('--epochs', '-e', type=int, default=EPOCHS)
def main(test_augmentation=None, model_type=None, dropout=None, epochs=None):

    print('Name: {}'.format(__file__))

    print('Read driving log ...')
    driving_log = []
    with open(os.path.join(DATA_DIR, 'driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            driving_log.append(line)

    # Write test augmentation images.
    if test_augmentation:
        print('Write image augmentation test files ...')
        index = np.random.randint(1, len(driving_log)) # Skip line 0 due to header.
        center_image_file = driving_log[index][0]
        print('Augmentation test image file: {}'.format(center_image_file))
        center_img = cv2.imread(os.path.join(DATA_DIR, center_image_file))
        test_weather_augmentation(center_img)
        sys.exit() # Don't do anything else if we're testing image augmentation.

    print('Load image files (w/crop) ...')
    images = []
    measurements = []
    iterlines = iter(driving_log)
    next(iterlines)  # skip the first line
    for line in tqdm(iterlines):
        images.append(read_and_crop_image(line[0].strip()))  # center image
        images.append(read_and_crop_image(line[1].strip()))  # left image
        images.append(read_and_crop_image(line[2].strip()))  # right image

        steering = float(line[3])
        measurements.append(steering)  # center image
        measurements.append(steering + STEERING_CORRECTION)  # left image
        measurements.append(steering - STEERING_CORRECTION)  # right image

    print('Add augmented images [{}] ...'.format(len(images)))
    augmented_images, augmented_measurements = [], []
    for img, measurement in tqdm(zip(images, measurements)):
        # Keep the original - unmodified image - in the augmented images list.
        augmented_images.append(img)
        augmented_measurements.append(measurement)

        # Mitigate the tendency of the model to turn left - since the training track
        # is basically an oval - by flipping the image  horizontally and changing the
        # sign for the steering angle measurement.
        augmented_images.append(cv2.flip(img, 1))  # flip around the y axis
        augmented_measurements.append(-measurement)

        # Help with overfitting:
        # Adjust light levels.
        augmented_images.append(add_brightness(img))
        augmented_measurements.append(measurement)
        augmented_images.append(add_shadow(img))
        augmented_measurements.append(measurement)
        # Add some random weather.
        weather = random.randrange(3)
        if weather == 0:
            augmented_images.append(add_snow(img))
            augmented_measurements.append(measurement)
        elif weather == 1:
            augmented_images.append(add_rain(img))
            augmented_measurements.append(measurement)
        elif weather == 2:
            augmented_images.append(add_fog(img))
            augmented_measurements.append(measurement)

    print('Train the model ...')
    X_train = np.array(augmented_images)
    y_train = np.array(augmented_measurements)

    # image_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
    cropped_image_shape = (CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH, IMAGE_CHANNELS)

    model = Sequential()
    # Preprocess incoming data, centered around zero with small standard deviation
    # data preprocessing: normalization and mean centering (shift mean to 0.0)
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=cropped_image_shape))
    # model.add(Lambda(lambda x: (x / 127.5) - 1., input_shape=cropped_image_shape))

    # Create the model.
    if model_type == 'LeNet':
        model = LeNet(model)
    elif model_type == 'NVIDIA':
        model = NVIDIA(model, dropout=dropout)
    else:
        print('Unrecognized model type: {}'.format(model_type))
        model = None # Cause an exception below.

    # Use 'mse' (mean squared error) rather than 'cross_entropy' because this is
    # a regression network rather than a classification network.
    # Use 'mae' (mean absolute error) rather than 'mse' to reduce the penalty
    # for wrong answers and make the model "take risks" so it will learn better.
    model.compile(loss='mae', optimizer='adam')
    history_object = model.fit(X_train, y_train,
                               validation_split=VALIDATION_SPLIT, shuffle=True,
                               epochs=epochs)

    # Save the model so it can be used for driving.
    model.save(model_type + '.h5')

    # Plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('Model Mean Absolute Error Loss')
    plt.ylabel('Mean Absolute Error Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training Set', 'Validation Set'], loc='upper right')
    plt.savefig('Loss.png')
    # plt.show()


if __name__ == '__main__':
    main()
