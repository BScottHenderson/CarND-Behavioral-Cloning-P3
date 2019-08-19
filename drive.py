import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from tensorflow.keras.models import load_model
import h5py
from tensorflow.keras import __version__ as keras_version


# Input image shape:
IMAGE_HEIGHT   = 160
IMAGE_WIDTH    = 320
IMAGE_CHANNELS = 3

# Cropping
CROP_TOP    = 70
CROP_BOTTOM = 25

# Low pass filter for steering angle adjustment. The value should be a percentage
# in the range [0.0, 1.0]. A value of 1.0 eliminates the previous steering angle
# in favor of the predicted value.
STEERING_SMOOTH = 0.35
prev_steering_angle = 0.


sio = socketio.Server()
app = Flask(__name__)
model = None


class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


controller = SimplePIController(0.1, 0.002)
set_speed = 9
controller.set_desired(set_speed)


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        global prev_steering_angle

        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)
#        predicted_steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))
        # Crop the input image since this is done in the model.
        predicted_steering_angle = float(model.predict(image_array[None, CROP_TOP:IMAGE_HEIGHT - CROP_BOTTOM, :, :], batch_size=1))

        # Low pass filter to smooth out steering changes:
        # -> Use the previously predicted steering angle.
        steering_angle = predicted_steering_angle if prev_steering_angle == 0.\
            else  (predicted_steering_angle * STEERING_SMOOTH) + (prev_steering_angle * (1. - STEERING_SMOOTH))
        prev_steering_angle = steering_angle

        # -> Use the telemetry steering angle.
        # For some reason the steering angle received via telemetry is not close to the
        # predicted steering angle. The resulting blended or "smoothed" steering angle
        # causes the vehicle to immediately begin turning circles even with using just
        # 10% of the telemetry steering angle (STEERING_SMOOTH = 0.9). With a smoothing
        # value of 0.99 (using just 1% of the incoming steering angle) the vehicle will
        # remain on the road but the effect seems to be to make the steering *less* smooth.
        # There is clearly something that I do not understand regarding the telemetry data.
        # steering_angle = float(steering_angle)
        # steering_angle = predicted_steering_angle if steering_angle == 0.\
        #     else  (predicted_steering_angle * STEERING_SMOOTH) + (steering_angle * (1. - STEERING_SMOOTH))

        # No filtering - just use the predicted steering angle directly.
        # steering_angle = predicted_steering_angle

        # Obtain the throttle setting from our simple PID controller based on current
        # telemetry speed. If it seemed necessary we could use the telemetry throttle
        # value and a low pass filter for smoothing - similar to that used for the steering
        # angle - to smooth throttle changes.
        throttle = controller.update(float(speed))

        # print('steering:{:+9.5f}, throttle:{:+9.5f}'.format(steering_angle, throttle))
        send_control(steering_angle, throttle)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    parser.add_argument(
        '--speed',
        type=int,
        nargs='?',
        default=0,
        help='Set the driving speed.'
    )
    args = parser.parse_args()

    # check that model Keras version is same as local Keras version
    f = h5py.File(args.model, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

    model = load_model(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # Adjust the speed.
    if args.speed > 0:
        controller.set_desired(args.speed)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
