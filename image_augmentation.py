# -*- coding: utf-8 -*-
"""
Image Augmentation: Make it rain, make it snow. How to modify photos to train self-driving cars
by Ujjwal Saxena

https://medium.freecodecamp.org/image-augmentation-make-it-rain-make-it-snow-how-to-modify-a-photo-with-machine-learning-163c0cb3843f
"""

import numpy as np
import cv2


#
# Sunny and Shady
#
def add_brightness(image):
    """
    The brightness of an image can be changed by changing the pixel values
    of “Lightness”- channel 1 of image in HLS color space. Converting the
    image back to RGB gives the same image with enhanced or suppressed
    lighting.
    """
    # Conversion to HLS
    image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    image_HLS = np.array(image_HLS, dtype=np.float64)
    # generates value between 0.5 and 1.5
    random_brightness_coefficient = np.random.uniform() + 0.5
    # scale pixel values up or down for channel 1 (Lightness)
    image_HLS[:, :, 1] = image_HLS[:, :, 1] * random_brightness_coefficient
    # Sets all values above 255 to 255
    image_HLS[:, :, 1][image_HLS[:, :, 1] > 255] = 255
    image_HLS = np.array(image_HLS, dtype=np.uint8)
    # Conversion to RGB
    image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)
    return image_RGB


#
# Shadows
#
def generate_shadow_coordinates(imshape, no_of_shadows=1):
    vertices_list = []
    for index in range(no_of_shadows):
        vertex = []
        # Dimensionality of the shadow polygon
        for dimensions in range(np.random.randint(3, 15)):
            vertex.append((imshape[1] * np.random.uniform(), imshape[0] // 3 + imshape[0] * np.random.uniform()))
        # Vertices for a single shadow polygon.
        vertices = np.array([vertex], dtype=np.int32)
        vertices_list.append(vertices)
    # List of shadow vertices.
    return vertices_list


def add_shadow(image, no_of_shadows=1):
    # Conversion to HLS
    image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    # Add shadows to an initially empty mask image.
    mask = np.zeros_like(image)
    # Getting list of shadow vertices.
    vertices_list = generate_shadow_coordinates(image.shape, no_of_shadows)
    for vertices in vertices_list:
        # Adding all shadow polygons on empty mask, single 255 denotes
        # only red channel
        cv2.fillPoly(mask, vertices, 255)
    # Use the mask to adjust pixels in the original image.
    # If red channel is hot, the image's "Lightness" channel's brightness
    # is lowered
    image_HLS[:, :, 1][mask[:, :, 0] == 255] = image_HLS[:, :, 1][mask[:, :, 0] == 255] * 0.5
    # Conversion to RGB
    image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)
    return image_RGB


#
# Snow
#
def add_snow(image):
    # Conversion to HLS
    image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    image_HLS = np.array(image_HLS, dtype=np.float64)
    brightness_coefficient = 2.5
    snow_point = 140  # increase this for more snow
    # scale pixel values up for channel 1 (Lightness)
    image_HLS[:, :, 1][image_HLS[:, :, 1] < snow_point] = image_HLS[:, :, 1][image_HLS[:, :, 1] < snow_point] * brightness_coefficient
    # Sets all values above 255 to 255
    image_HLS[:, :, 1][image_HLS[:, :, 1] > 255] = 255
    image_HLS = np.array(image_HLS, dtype=np.uint8)
    # Conversion to RGB
    image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)
    return image_RGB


#
# Rain
#
def generate_random_lines(imshape, slant, drop_length):
    drops = []
    for i in range(500):  # If You want heavy rain, try increasing this
        if slant < 0:
            x = np.random.randint(slant, imshape[1])
        else:
            x = np.random.randint(0, imshape[1] - slant)
        y = np.random.randint(0, imshape[0] - drop_length)
        drops.append((x, y))
    return drops


def add_rain(image):
    # Generate rain drops as lines.
    slant_extreme = 10
    slant = np.random.randint(-slant_extreme, slant_extreme)
    drop_length = 10
    drop_width = 2
    drop_color = (200, 200, 200)  # a shade of gray
    rain_drops = generate_random_lines(image.shape, slant, drop_length)

    # Add rain drops to the image.
    for rain_drop in rain_drops:
        cv2.line(image, (rain_drop[0], rain_drop[1]),
                        (rain_drop[0] + slant, rain_drop[1] + drop_length),
                 drop_color, drop_width)
    image = cv2.blur(image, (7, 7))  # rainy views are blurry

    # Darken the image a bit - rainy days are usually shady
    brightness_coefficient = 0.7
    # Conversion to HLS
    image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    # scale pixel values down for channel 1 (Lightness)
    image_HLS[:, :, 1] = image_HLS[:, :, 1] * brightness_coefficient
    # Conversion to RGB
    image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)
    return image_RGB


#
# Fog
#
def generate_random_blur_coordinates(imshape, hw):
    blur_points = []
    midx = imshape[1] // 2 - hw - 100
    midy = imshape[0] // 2 - hw - 100
    # radially generating coordinates
    index = 1
    while (midx > -100 or midy > -100):
        for i in range(250 * index):
            x = np.random.randint(midx, imshape[1] - midx - hw)
            y = np.random.randint(midy, imshape[0] - midy - hw)
            blur_points.append((x, y))
        midx -= 250 * imshape[1] // sum(imshape)
        midy -= 250 * imshape[0] // sum(imshape)
        index += 1
    return blur_points


def add_blur(image, x, y, hw):
    image[y:y + hw, x:x + hw, 1] = image[y:y + hw, x:x + hw, 1] + 1
    # Sets all values above 255 to 255
    image[:, :, 1][image[:, :, 1] > 255] = 255
    image[y:y + hw, x:x + hw, 1] = cv2.blur(image[y:y + hw, x:x + hw, 1], (5, 5))
    return image


def add_fog(image):
    # Conversion to HLS
    image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hw = 100
    image_HLS[:, :, 1] = image_HLS[:, :, 1] * 0.8
    haze_list = generate_random_blur_coordinates(image.shape, hw)
    for haze_points in haze_list:
        # Sets all values above 255 to 255
        image_HLS[:, :, 1][image_HLS[:, :, 1] > 255] = 255
        image_HLS = add_blur(image_HLS, haze_points[0], haze_points[1], hw)
    # Conversion to RGB
    image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)
    return image_RGB
