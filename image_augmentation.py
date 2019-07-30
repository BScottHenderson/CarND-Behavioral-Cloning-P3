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

def add_brightness(img):
    """
    The brightness of an image can be changed by changing the pixel values of the
    “Lightness” channel [1] of the image in HLS color space. Converting the image
    back to RGB gives the same image with enhanced or suppressed lighting.
    """
    # Convert image to HLS.
    img_HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    img_HLS = np.array(img_HLS, dtype=np.float64)
    # Generate a random value in [0.5, 1.5].
    random_brightness_coefficient = np.random.uniform() + 0.5
    # Scale pixel values up or down for channel 1 (Lightness)
    img_HLS[:, :, 1] = img_HLS[:, :, 1] * random_brightness_coefficient
    # Make sure the color value does not exceed 255.
    img_HLS[:, :, 1][img_HLS[:, :, 1] > 255] = 255
    # Convert image back to RGB.
    img_HLS = np.array(img_HLS, dtype=np.uint8)
    img_RGB = cv2.cvtColor(img_HLS, cv2.COLOR_HLS2RGB)
    return img_RGB


#
# Shadows
#

def add_shadow(img, nshadows=1):
    # Convert image to HLS.
    img_HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # Add shadows to an initially empty mask image.
    mask = np.zeros_like(img)
    # Generate a list of shadow polygons.
    shadow_list = generate_shadow_coordinates(img.shape, nshadows)
    # Add all shadow polygons to the empty mask; single 255 denotes only red channel.
    for shadow in shadow_list:
        cv2.fillPoly(mask, shadow, 255)
    # Use the mask to adjust pixels in the original image.
    # If red channel is hot, the image "Lightness" channel's brightness is lowered.
    img_HLS[:, :, 1][mask[:, :, 0] == 255] = img_HLS[:, :, 1][mask[:, :, 0] == 255] * 0.5
    # Convert image back to RGB.
    img_HLS = np.array(img_HLS, dtype=np.uint8)
    img_RGB = cv2.cvtColor(img_HLS, cv2.COLOR_HLS2RGB)
    return img_RGB


def generate_shadow_coordinates(imshape, nshadows=1):
    shadow_list = []
    for _ in range(nshadows):
        shadow = []
        # Dimensionality of the shadow polygon.
        for _ in range(np.random.randint(3, 15)):
            shadow.append((imshape[1] * np.random.uniform(), imshape[0] // 3 + imshape[0] * np.random.uniform()))
        # Add vertices for a single shadow polygon.
        shadow = np.array([shadow], dtype=np.int32)
        shadow_list.append(shadow)
    # List of shadow vertices.
    return shadow_list


#
# Snow
#

def add_snow(img, snow_brightness=2.5, snow_point=140):
    """
    Brighten the darkest parts of the image.
    Increase 'snow_point' for more snow.
    """
    # Convert image to HLS.
    img_HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    img_HLS = np.array(img_HLS, dtype=np.float64)
    # Scale pixel values up for channel 1 (Lightness)
    img_HLS[:, :, 1][img_HLS[:, :, 1] < snow_point] *= snow_brightness
    # Make sure the color value does not exceed 255.
    img_HLS[:, :, 1][img_HLS[:, :, 1] > 255] = 255
    # Convert image back to RGB.
    img_HLS = np.array(img_HLS, dtype=np.uint8)
    img_RGB = cv2.cvtColor(img_HLS, cv2.COLOR_HLS2RGB)
    return img_RGB


#
# Rain
#

def add_rain(img):
    # Generate rain drops as lines.
    slant_extreme = 10
    slant = np.random.randint(-slant_extreme, slant_extreme)
    drop_length = 10
    drop_width  = 2
    drop_color  = (200, 200, 200)  # a shade of gray
    rain_drops = generate_random_lines(img.shape, slant, drop_length)

    # Add rain drops to the image.
    for drop in rain_drops:
        cv2.line(img, (drop[0], drop[1]),
                      (drop[0] + slant, drop[1] + drop_length),
                 drop_color, drop_width)
    img = cv2.blur(img, (7, 7))  # Rainy views are blurry.

    # Darken the image a bit - rainy days are usually shady.
    brightness_coefficient = 0.7
    # Convert image to HLS.
    img_HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # Scale pixel values down for channel 1 (Lightness)
    img_HLS[:, :, 1] = img_HLS[:, :, 1] * brightness_coefficient
    # Convert image back to RGB.
    img_HLS = np.array(img_HLS, dtype=np.uint8)
    img_RGB = cv2.cvtColor(img_HLS, cv2.COLOR_HLS2RGB)
    return img_RGB


def generate_random_lines(imshape, slant, drop_length, ndrops=500):
    """ For heavy rain, try increasing 'ndrops'. """
    drops = []
    for _ in range(ndrops):
        x = np.random.randint(slant, imshape[1]) if slant < 0 else \
            np.random.randint(0, imshape[1] - slant)
        y = np.random.randint(0, imshape[0] - drop_length)
        drops.append((x, y))
    return drops


#
# Fog
#

def add_fog(img, hw=100):
    """
    Fog intensity is an important parameter to train a car for how much throttle it should give.
    For coding such a function, you can take random patches from all over the image, and increase
    the image’s lightness within those patches. With a simple blur, this gives a nice hazy effect.
    """
    # Convert image to HLS.
    img_HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    img_HLS[:, :, 1] = img_HLS[:, :, 1] * 0.8
    haze_list = generate_random_blur_coordinates(img.shape, hw)
    for haze_points in haze_list:
        # # Make sure the color value does not exceed 255.
        # img_HLS[:, :, 1][img_HLS[:, :, 1] > 255] = 255
        img_HLS = add_blur(img_HLS, haze_points[0], haze_points[1], hw)
    # Convert image back to RGB.
    img_HLS = np.array(img_HLS, dtype=np.uint8)
    img_RGB = cv2.cvtColor(img_HLS, cv2.COLOR_HLS2RGB)
    return img_RGB


def generate_random_blur_coordinates(imshape, hw):
    blur_points = []
    midx = imshape[1] // 2 - hw - 100
    midy = imshape[0] // 2 - hw - 100
    # radially generating coordinates
    index = 1
    while (midx > -100 or midy > -100):
        for _ in range(250 * index):
            x = np.random.randint(midx, imshape[1] - midx - hw)
            y = np.random.randint(midy, imshape[0] - midy - hw)
            blur_points.append((x, y))
        midx -= 250 * imshape[1] // sum(imshape)
        midy -= 250 * imshape[0] // sum(imshape)
        index += 1
    return blur_points


# def add_blur(img, x, y, hw):
#     # Increase 'L' channel by 1.
#     img[y:y + hw, x:x + hw, 1] = img[y:y + hw, x:x + hw, 1] + 1
#     # Make sure the adjusted value does not exceed 255.
#     img[:, :, 1][img[:, :, 1] > 255] = 255
#     img = np.array(img, dtype=np.uint8)
#     # Blur
#     img[y:y + hw, x:x + hw, 1] = cv2.blur(img[y:y + hw, x:x + hw, 1], (5, 5))
#     return img


def add_blur(img, x, y, hw):
    # Create a grid of wrapped indices since numpy arrays do not handle
    # slicing with negative values and wrap-around without some help.
    wrappedRowIndices = np.arange(y, y + hw) % img.shape[0]
    wrappedColIndices = np.arange(x, x + hw) % img.shape[1]
    index_grid = np.ix_(wrappedRowIndices, wrappedColIndices, [1])

    # Increase 'L' channel by 1.
    img[index_grid] = img[index_grid] + 1
    # Make sure the adjusted value does not exceed 255.
    img[:, :, 1][img[:, :, 1] > 255] = 255
    img = np.array(img, dtype=np.uint8)

    # Blur
    blur_patch = cv2.blur(img[index_grid], (5, 5)).reshape(hw, hw, 1)
    img[index_grid] = blur_patch

    return img
