# **Behavioral Cloning** 

## Scott Henderson
## Self Driving Car Nanodegree Term 1 Project 3

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image]: ./images/image.png "Sample Image"
[image_bright]: ./images/image_0_bright.png "Brightness-adjusted image"
[image_shadow]: ./images/image_1_shadow.png "Image with random shadows"
[image_snow]: ./images/image_2_snow.png "Image with snow"
[image_rain]: ./images/image_3_rain.png "Image with rain"
[image_fog]: ./images/image_4_fog.png "Image with fog"
[image_torrential_rain]: ./images/image_5_torrential_rain.png "Image with torrential rain"
[loss]: ./Loss.png "Loss vs Epochs"

---
### Model Architecture and Training Strategy

#### Model Architecture and Data Preparation

Source code for the project is almost entirely in model.py with some additional functions in image_augmentation.py (more on that below).

I began with the LeNet neural network architecture just to test the pipeline and make sure everything was working. I then quickly switched to the NVIDIA model. The NVIDIA model consists of five convolutional layers and four fully connected layers. All of the convolutional layers use a ReLU activation function as a nonlinearity. (model.py lines 110-121)

This version of the model worked but there was still some overfitting and the car strayed from the road just a bit. So I modified the NVIDIA model to add dropout layers after each convolution layer and an L2 regularization after each fully connected layer. Both at the suggestion of Eric Lavigne via the "s-t1-p-behavioral-clo" Slack channel. (model.py lines 124-149)

Data preparation consisted of just two steps:
1. Cropping the incoming images to remove the top portion (sky, trees, power lines and other data not relevant to following the road) and the bottom portion (car hood) of each image. This change improved training speed and also performance as the input imges contained much more data relevant to the problem of staying on the road. (model.py lines 82-95)
1. Normalizing the image data using a Keras lambda layer. (model.py line 206)

Another change I made at the suggestion of Eric Lavigne ("s-t1-p-behavioral-clo" Slack channel) was to switch the loss function from mean squared error to mean absolute error. The reason for the change is that mse imposes a significant penalty for incorrect answers and leads the training process to be on the conservative side. Using mae instead allows the model to "take chances" and ultimately improves training performance.

At this point the model performed fairly well but still suffered from overfitting - even after adding dropout and L2 regularization layers.


#### Attempts to reduce overfitting in the model

As described above I modified the NVIDIA architecture to add dropout and L2 regularization layers. But there was still overfitting as indicated by the gap between training loss and validation loss.

Early on I made the decision to use the provided test data rather than try to produce my own test data using the simulator. My feeling was that I could spend a lot of time learning to use the simulator and generating test data myself. I chose to spend that time on model architecture and other changes as well as image augmentation.

For this reason I used all three camera images - center, left, right - rather than just the center image. But I also added augmented images. At first just a horizontally flipped version of the incoming image (model.py line 182-183) - this to reduce the tendency of the model to steer left since most of the training data consists of left turns. I also added other augmented images.

A while back I came across an article desribing how to augment images by applying various modifications - mainly weather patterns:

Image Augmentation: Make it rain, make it snow. How to modify photos to train self-driving cars
by Ujjwal Saxena

https://medium.freecodecamp.org/image-augmentation-make-it-rain-make-it-snow-how-to-modify-a-photo-with-machine-learning-163c0cb3843f

I used the code more or less as-is from this article to add additional augmented images. (model.py lines 184-194). The image augmentation code can be found in image_augmentation.py

Original sample image:

![alt text][image]

Augmented images:

Brightness adjustment<br>
![alt text][image_bright]
<br>Random shadows<br>
![alt text][image_shadow]
<br>Snow<br>
![alt text][image_snow]
<br>Rain<br>
![alt text][image_rain]
<br>Fog<br>
![alt text][image_fog]
<br>Torrential rain (rain+fog)<br>
![alt text][image_torrential_rain]

Adding these images significantly increased training time. Especially since I do not have access to AWS or a particularly fast computer. Also adding all of these augmentations did not particularly help with the gap between training loss and validation loss. Also, though the overall loss was fairly low, the model would cause the car to run off the road. Possibly because some of these images are just too occluded.  In the end I chose to include only the brightness-adjusted image and the image with random shadows added. These seemed to be the safest and the end result turned out well. Though there is still a gap between training loss and validation loss indicating overfitting.


#### Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. (model.py line 217)  I did try various values for the dropout and L2 regularization rate as well as the number of epochs to use for training. (model.py lines 48-52)


### Abandoned Efforts

I made a few modifications to the drive.py file. The only one that was strictly necessary was to crop the incoming image using the same parameters used in data preprocessing for model training. Otherwise the incoming image would not be useable with the model. (drive.py line 78).

I also attempted to add a low pass filter to smooth out steering changes (another suggestion by Eric Lavigne via the "s-t1-p-behavioral-clo" Slack channel). I was not able to make this work in the sense that my attempted implmentation tended to cause the car to immediately veer off road. Also the final result was not nearly as "shaky" as initial results so I decided this was an issue best left for another time.

I also attempted to implement a generator function to use with training the model (intended to allow training on very large datasets) (predict_steering.py lines 48-98, 156-161). This attempt worked in the sense that the model was trained but the performance suffered considerably and the car would not stay on the road. It's possible that this could have been avoided by trying different values for the batch size (I tried only the default of 32) but I was able to train on the existing dataset without too much trouble so decided against pursuing this approach.


### Results

As indicated above the gap between training loss and validation loss still suggests overfitting.

![alt text][Loss]

There are lots of other options for attempting to reduce this gap:
1. Gather more training data.
1. Further adjustments to the droput and L2 regularization rates.
1. Increase the number of epochs used for training - though this may not have helped since the validation loss begins to increase in later epochs as is.
1. Add additional augmented images - after perhaps adjusting parameters used to produce various weather effects.
1. Try different color spaces for incoming images and different image augmentation techniques.

However the model was able to accurately steer the car around the track. See run1.mp4 for the final run. Given that this was the main goal of this exercise I decided to call the result good and move on.
