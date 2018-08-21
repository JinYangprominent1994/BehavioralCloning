# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network.

First, I use a Lambda layer to normalize the input images, and then use a cropping layer to crop input images in order to maintain road in these images.

The architecture of my model includes five Convolution2D layers. Next, I use a Flatten layer to flatten the array.

Then I use four Dense layers. The output of my model is the angle that vehicles would take.


#### 2. Attempts to reduce overfitting in the model


The model was trained and validated on different data sets to ensure that the model was not overfitting. I split the set to training set and validation set,

and the ratio of splitting is 0.2.

I shuffle the training set and validation set to avoid overfitting.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.

I used a combination of center lane driving, recovering from the left and right sides of the road.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to using an end-to-end neural network.

My first step was to use a convolution neural network model. I thought this model might be appropriate because it could recognize objects correctly.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.

The ratio of splitting is 0.2, which is a common number used by other people. This implied that the model was overfitting.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track;

therefore, to improve the driving behavior in these cases, I used images captured by the center camera, the left camera and the right camera.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.


#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes.

First, I use a Lambda layer to normalize the input images, and then use a cropping layer to crop input images in order to maintain road in these images.

The architecture of my model includes five Convolution2D layers. Next, I use a Flatten layer to flatten the array.

Then I use four Dense layers(Dense(100),Dense(50),Dense(10),Dense(1)). The output of my model is the angle that vehicles would take.


#### 3. Creation of the Training Set & Training Process

I used the data provided by Udacity to train my model in order to acquire the best performance.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.

The ideal number of epochs was 5.

I tried different epoch values, like 1,2,3,5. The result is that if I set epoch = 5, I would acquire the best testing performance.  

I used an Adam optimizer so that manually training the learning rate wasn't necessary.
