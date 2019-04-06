# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[training_history_one_lap]: ./writeup_images/training_history_one_lap.png "Training History One Lap"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on [the NVIDIA model](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) used to control self-driving cars.

My model consists of the following:
* 3 convolutional layers with 5x5 filter sizes and depths between 24 and 48 (model.py lines 83-85) 
* 2 convolutional layers with 3x3 filter sizes and depths of 64 (model.py lines 86-87) 
* 3 fully connected layers with 100, 50, and 10 neurons

The model includes RELU activation in all layers except the final layer (code lines 83-92),
and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer right after the final convolutional layer in order to reduce overfitting (model.py line 88). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 29). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 100).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start from a simple model
and repeat test runs to see what action should be taken next.
Actions to improve autonomous driving include:
* augmenting data
* collecting more data
* changing the model architecture
* changing training hyperparameters

My first step was to use a convolution neural network model similar to the LeNet architecture.
I thought this model might be appropriate because it is a well known architecture, and
it is known to work well with different input sizes.

In early test runs, the car failed to turn tight corners.
I augmented the training data by using multiple cameras and flipping images,
then switched to [the NVIDIA model](https://devblogs.nvidia.com/deep-learning-self-driving-cars/).

The NVIDIA model improved the driving behavior significantly.
However, the model seemed to be overfitting and struggled to keep driving in the center on tight corners,
Here is the training history of the model at this point.

![alt text][training_history_one_lap]

To combat overfitting, I added a dropout layer and collected more data.
For details, see sections 2 and 3.

#### 2. Final Model Architecture

The final model architecture (model.py lines 80-93) consisted of a convolution neural network with the following layers.



Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
