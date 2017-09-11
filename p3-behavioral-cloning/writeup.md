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

[image1]: ./writeup/mse_on_new_data.png "mse"
[image2]: ./writeup/center_2017_09_11_14_22_47_721.jpg "center lane"
[image3]: ./writeup/center_2017_09_11_14_24_38_649.jpg "original"
[image4]: ./writeup/center_2017_09_11_14_24_38_649_flipped.jpg "flipped"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results
* track1.mp4
* track2.mp4

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 16 and 128.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. In fact, the mean squared error on training and validation set were always close after 10 epochs and overfitting was not a big problem.

That said, I still added a dropout layer to improve generalization in case the model had to handle new images when running autonomously.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. Images from center, left and right cameras were used. After a few experiements, I chose 0.3 as the steering angle correction for left and right cameras.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to make sure the model has low mean squared error on both training and validation set. This implies the model has the right capacity.

My first step was to use a convolution neural network model similar to the one described in NVIDIA's paper End to End Learning for Self-Driving Cars. I thought this model might be appropriate because we are trying to solve a similar problem.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. The mse of my first model on the training and validation set were both relatively low, so I ran the simulator directly. The vehicle fell off the track immediately.

Since the model was already proven by NVIDIA and the mse was low, so I decided to concentrate on collecting more good quality data. It was a long process to train myself to drive smoothly on both tracks. Finally I was able to collect 3 full laps for each track and trained the model with them. The mse on training and validataion set were still close and relatively low, so I ran the simulator again. The vehicle was able to drive autonomously most of the time but fell off the track at a few spots.

I tried to collect more data at those spots and trained the model again and again. The trained model didn't improve significantly with the additional training data.

Then I realized that it was very difficult for the model to unlearn the bad behavior. I discarded the old data and collected 3 new full laps for each track carefully. Also, I added one more convolution layer in order to make sure the model has sufficient capacity to absorb my behavior.

Here is the mse of the final model on the final dataset:

![alt text][image1]

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

My final model consisted of the following layers:

| Layer           | Description                          | Size      |
|:---------------:|:------------------------------------:|:---------:|
| Input           | RGB image                            | 160x320x3 |
| Resize          | Resize image to speed up training    | 80x160x3  |
| Cropping        | Crop uninteresting part of the image | 50x160x3  |
| Normalization   | x = x / 255 - 0.5                    |           |
| Convolution 3x3 | 1x1 stride, same padding             | 50x160x16 |
| RELU            |                                      |           |
| Max pooling 2x2 | 2x2 stride, valid padding            | 25x80x16  |
| Convolution 3x3 | 1x1 stride, same padding             | 25x80x32  |
| RELU            |                                      |           |
| Max pooling 2x2 | 2x2 stride, valid padding            | 12x40x32  |
| Convolution 3x3 | 1x1 stride, same padding             | 12x40x64  |
| RELU            |                                      |           |
| Max pooling 2x2 | 2x2 stride, valid padding            | 6x20x64   |
| Convolution 3x3 | 1x1 stride, same padding             | 6x20x128  |
| RELU            |                                      |           |
| Max pooling 2x2 | 2x2 stride, valid padding            | 3x10x128  |
| Convolution 3x3 | 1x1 stride, same padding             | 3x10x128  |
| RELU            |                                      |           |
| Max pooling 2x2 | 2x2 stride, valid padding            | 1x5x128   |
| Flatten         |                                      | 640       |
| Drouput         | 0.5                                  |           |
| Fully connected |                                      | 100       |
| Fully connected |                                      | 50        |
| Fully connected |                                      | 1         |

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 3 laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

To augment the dataset, I also flipped images and angles thinking that this would resolved the left turn bias. For example, here is an image that has then been flipped:

![alt text][image3]
![alt text][image4]

Left and right images were also collected. They were fed to the model as they were coming from the center camera. This taught the model how to steer if the car drifts off to the left or the right.

Then I repeated this process on track two in order to get more data points.

After the collection process, I had 145446 number of data points. I then preprocessed this data by the following steps:

1. Resize the image by a factor of 2. This is to speed up the training process.
2. The top of the image is trees and mountains that don't contribute to the driving, so I cropped them. The same to the bottom of the image.
3. Normalize each pixel of the image.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by the convergence of mse on training and validation set.

I used an adam optimizer so that manually training the learning rate wasn't necessary.
