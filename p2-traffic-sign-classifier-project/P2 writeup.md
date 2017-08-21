# **Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image11]: ./writeup/distribution_training.png "Distribution Training"
[image12]: ./writeup/distribution_validation.png "Distribution Validation"
[image13]: ./writeup/grayscale.png
[image14]: ./writeup/jittered.png
[image15]: ./resized-test-images/1.jpeg
[image16]: ./resized-test-images/b.jpg
[image17]: ./resized-test-images/d.jpg
[image18]: ./resized-test-images/e.jpg
[image19]: ./resized-test-images/f.jpg
[image20]: ./resized-test-images/h.jpg

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/yuanjunyi/car/blob/master/p2-traffic-sign-classifier-project/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. They are two bar charts showing how the classes are distributed in the training set and the validation set.

![alt text][image11]

![alt text][image12]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because Pierre Sermanet and Yann LeCun claimed in their paper Traffic Sign Recognition with Multi-Scale Convolutional Networks that grayscale image have better results.

I also applied histogram equalization to the grayscale image to enhance the contrast.

Here is an example of a traffic sign image before and after histogram equalization.

![alt text][image13]

As a last step, I normalized the image data because I want to make sure each feature has same distribution so that a global learning rate is effective to each feature.

I decided to generate additional data because the original training set is too small according to Pierre Sermanet and Yann LeCun's paper.

To add more data to the the data set, I applied translation, scaling and rotation to original image, which is also learnt from Pierre Sermanet and Yann LeCun's paper.

Here is an example of an original image, its translation, scaling, rotation versions and an augmented image which combines the three transformations:

![alt text][image14]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 grayscale image   					|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x108 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x108 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 16x16x108 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x108 					|
| Fully connected		| outputs 100        							|
| Dropout				| 0.5 probability        						|
| RELU					|													|
| Fully connected		| outputs 43						|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used Adam Optimizer. Batch size is 128. Number of epochs is 30.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.5%
* validation set accuracy of 96.1%
* test set accuracy of  92.8%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I used LeNet-5.

* What were some problems with the initial architecture?
With LeNet-5 and the original training set, I failed to reach an accuracy of over 80%. So I was suspecting that the training set is too small and that the capacity of the model is not sufficient.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
After resolving the small dataset problem, I focused to improve the model. I just randomly increase its capacity by increasing the number of convolution feature layers and monitor how the training accuracy and evaluation accuracy evolve. Once it seems overfitting, I introduced dropout and L2 regularization.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image15] ![alt text][image16] ![alt text][image17]
![alt text][image18] ![alt text][image19] ![alt text][image20]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Road work      		| Slippery road   								|
| Speed limit (30km/h)  | Speed limit (30km/h) 							|
| No passing			| Speed limit (120km/h)							|
| Priority road     	| Priority road              					|
| Slippery road	      	| Slippery road					 				|
| Slippery road			| Slippery Road      							|

The model was able to correctly guess 4 of the 6 traffic signs, which gives an accuracy of 66.7%. This compares favorably to the accuracy on the test set of 92.8%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is uncertain and it makes a wrong prediction.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .39         			| Slippery road   								|
| .15     				| Road work										|
| .11					| Bicycles crossing								|
| .08	      			| Beware of ice/snow			 				|
| .08				    | Bumpy road     		  						|

For the second image, the model is very certain and it makes a correct prediction.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .81         			| Speed limit (30km/h)  						|
| .05     				| Speed limit (80km/h)							|
| .05					| Speed limit (80km/h)							|
| .02	      			| Speed limit (80km/h)					 		|
| .01				    | Speed limit (70km/h)     						|

For the third image, the model is uncertain and it makes a wrong prediction.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .37         			| Speed limit (120km/h)   						|
| .11     				| Speed limit (70km/h)							|
| .07					| No passing									|
| .07	      			| Speed limit (50km/h)							|
| .06				    | Speed limit (20km/h)     						|

For the fourth image, the model is extremely certain and it makes a correct prediction.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .99         			| Priority road   								|
| .00     				| No vehicles									|
| .00					| Yield											|
| .00	      			| End of all speed and passing limits			|
| .00				    | Keep right     								|

For the fifth image, the model very certain and it makes a correct prediction.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .78         			| Slippery road   								|
| .07     				| Bicycles crossing								|
| .04					| Wild animals crossing							|
| .01	      			| Dangerous curve to the left					|
| .01				    | Beware of ice/snow     						|

For the sixth image, the model is uncertain and it makes a correct prediction.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .32         			| Slippery road   								|
| .07     				| Right-of-way at the next intersection			|
| .07					| Double curve									|
| .07	      			| Dangerous curve to the right	 				|
| .05				    | Beware of ice/snow     						|
