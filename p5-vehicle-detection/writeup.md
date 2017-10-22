**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./writeup/car_not_car.jpg
[image2]: ./writeup/hog.jpg
[image3]: ./writeup/sliding_window.jpg
[image4]: ./writeup/heatmap.jpg
[video1]: ./test_videos_output/project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines #60 through #74 of the file called `vehicle_detection.py`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and for each combination I trained a linear SVM respectively. I used 20% percent of the dataset as the cross validation set and it turned out the performance of the various combinations were quite close: most of them had an over 98% test accuracy. So I decided to stop optimizing the model and focused on the pipeline. The final choice of HOG parameters is: `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using both HOG features and color features in lines #77 through #117 in `vehicle_detection.py`.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to restrict the sliding window search to the right part of the image and use 3 scales of window. The searching area and window scale are as follows:

| Search Area                                          | Window Scale |
| -----------------------------------------------------|:------------:|
| `xstart=800`, `xstop=1280`, `ystart=350`, `ystop=500`| `64`         |
| `xstart=800`, `xstop=1280`, `ystart=350`, `ystop=500`| `96`         |
| `xstart=800`, `xstop=1280`, `ystart=350`, `ystop=700`| `128`        |

The overlap between adjcent windows is 2 cells, which is 16 pixels.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 3 scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image3]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./test_videos_output/project_video.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

### Here's an example:

![alt text][image4]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main problem I faced is the accuracy of the classifier. Although the test accuracy was quite high and over 98%, the model didn't generalize well when being tested on the video stream. Speificly, it always made wrong prediction on not car samples.

So I decided to collect more not car training data. I extracted the frames that the model made wrong predictions and run a script named `extract.py` to extract the subimages. Then I added these images to the training set. Unfortunatly all the effort of increasing the training set didn't pay off. After discussing with my mentor, I decided to focus on the pipeline to filter the false positive detection.

The final implementation of the pipeline used a buffer to keep the detection of 5 continus frames. A heatmap is built using 5 frames's detection and a larger threshold is applied on the accumulated heatmap. This method filtered the false positive detection to some extend.