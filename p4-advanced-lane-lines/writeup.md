**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibrated.jpg "Calibrated"
[image2]: ./output_images/undistorted.jpg "Undistorted"
[image3]: ./output_images/combine.jpg "Threshold Combination"
[image4]: ./output_images/warped_straight_line.jpg "Warp Example"
[image5]: ./output_images/lines.jpg "Fit Visual"
[image6]: ./output_images/projected.jpg "Output"
[image7]: ./output_images/pipeline.png "Pipeline"
[video1]: ./test_videos_output/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is in lines #74 through #99 of the file called `advanced_lane_lines.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `points_3D` is just a replicated array of coordinates, and `points_3D_in_real_world` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `points_2D_in_image_plane` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `points_3D_in_real_world` and `points_2D_in_image_plane` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines #318 through #382 in `advanced_lane_lines.py`).  Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_transform()`, which appears in lines #154 through #177 in `advanced_lane_lines.py`. The `perspective_transform()` function takes as inputs an image (`image`), as well as the transform matrix (`M`). I chose to hardcode the source and destination points in the following manner:

```python
# 4 source coordinates
src = np.float32(
    [[591, 450],
     [687, 450],
     [1115, 720],
     [190, 720]])
# 4 desired coordinates
dst = np.float32(
    [[320, 0],
     [960, 0],
     [960, 720],
     [320, 720]])
```

I verified that my perspective transform was working as expected by applying it on a straight line image and verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I located the lane lines in the warped image by using the sliding window method. And fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines #49 through #56 in my code in `advanced_lane_lines.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines #439 through #467 in my code in `advanced_lane_lines.py` in the function `project_lines()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./test_videos_output/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main problem I faced in the project is to create a good quality thresholded binary image. As long as this step is successful, my pipeline can work.

When I tested my pipeline on project_video.mp4, I saved the problematic frames and then fine tuned the pipeline. Here is an example:

![alt text][image7]

In this example, there are black shadows on the road and the HLS threshold can not filter them. For this reason, I added the grayscale threshold. I also fine tuned the threshlds of gradient magnitude and gradient direction using the same approach and removed the small noise.


