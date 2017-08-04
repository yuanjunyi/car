# **Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

[//]: # (Image References)

[image1]: ./test_images_output/shadow2_gray.jpg "Grayscale"
[image2]: ./test_images_output/shadow2_grayscaleThreshold.jpg "Grayscale Threshold"
[image3]: ./test_images_output/shadow2_canny.jpg "Canny Edges Detection"
[image4]: ./test_images_output/shadow2_region.jpg "Region of Interest"
[image5]: ./test_images_output/shadow2_linesOriginal.jpg "Hough Line Transform"
[image6]: ./test_images_output/shadow2_lines.jpg "Process Lines"
[image7]: ./test_images_output/shadow2.jpg "Annotated"
[image8]: ./test_images/shadow2.jpg "Shadow2"

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 7 steps.
  1. Convert the initial image to grayscale.
  ![alt text][image1]
  2. Remove the shadow by applying a grayscale threshold.
  ![alt text][image2]
  3. Detect the edges.
  ![alt text][image3]
  4. Select the region of interest.
  ![alt text][image4]
  5. Find the lines in the region of interest.
  ![alt text][image5]
  6. Process the lines found in step 5. Draw the two lane lines on a blank image.
  ![alt text][image6]
  7. Annotate the lane lines on the initial image by adding the line image to it.
  ![alt text][image7]
  
In order to draw a single line on the left and right lanes, I modified the draw_lines() function by the following steps:
  1. Discard the lines found by Hough line transform if they are approximately vertical or horizontal.
  2. Compute the slope of each line: slope = (y2-y1)/(x2-x1). Discard the lines whose absolute value of slope is under 0.5.
  3. Compute the intercept of each line: intercept = y-slope*x.
  4. Use the sign of slope to group the lines into left lane and right lane.
  5. Compute the mean slope and mean intercept of the two groups respectively.
  6. Find the minimum y-axis coordinate of all the lines and name it y1. Find the height of the initial image and name it y2. We'll use the same y1 and y2 to draw the left and right lane lines. This allows us to have the same y-axis level between the left and right lane lines.
  7. Use the mean slope and mean intercept computed in step 5 to compute x1 and x2 of the two groups. Name them x1_left, x2_left, x1_right, x2_right.
  8. Draw left lane line (x1_left, y1), (x2_left, y2). Draw right lane line (x1_right, y1), (x2_right, y2).

### 2. Identify potential shortcomings with your current pipeline

The step 2 of my pipeline can remove the effect of shadow because the color of shadows and the color of road are close. By applying a grayscale threshold, we actually merge the shadows and the road. As lane lines are white and yellow, we can still detect them from the dark mixture. However, this approach will not work for other marks on the road if the color of the marks is close to the color of the lane lines. Below is an example in the challenge.

There are black shadows and a light colored triangle mark on the road.
![alt text][image8]

A grayscale threshold can effectively filter the shadows but it doesn't manage to filter the light colored mark.
![alt text][image2]

### 3. Suggest possible improvements to your pipeline

A possible improvement would be to filter the shadows by using the lightness information in the HSL representation of color.