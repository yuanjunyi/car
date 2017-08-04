import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def grayscale_threshold(gray, threshold):
    mask = gray[:, :] < threshold
    gray[mask] = threshold
    mpimg.imsave('test_images_output/shadow2_grayscaleThreshold.jpg', gray, cmap='gray')
    return gray

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mpimg.imsave('test_images_output/shadow2_gray.jpg', gray, cmap='gray')
    return gray
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    lines = cv2.Canny(img, low_threshold, high_threshold)
    mpimg.imsave('test_images_output/shadow2_canny.jpg', lines, cmap='gray')
    return lines

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    mpimg.imsave('test_images_output/shadow2_region.jpg', masked_image, cmap='gray')
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=20):
    """
    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    left_line_slope = []
    left_line_intercept = []
    left_line_y = []

    right_line_slope = []
    right_line_intercept = []
    right_line_y = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            if not math.isclose(x1, x2, abs_tol=5) and not math.isclose(y1, y2, abs_tol=5):
                slope = (y2-y1) / (x2-x1)
                intercept1 = y1 - slope*x1
                intercept2 = y2 - slope*x2
                if abs(slope) > 0.5:
                    if slope < 0:
                        left_line_slope.append(slope)
                        left_line_intercept.append(intercept1)
                        left_line_intercept.append(intercept2)
                        left_line_y.append(y1)
                        left_line_y.append(y2)
                    else:
                        right_line_slope.append(slope)
                        right_line_intercept.append(intercept1)
                        right_line_intercept.append(intercept2)
                        right_line_y.append(y1)
                        right_line_y.append(y2)

    h, w = img.shape[:2]
    y_bottom = h
    if left_line_slope and right_line_slope:
        y_top = min(min(left_line_y), min(right_line_y))

        mean_left_line_slope = np.mean(left_line_slope)
        mean_left_line_intercept = np.mean(left_line_intercept)
        mean_right_line_slope = np.mean(right_line_slope)
        mean_right_line_intercept = np.mean(right_line_intercept)

        left_line_x_bottom = int((y_bottom-mean_left_line_intercept) / mean_left_line_slope)
        left_line_x_top = int((y_top-mean_left_line_intercept) / mean_left_line_slope)
        cv2.line(img, (left_line_x_bottom, y_bottom), (left_line_x_top, y_top), color, thickness)

        right_line_x_bottom = int((y_bottom-mean_right_line_intercept) / mean_right_line_slope)
        right_line_x_top = int((y_top-mean_right_line_intercept) / mean_right_line_slope)
        cv2.line(img, (right_line_x_bottom, y_bottom), (right_line_x_top, y_top), color, thickness)
    elif left_line_slope:
        y_top = min(left_line_y)
        mean_left_line_slope = np.mean(left_line_slope)
        mean_left_line_intercept = np.mean(left_line_intercept)
        left_line_x_bottom = int((y_bottom-mean_left_line_intercept) / mean_left_line_slope)
        left_line_x_top = int((y_top-mean_left_line_intercept) / mean_left_line_slope)
        cv2.line(img, (left_line_x_bottom, y_bottom), (left_line_x_top, y_top), color, thickness)
    elif right_line_slope:
        y_top = min(right_line_y)
        mean_right_line_slope = np.mean(right_line_slope)
        mean_right_line_intercept = np.mean(right_line_intercept)
        right_line_x_bottom = int((y_bottom-mean_right_line_intercept) / mean_right_line_slope)
        right_line_x_top = int((y_top-mean_right_line_intercept) / mean_right_line_slope)
        cv2.line(img, (right_line_x_bottom, y_bottom), (right_line_x_top, y_top), color, thickness)

def draw_lines_test(img, lines, color=[255, 0, 0], thickness=1):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1,y1), (x2,y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    #draw_lines(line_img, lines)
    draw_lines_test(line_img, lines)
    mpimg.imsave('test_images_output/shadow2_linesOriginal.jpg', line_img)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def process_image(image):
    h, w = image.shape[:2]
    gray = grayscale(image)
    gray = grayscale_threshold(gray, 100)

    low_threshold = 50
    high_threshold = 150
    edges = canny(gray, low_threshold, high_threshold)

    vertices = np.array([[[w*0.45, h*0.6], [w*0.6, h*0.6], [w*0.95, h], [w*0.15, h]]], dtype=np.int32)
    interesting_region = region_of_interest(edges, vertices)

    rho = 1
    theta = np.pi/180
    threshold = 50
    min_line_length = 5
    max_line_gap = 30
    line_image = hough_lines(interesting_region, rho, theta, threshold, min_line_length, max_line_gap)

    return weighted_img(line_image, image)

def show_image(image, filename):
    plt.figure()
    plt.title(filename)
    plt.imshow(image)

def test_one_image(filename):
    image = mpimg.imread('test_images/' + filename)
    show_image(process_image(image), filename)
    plt.show()

def test_with_image():
    filenames = os.listdir('test_images/')
    for filename in filenames:
        image = mpimg.imread('test_images/' + filename)
        processed_image = process_image(image)
        show_image(processed_image, filename)
        mpimg.imsave('test_images_output/' + filename, processed_image)
    plt.show()

def test_with_video():
    filenames = os.listdir('test_videos')
    for filename in filenames:
        clip = VideoFileClip('test_videos/' + filename)
        processed_clip = clip.fl_image(process_image)
        processed_clip.write_videofile('test_videos_output/' + filename, audio=False)

if __name__ == '__main__':
    test_one_image("shadow2.jpg")