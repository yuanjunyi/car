import sys
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip


P = {
    'm_per_pixel_y': 30 / 720,
    'm_per_pixel_x': 3.7 / 640
}


class Line():
    def __init__(self):
        self.current_fit = None
        self.previous_fit = None
        self.recent_fits = []
        self.recent_fits_m = []
        self.number_of_averaged_frames = 10

    def prepare_for_next_frame(self):
        self.previous_fit = self.current_fit
        self.current_fit = None

    def set_current_fit(self, fit):
        self.current_fit = fit
        self.recent_fits.append(self.current_fit)
        if len(self.recent_fits) > self.number_of_averaged_frames:
            self.recent_fits.pop(0)

    def add_current_fit_m(self, fit_m):
        self.recent_fits_m.append(fit_m)
        if len(self.recent_fits_m) > self.number_of_averaged_frames:
            self.recent_fits_m.pop(0)

    def get_previous_fit(self):
        return self.previous_fit

    def get_smooth_fit(self):
        if self.recent_fits:
            return np.mean(self.recent_fits, axis=0)
        else:
            print('recent_fits is empty')
            return None

    def get_curvature(self, y_pixel):
        if self.recent_fits_m:
            fit_m = np.mean(self.recent_fits_m, axis=0)
            y_m = y_pixel * P['m_per_pixel_y']
            return ((1 + (2*fit_m[0]*y_m + fit_m[1])**2)**1.5) / np.absolute(2*fit_m[0])
        else:
            print('recent_fits_m is empty')
            return None


left_line = Line()
right_line = Line()


def fit_polynomial(y, x, scaley=1, scalex=1):
    # Fit for f(y) = Ay**2 + By + C rather than f(x) because lines are near vertical
    return np.polyfit(y*scaley, x*scalex, 2)


def load_camera_calibration():
    camera_matrix = np.load('camera_matrix.npy')
    distortion_coefficients = np.load('distortion_coefficients.npy')
    return camera_matrix, distortion_coefficients


def calibrate_camera():
    nx = 9
    ny = 6
    points_3D_in_real_world = []
    points_2D_in_image_plane = []
    # Define points_3D as np.float32 rather than np.float because calibrateCamera() requests Point3f
    points_3D = np.zeros((nx*ny, 3), np.float32)
    points_3D[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    filenames = glob.glob('camera_cal/calibration*.jpg')
    for filename in filenames:
        image = cv2.imread(filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret == True:
            image_shape = image.shape[0:2]
            points_2D_in_image_plane.append(corners)
            points_3D_in_real_world.append(points_3D)
    ret, camera_matrix, distortion_coefficients, rotation_vecs, translation_vecs = \
        cv2.calibrateCamera(points_3D_in_real_world,
                            points_2D_in_image_plane,
                            image_shape,
                            None,
                            None)
    np.save('camera_matrix.npy', camera_matrix)
    np.save('distortion_coefficients.npy', distortion_coefficients)
    return camera_matrix, distortion_coefficients


def undistort_image(image, camera_matrix, distortion_coefficients):
    return cv2.undistort(image, camera_matrix, distortion_coefficients, None, camera_matrix)


def hls_threshold(image, threshold=(0, 255)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary = np.zeros_like(s_channel, np.float)
    binary[(threshold[0] <= s_channel) & (s_channel <= threshold[1])] = 1
    return binary


def grayscale_threshold(image, threshold=(0, 255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    binary = np.zeros_like(gray, np.float)
    binary[(threshold[0] <= gray) & (gray <= threshold[1])] = 1
    return binary


def abs_sobel_threshold(gray, orient='x', ksize=3, threshold=(0, 255)):
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    else:
        raise ValueError('orient need to be x or y')
    abs_sobel = np.absolute(sobel)
    scaled = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary = np.zeros_like(scaled, np.float)
    binary[(threshold[0] <= scaled) & (scaled <= threshold[1])] = 1
    return binary


def gradient_magnitude_threshold(gray, ksize=3, threshold=(0, 255)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    scaled = np.uint8(255 * magnitude / np.max(magnitude))
    binary = np.zeros_like(scaled, np.float)
    binary[(threshold[0] <= scaled) & (scaled <= threshold[1])] = 1
    return binary


def gradient_direction_threshold(gray, ksize=3, threshold=(0, np.pi/2)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    direction = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary = np.zeros_like(direction, np.float)
    binary[(threshold[0] <= direction) & (direction <= threshold[1])] = 1
    return binary


def perspective_transform(image, M):
    image_size = (image.shape[1], image.shape[0])
    warped = cv2.warpPerspective(image, M, image_size, flags=cv2.INTER_LINEAR)
    return warped


def perspective_transform_matrix():
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

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv


def find_lines_using_previous_frame(binary_warped, previous_left_fit, previous_right_fit):
    margin = 100
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    previous_left_x = previous_left_fit[0]*nonzeroy**2 + \
                      previous_left_fit[1]*nonzeroy + \
                      previous_left_fit[2]
    left_inds = ((nonzerox > previous_left_x - margin) &
                 (nonzerox < previous_left_x + margin))

    previous_right_x = previous_right_fit[0]*nonzeroy**2 + \
                       previous_right_fit[1]*nonzeroy + \
                       previous_right_fit[2]
    right_inds = ((nonzerox > previous_right_x - margin) &
                  (nonzerox < previous_right_x + margin))
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_inds]
    lefty = nonzeroy[left_inds] 
    rightx = nonzerox[right_inds]
    righty = nonzeroy[right_inds]

    # Fit a second order polynomial in image space: pixel
    left_fit = fit_polynomial(lefty, leftx)
    right_fit = fit_polynomial(righty, rightx)

    # Fit a second order polynomial in real world: meter
    left_fit_m = fit_polynomial(lefty, leftx, P['m_per_pixel_y'], P['m_per_pixel_x'])
    right_fit_m = fit_polynomial(righty, rightx, P['m_per_pixel_y'], P['m_per_pixel_x'])

    return left_fit, right_fit, left_fit_m, right_fit_m


def find_lines_from_scratch(binary_warped, display=False):
    # Take a histogram of the bottom half of the image
    h, w = binary_warped.shape[:2]
    histogram = np.sum(binary_warped[int(h/2):,:], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = int(w/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = int(h/nwindows)
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 500

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_inds = []
    right_inds = []

    if display:
        # Create an output image to draw on and visualize the result       
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = h - (window+1) * window_height
        win_y_high = h - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        if display:
            cv2.rectangle(     
            out_img, (win_xleft_low,win_y_low), (win_xleft_high,win_y_high), (0,1,0), 2)      
            cv2.rectangle(        
            out_img, (win_xright_low,win_y_low), (win_xright_high,win_y_high), (0,1,0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = \
            ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
             (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]

        good_right_inds = \
            ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
             (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_inds.append(good_left_inds)
        right_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_inds = np.concatenate(left_inds)
    right_inds = np.concatenate(right_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_inds]
    lefty = nonzeroy[left_inds]
    rightx = nonzerox[right_inds]
    righty = nonzeroy[right_inds]

    # Fit a second order polynomial in image space
    left_fit = fit_polynomial(lefty, leftx)
    right_fit = fit_polynomial(righty, rightx)

    # Fit a second order polynomial in world space
    left_fit_m = fit_polynomial(lefty, leftx, P['m_per_pixel_y'], P['m_per_pixel_x'])
    right_fit_m = fit_polynomial(righty, rightx, P['m_per_pixel_y'], P['m_per_pixel_x'])

    if display:
        out_img[nonzeroy[left_inds], nonzerox[left_inds]] = (1,0,0)     
        out_img[nonzeroy[right_inds], nonzerox[right_inds]] = (0,0,1)
        ploty = np.linspace(0, h-1, h)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
        cv2.polylines(out_img, np.int_([pts_left]), False, (1,1,0), 2)
        cv2.polylines(out_img, np.int_([pts_right]), False, (1,1,0), 2)
        mpimg.imsave('output_images/lines.jpg', out_img)

    return left_fit, right_fit, left_fit_m, right_fit_m


def pipeline(image, camera_matrix, distortion_coefficients, display=False):
    # Undistort image
    undistorted = undistort_image(image, camera_matrix, distortion_coefficients)

    # HLS threshold
    hls_binary = hls_threshold(undistorted, threshold=(170, 255))

    # Grayscale threshold
    grayscale_binary = grayscale_threshold(undistorted, threshold=(50, 255))

    # Gradient threshold
    gray = cv2.cvtColor(undistorted, cv2.COLOR_RGB2GRAY)
    gradientx = abs_sobel_threshold(gray, orient='x', ksize=3, threshold=(30, 100))
    gradienty = abs_sobel_threshold(gray, orient='y', ksize=3, threshold=(30, 100))
    gradient_magnitude = gradient_magnitude_threshold(gray, ksize=9, threshold=(80, 100))
    gradient_direction = gradient_direction_threshold(gray, ksize=15, threshold=(0.7, 1.3))

    combine = np.zeros_like(gray, np.float)
    combine[((hls_binary == 1)| \
            ((gradientx == 1) & (gradienty == 1)) | \
            ((gradient_magnitude == 1) & (gradient_direction == 1))) & \
            (grayscale_binary == 1)] = 1 
    
    # Perspective transform
    M, Minv = perspective_transform_matrix()
    binary_warped = perspective_transform(combine, M)

    if display:
        mpimg.imsave('output_images/undistorted.jpg', undistorted)
        mpimg.imsave('output_images/combine.jpg', combine, cmap='gray')
        mpimg.imsave('output_images/binary_warped.jpg', binary_warped, cmap='gray')

        gradientxy = np.zeros_like(gray, np.float)
        gradientxy[(gradientx == 1) & (gradienty == 1)] = 1

        gradientmd = np.zeros_like(gray, np.float)
        gradientmd[(gradient_magnitude == 1) & (gradient_direction == 1)] = 1

        f, ax = plt.subplots(3, 4, figsize=(16,8))
        ax[0, 0].set_title('image')
        ax[0, 0].imshow(image)
        ax[0, 1].set_title('undistorted')
        ax[0, 1].imshow(undistorted)
        ax[0, 2].set_title('hls_threshold')
        ax[0, 2].imshow(hls_binary, cmap='gray')
        ax[0, 3].set_title('grayscale_threshold')
        ax[0, 3].imshow(grayscale_binary, cmap='gray')
        ax[1, 0].set_title('gradientx')
        ax[1, 0].imshow(gradientx, cmap='gray')
        ax[1, 1].set_title('gradienty')
        ax[1, 1].imshow(gradienty, cmap='gray')
        ax[1, 2].set_title('gradient_magnitude')
        ax[1, 2].imshow(gradient_magnitude, cmap='gray')
        ax[1, 3].set_title('gradient_direction')
        ax[1, 3].imshow(gradient_direction, cmap='gray')
        ax[2, 0].set_title('gradx & grady')
        ax[2, 0].imshow(gradientxy, cmap='gray')
        ax[2, 1].set_title('gradmag & graddir')
        ax[2, 1].imshow(gradientmd, cmap='gray')
        ax[2, 2].set_title('combine')
        ax[2, 2].imshow(combine, cmap='gray')
        ax[2, 3].set_title('binary_warped')
        ax[2, 3].imshow(binary_warped, cmap='gray')

    return binary_warped, undistorted, Minv


def sanity_check(binary_warped, left_fit, right_fit):
    h, w = binary_warped.shape[:2]
    distance_threshold_min = 640 * 0.5
    distance_threshold_max = 640 * 1.5

    y_low, y_mid, y_high = int(h*0.1), int(h*0.5), int(h*0.9)

    x_left_low = left_fit[0]*y_low**2 + left_fit[1]*y_low + left_fit[2]
    x_left_mid = left_fit[0]*y_mid**2 + left_fit[1]*y_mid + left_fit[2]
    x_left_high = left_fit[0]*y_high**2 + left_fit[1]*y_high + left_fit[2]

    x_right_low = right_fit[0]*y_low**2 + right_fit[1]*y_low + right_fit[2]
    x_right_mid = right_fit[0]*y_mid**2 + right_fit[1]*y_mid + right_fit[2]
    x_right_high = right_fit[0]*y_high**2 + right_fit[1]*y_high + right_fit[2]

    distance_low = x_right_low - x_left_low
    distance_mid = x_right_mid - x_left_mid
    distance_high = x_right_high - x_left_high

    valid = distance_low > distance_threshold_min and \
            distance_low < distance_threshold_max and \
            distance_mid > distance_threshold_min and \
            distance_mid < distance_threshold_max and \
            distance_high > distance_threshold_min and \
            distance_high < distance_threshold_max
    
    if not valid:
        print(distance_low, distance_mid, distance_high)
        print('sanity check failed')

    return valid


def find_lines(binary_warped):
    left_line.prepare_for_next_frame()
    right_line.prepare_for_next_frame()
    previous_left_fit = left_line.get_previous_fit()
    previous_right_fit = right_line.get_previous_fit()
    if previous_left_fit is None or previous_right_fit is None:
        left_fit, right_fit, left_fit_m, right_fit_m = find_lines_from_scratch(binary_warped)
    else:
        left_fit, right_fit, left_fit_m, right_fit_m = find_lines_using_previous_frame(binary_warped, previous_left_fit, previous_right_fit)

    if sanity_check(binary_warped, left_fit, right_fit):
        left_line.set_current_fit(left_fit)
        right_line.set_current_fit(right_fit)
        left_line.add_current_fit_m(left_fit_m)
        right_line.add_current_fit_m(right_fit_m)
        return True
    else:
        print('skip this frame')
        return False


def project_lines(undistorted, binary_warped, Minv):
    left_fit = left_line.get_smooth_fit()
    right_fit = right_line.get_smooth_fit()
    if left_fit is None or right_fit is None:
        print('projection failed')
        return undistorted

    h, w = binary_warped.shape[:2]
    ploty = np.linspace(0, h-1, h)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw the lines on
    zero_warped = np.zeros_like(binary_warped).astype(np.uint8)
    color_warped = np.dstack((zero_warped, zero_warped, zero_warped))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warped, np.int_([pts]), (0,255,0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    lines = cv2.warpPerspective(color_warped, Minv, (w,h))
    # Combine the result with the original undistorred image
    result = cv2.addWeighted(undistorted, 1, lines, 0.3, 0)
    return result


def display_curvature(image):
    h, w = image.shape[:2]
    y = h - 1
    left_curvature = left_line.get_curvature(y)
    right_curvature = right_line.get_curvature(y)
    if left_curvature is None or right_curvature is None:
        return image

    left_text = 'Left: radius of curvature = %dm' % left_curvature
    right_text = 'Right: radius of curvature = %dm' % right_curvature
    white = (255, 255, 255)
    cv2.putText(image, left_text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, white, 1, cv2.LINE_AA)
    cv2.putText(image, right_text, (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, white, 1, cv2.LINE_AA)
    return image


def display_position(image):
    left_fit = left_line.get_smooth_fit()
    right_fit = right_line.get_smooth_fit()
    if left_fit is None or right_fit is None:
        return image

    h, w = image.shape[:2]
    y = h - 1

    leftx = left_fit[0]*y**2 + left_fit[1]*y + left_fit[2]
    rightx = right_fit[0]*y**2 + right_fit[1]*y + right_fit[2]
    lane_center = (leftx + rightx) / 2
    offset = np.absolute(lane_center-w/2) * P['m_per_pixel_x']
    if lane_center < w/2:
        text = 'Vehicle is %.2fm left of center' % offset
    else:
        text = 'Vehicle is %.2fm right of center' % offset
    white = (255, 255, 255)
    cv2.putText(image, text, (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1, white, 1, cv2.LINE_AA)
    return image


def process(image):
    camera_matrix, distortion_coefficients = load_camera_calibration()
    binary_warped, undistorted, Minv = pipeline(image, camera_matrix, distortion_coefficients)
    found = find_lines(binary_warped)
    projected = project_lines(undistorted, binary_warped, Minv)
    annotated = display_curvature(projected)
    annotated = display_position(annotated)
    return annotated


if __name__ == '__main__':
    # Calibrate camera again when necessary
    if len(sys.argv) == 2 and sys.argv[1] == 'cal':
        calibrate_camera()

    # output_path  = 'test_videos_output/project_video.mp4'
    # input_clip = VideoFileClip("project_video.mp4")
    # output_clip = input_clip.fl_image(process)
    # output_clip.write_videofile(output_path, audio=False)

    # clip = VideoFileClip("project_video.mp4")
    # clip.save_frame('test_images/test222.jpg', 22.2)
    # clip.save_frame('test_images/test223.jpg', 22.3)
    # clip.save_frame('test_images/test224.jpg', 22.4)

    # RGBA
    # rgba = mpimg.imread('test_images/shadow.jpg')
    # image = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
    
    # RGB
    camera_matrix, distortion_coefficients = load_camera_calibration()
    calibration = mpimg.imread('camera_cal/calibration1.jpg')
    calibrated = undistort_image(calibration, camera_matrix, distortion_coefficients)
    mpimg.imsave('output_images/calibrated.jpg', calibrated)

    straight_line = mpimg.imread('test_images/straight_lines2.jpg')
    M, Minv = perspective_transform_matrix()
    warped_straight_line = perspective_transform(straight_line, M)
    mpimg.imsave('output_images/warped_straight_line.jpg', warped_straight_line)

    image = mpimg.imread('test_images/test416.jpg')
    binary_warped, undistorted, Minv = pipeline(image, camera_matrix, distortion_coefficients, display=True)
    left_fit, right_fit, a, b = find_lines_from_scratch(binary_warped, display=True)
    left_line.set_current_fit(left_fit)
    right_line.set_current_fit(right_fit)
    projected = project_lines(undistorted, binary_warped, Minv)
    mpimg.imsave('output_images/projected.jpg', projected)

    plt.show()