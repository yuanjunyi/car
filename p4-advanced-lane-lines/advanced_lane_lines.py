import sys
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def calibrate_camera(new_calibration):
    if not new_calibration:
        camera_matrix = np.load('camera_matrix.npy')
        distortion_coefficients = np.load('distortion_coefficients.npy')
        return camera_matrix, distortion_coefficients
    else:
        nx = 9
        ny = 6

        points_3D_in_real_world = []
        points_2D_in_image_plane = []

        # Define points_3D as np.float32 instead of np.float because
        # calibrateCamera() requests Point3f.
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
        [[194, 719],
         [1121, 719],
         [687, 452],
         [587, 452]])

    # 4 desired coordinates
    dst = np.float32(
        [[315, 719],
         [965, 719],
         [965, 0],
         [315, 0]])

    M = cv2.getPerspectiveTransform(src, dst)
    return M


def region_of_interest(image):
    mask = np.zeros_like(image, np.float)
    mask_color = 1
    h, w = image.shape[:2]
    vertices = np.array([[[w*0.45, h*0.6], [w*0.6, h*0.6], [w*0.95, h], [w*0.15, h]]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


if __name__ == '__main__':

    # Calibrate camera
    new_calibration = False
    if len(sys.argv) == 2:
        new_calibration = sys.argv[1] == 'calibrate'
    camera_matrix, distortion_coefficients = calibrate_camera(new_calibration)

    # Pipeline
    image = mpimg.imread('test_images/test2.jpg')

    # Undistort image
    undistorted = undistort_image(image, camera_matrix, distortion_coefficients)

    # HLS threshold
    hls_threshold = hls_threshold(undistorted, threshold=(170, 255))

    # Gradient threshold
    gray = cv2.cvtColor(undistorted, cv2.COLOR_RGB2GRAY)
    gradientx = abs_sobel_threshold(gray, orient='x', ksize=3, threshold=(20, 100))
    gradienty = abs_sobel_threshold(gray, orient='y', ksize=3, threshold=(20, 100))
    gradient_magnitude = gradient_magnitude_threshold(gray, ksize=9, threshold=(30, 100))
    gradient_direction = gradient_direction_threshold(gray, ksize=15, threshold=(0.7, 1.3))

    # Combine threshold
    combine = np.zeros_like(gray, np.float)
    combine[(hls_threshold == 1) | \
            ((gradientx == 1) & (gradienty == 1)) | \
            ((gradient_magnitude == 1) & (gradient_direction == 1))] = 1

    # Filter uninteresting region
    interesting_region = region_of_interest(combine)

    # Perspective transform
    M = perspective_transform_matrix()
    warped = perspective_transform(interesting_region, M)

    f, ax = plt.subplots(3, 4, figsize=(16,8))
    ax[0, 0].set_title('image')
    ax[0, 0].imshow(image)
    ax[0, 1].set_title('undistorted')
    ax[0, 1].imshow(undistorted)
    ax[0, 2].set_title('hls_threshold')
    ax[0, 2].imshow(hls_threshold, cmap='gray')
    ax[0, 3].set_title('gray')
    ax[0, 3].imshow(gray, cmap='gray')
    ax[1, 0].set_title('gradientx')
    ax[1, 0].imshow(gradientx, cmap='gray')
    ax[1, 1].set_title('gradienty')
    ax[1, 1].imshow(gradienty, cmap='gray')
    ax[1, 2].set_title('gradient_magnitude')
    ax[1, 2].imshow(gradient_magnitude, cmap='gray')
    ax[1, 3].set_title('gradient_direction')
    ax[1, 3].imshow(gradient_direction, cmap='gray')
    ax[2, 0].set_title('combine')
    ax[2, 0].imshow(combine, cmap='gray')
    ax[2, 1].set_title('interesting_region')
    ax[2, 1].imshow(interesting_region, cmap='gray')
    ax[2, 2].set_title('warped')
    ax[2, 2].imshow(warped, cmap='gray')
    plt.show()