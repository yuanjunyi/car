import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob


def display(img1, img2):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.set_title('img1')
    ax1.imshow(img1)
    ax2.set_title('img2')
    ax2.imshow(img2)


def calibrate_camera():
    nx = 9
    ny = 6

    points_3D_in_real_world = []
    points_2D_in_image_plane = []

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

    return camera_matrix, distortion_coefficients


def undistort_image(image, camera_matrix, distortion_coefficients):
    return cv2.undistort(image, camera_matrix, distortion_coefficients, None, camera_matrix)


if __name__ == '__main__':
    camera_matrix, distortion_coefficients = calibrate_camera()
    
    image = mpimg.imread('camera_cal/calibration1.jpg')
    undistorted_image = undistort_image(image, camera_matrix, distortion_coefficients)
    display(image, undistorted_image)
    plt.show()