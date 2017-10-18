import sys
import cv2
import glob
import time
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.feature import hog


def get_hog_features(gray,
                     orientations,
                     pixels_per_cell,
                     cells_per_block,
                     visualise=False,
                     feature_vector=True):
    if visualise == True:
        features, hog_image = hog(gray,
                                  orientations=orientations, 
                                  pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                                  cells_per_block=(cells_per_block, cells_per_block),
                                  block_norm='L2-Hys',
                                  visualise=visualise,
                                  feature_vector=feature_vector)
        return features, hog_image
    else:      
        features = hog(gray,
                       orientations=orientations, 
                       pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                       cells_per_block=(cells_per_block, cells_per_block),
                       block_norm='L2-Hys',
                       visualise=visualise,
                       feature_vector=feature_vector)
        return features


def extract_features(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    hog_features = []
    for channel in range(image.shape[2]):
        hog_features.extend(get_hog_features(image[:,:,channel],
                                             orientations=9,
                                             pixels_per_cell=8,
                                             cells_per_block=2,
                                             visualise=False,
                                             feature_vector=True))
    return np.ravel(hog_features)


def train_model():
    car_images = []
    car_images.extend(glob.glob('vehicles/GTI_Far/image*.png'))
    car_images.extend(glob.glob('vehicles/GTI_Left/image*.png'))
    car_images.extend(glob.glob('vehicles/GTI_MiddleClose/image*.png'))
    car_images.extend(glob.glob('vehicles/GTI_Right/image*.png'))
    car_images.extend(glob.glob('vehicles/KITTI_extracted/*.png'))

    notcar_images = []
    notcar_images.extend(glob.glob('non-vehicles/Extras/extra*.png'))
    notcar_images.extend(glob.glob('non-vehicles/GTI/image*.png'))
    t2 = time.time()

    print('extracting features...')
    car_features = [extract_features(mpimg.imread(f)) for f in car_images]
    notcar_features = [extract_features(mpimg.imread(f)) for f in notcar_images]

    X = np.vstack(car_features+notcar_features).astype(np.float64)
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features)))) 
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2)
    t3 = time.time()
    print(round(t3-t2), 'seconds to extract features')

    print('training...')
    svc = LinearSVC()
    svc.fit(X_train, y_train)
    t4 = time.time()
    print(round(t4-t3), 'seconds to train svc')
    
    print('test accuracy =', round(svc.score(X_test, y_test), 4))

    joblib.dump(X_scaler, 'X_scaler.pkl')
    joblib.dump(svc, 'svc.pkl')
    print('model saved')
    return X_scaler, svc


def load_model():
    X_scaler = joblib.load('X_scaler.pkl')
    svc = joblib.load('svc.pkl')
    print('model loaded')
    return X_scaler, svc


def find_cars(image, scale, X_scaler, svc):
    assert(image.dtype == np.float32)
    draw_image = np.copy(image)

    ystart = 400
    ystop = 650
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    image = image[ystart:ystop, :, :]
    h, w = image.shape[:2]
    image = cv2.resize(image, (int(w/scale), int(h/scale)))
    h, w = image.shape[:2]

    pixels_per_cell = 8
    cells_per_block = 2
    window = 64
    cell_per_step = 2

    block_per_window = window // pixels_per_cell - cells_per_block + 1

    # xnumber_of_blocks = w // pixels_per_cell - cells_per_block + 1
    # ynumber_of_blocks = h // pixels_per_cell - cells_per_block + 1
    xsteps = (w // pixels_per_cell - window // pixels_per_cell) // cell_per_step + 1
    ysteps = (h // pixels_per_cell - window // pixels_per_cell) // cell_per_step + 1

    channels = [image[:,:,0], image[:,:,1], image[:,:,2]]
    hogs = [get_hog_features(c, 9, pixels_per_cell, cells_per_block, visualise=False, feature_vector=False) for c in channels]

    for xstep in range(xsteps):
        for ystep in range(ysteps):
            xcell = xstep * cell_per_step
            ycell = ystep * cell_per_step

            window_hogs = [h[ycell:ycell+block_per_window, xcell:xcell+block_per_window].ravel() for h in hogs]
            window_hog_features = np.hstack(window_hogs)

            xleft = xcell * pixels_per_cell
            ytop = ycell * pixels_per_cell
            # window_image = cv2.resize(image[ytop:ytop+window, xleft:xleft+window], (64,64))

            window_features = window_hog_features
            window_features = window_features.reshape(1, -1)
            scaled_window_features = X_scaler.transform(window_features)
            prediction = svc.predict(scaled_window_features)

            if prediction == 1:
                unscaled_xleft = np.int(xleft*scale)
                unscaled_ytop = np.int(ytop*scale)
                unscaled_window = np.int(window*scale)
                cv2.rectangle(draw_image,
                              (unscaled_xleft, unscaled_ytop+ystart),
                              (unscaled_xleft+unscaled_window, unscaled_ytop+ystart+unscaled_window),
                              (0,0,1),
                              3)
    return draw_image


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'train':
        X_scaler, svc = train_model()
    else:
        X_scaler, svc = load_model()

    image = mpimg.imread('test_images/test1.jpg')
    assert(image.dtype == np.uint8)
    image = image.astype(np.float32) / 255
    annoted = find_cars(image, 2, X_scaler, svc)
    plt.imshow(annoted)
    plt.show()