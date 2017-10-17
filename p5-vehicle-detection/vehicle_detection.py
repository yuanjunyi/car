import cv2
import glob
import time
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.feature import hog


def get_hog_features(image,
                     orientations,
                     pixel_per_cell,
                     cell_per_block,
                     visualise=False,
                     feature_vector=True):
    if visualise == True:
        features, hog_image = hog(image,
                                  orientations=orientations, 
                                  pixels_per_cell=(pixel_per_cell, pixel_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  visualise=visualise,
                                  feature_vector=feature_vector)
        return features, hog_image
    else:      
        features = hog(image,
                       orientations=orientations, 
                       pixels_per_cell=(pixel_per_cell, pixel_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       visualise=visualise,
                       feature_vector=feature_vector)
        return features


def extract_features(image):
    hog_features = []
    for channel in range(image.shape[2]):
        hog_features.extend(get_hog_features(image[:,:,channel],
                                             orientations=9,
                                             pixel_per_cell=8,
                                             cell_per_block=2,
                                             visualise=False,
                                             feature_vector=True))
    return np.ravel(hog_features)


def train():
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
    scaler = StandardScaler().fit(X)
    scaled_X = scaler.transform(X)
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features)))) 
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2)
    t3 = time.time()
    print(round(t3-t2), 'seconds to extract features')

    print('training...')
    svc = LinearSVC()
    svc.fit(X_train, y_train)
    t4 = time.time()
    print(round(t4-t3), 'seconds to train SVC')
    
    print('test accuracy =', round(svc.score(X_test, y_test), 4))
    return svc


if __name__ == '__main__':
    train()