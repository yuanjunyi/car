import sys
import cv2
import glob
import time
import csv
import itertools
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip


P = {
    'orientations': 9,
}


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
                                  transform_sqrt=True,
                                  feature_vector=feature_vector)
        return features, hog_image
    else:      
        features = hog(gray,
                       orientations=orientations, 
                       pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                       cells_per_block=(cells_per_block, cells_per_block),
                       block_norm='L2-Hys',
                       visualise=visualise,
                       transform_sqrt=True,
                       feature_vector=feature_vector)
        return features


def bin_spatial(image, size=(16, 16)):
    features = cv2.resize(image, size).ravel()
    return features

# `image` should be np.float32 and pixel data range is [0, 1]
def color_hist(image, nbins=32, bins_range=(0, 1)):
    hist_features = []
    for channel in range(image.shape[2]):
        hist, bin_edges = np.histogram(image[:,:,channel], bins=nbins, range=bins_range)
        hist_features.append(hist)
    return np.concatenate(hist_features)

# `image` should be np.float32 and pixel data range is [0, 1] 
def extract_features(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    hog_features = []
    for channel in range(image.shape[2]):
        hog_features.append(get_hog_features(image[:,:,channel],
                                             orientations=P['orientations'],
                                             pixels_per_cell=8,
                                             cells_per_block=2,
                                             visualise=False,
                                             feature_vector=True))
    features = []
    features.append(np.concatenate(hog_features))
    # features.append(bin_spatial(image))
    # features.append(color_hist(image))
    return np.concatenate(features)


def train_model():
    car_images = []
    car_images.extend(glob.glob('vehicles/GTI_Far/image*.png'))
    car_images.extend(glob.glob('vehicles/GTI_Left/image*.png'))
    car_images.extend(glob.glob('vehicles/GTI_MiddleClose/image*.png'))
    car_images.extend(glob.glob('vehicles/GTI_Right/image*.png'))
    car_images.extend(glob.glob('vehicles/KITTI_extracted/*.png'))
    print('car', len(car_images))

    notcar_images = []
    notcar_images.extend(glob.glob('non-vehicles/Extras/extra*.png'))
    notcar_images.extend(glob.glob('non-vehicles/GTI/image*.png'))
    notcar_images.extend(glob.glob('teach_images/*.png'))
    print('notcar', len(notcar_images))
    t2 = time.time()

    print('extracting features...')
    car_features = [extract_features(mpimg.imread(f)) for f in car_images]
    notcar_features = [extract_features(mpimg.imread(f)) for f in notcar_images]
    
    # udacity_car_features = udacity_dataset_car_features(5000)
    # car_features.extend(udacity_car_features)

    # udacity_car_features, udacity_notcar_features = udacity_dataset_features()
    # car_features.extend(udacity_car_features)
    # notcar_features.extend(udacity_notcar_features)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
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
    return X_scaler, svc

# `image` should be np.float32 and pixel data range is [0, 1]
def find_cars(image, scale, ystart, ystop, X_scaler, svc):
    draw_image = np.copy(image)

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

    xsteps = (w // pixels_per_cell - window // pixels_per_cell) // cell_per_step + 1
    ysteps = (h // pixels_per_cell - window // pixels_per_cell) // cell_per_step + 1

    channels = [image[:,:,0], image[:,:,1], image[:,:,2]]
    hogs = [get_hog_features(c, P['orientations'], pixels_per_cell, cells_per_block, visualise=False, feature_vector=False) for c in channels]

    bbox_list = []
    for xstep in range(xsteps):
        for ystep in range(ysteps):
            xcell = xstep * cell_per_step
            ycell = ystep * cell_per_step

            window_hogs = [h[ycell:ycell+block_per_window, xcell:xcell+block_per_window].ravel() for h in hogs]
            window_hog_features = np.hstack(window_hogs)

            xleft = xcell * pixels_per_cell
            ytop = ycell * pixels_per_cell
            # window_image = cv2.resize(image[ytop:ytop+window, xleft:xleft+window], (64,64))

            # spatial_features = bin_spatial(window_image)
            # hist_features = color_hist(window_image)
            # window_features = np.hstack((window_hog_features, spatial_features, hist_features)).reshape(1, -1)
            window_features = window_hog_features.reshape(1, -1)
            scaled_window_features = X_scaler.transform(window_features)
            prediction = svc.predict(scaled_window_features)

            if prediction == 1:
                unscaled_xleft = np.int(xleft*scale)
                unscaled_ytop = np.int(ytop*scale)
                unscaled_window = np.int(window*scale)
                bbox_list.append(((unscaled_xleft, unscaled_ytop+ystart),
                                  (unscaled_xleft+unscaled_window, unscaled_ytop+ystart+unscaled_window)))
    return bbox_list


def draw_bbox(image, bbox_list):
    assert(image.dtype == np.float32)
    draw_image = np.copy(image)
    for b in bbox_list:
        cv2.rectangle(draw_image, b[0], b[1], (0,0,1), 3)
    return draw_image


def build_heatmap(image, bbox_list):
    heatmap = np.zeros_like(image[:,:,0], np.uint8)
    for b in bbox_list:
        heatmap[b[0][1]:b[1][1], b[0][0]:b[1][0]] += 1
    return heatmap


def heatmap_threshold(heatmap, threshold):
    heatmap_thresh = np.copy(heatmap)
    heatmap_thresh[heatmap_thresh <= threshold] = 0
    return heatmap_thresh


def extract_bbox(heatmap):
    labels, nlabel = label(heatmap)

    bbox_list = []
    for i in range(1, nlabel+1):
        nonzero = (labels == i).nonzero()
        nonzeroy = nonzero[0]
        nonzerox = nonzero[1]
        bbox_list.append(((np.min(nonzerox), np.min(nonzeroy)),
                          (np.max(nonzerox), np.max(nonzeroy))))
    return bbox_list


def extract_features_row(row):
    image = mpimg.imread('object-detection-crowdai/'+row['Frame'])
    ymin = min(int(row['ymin']), int(row['ymax']))
    ymax = max(int(row['ymin']), int(row['ymax']))
    xmin = min(int(row['xmin']), int(row['xmax']))
    xmax = max(int(row['xmin']), int(row['xmax']))
    if ymax<=ymin or xmax<=xmin:
        return None
    else:        
        patch = cv2.resize(image[ymin:ymax, xmin:xmax], (64,64))
        patch = patch.astype(np.float32) / 255
        return extract_features(patch)


def udacity_dataset_car_features(N):
    car_features = []
    with open('object-detection-crowdai/labels.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['Label'] == 'Car' and len(car_features) < N:
                features = extract_features_row(row)
                if features is not None:
                    car_features.append(features)
                    if len(car_features) == N:
                        return car_features


def udacity_dataset_features():
    car_features = []
    notcar_features = []
    with open('object-detection-crowdai/labels.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['Label'] == 'Car' and len(car_features) < 5000:
                features = extract_features_row(row)
                if features is not None:
                    car_features.append(features)
            if row['Label'] == 'Pedestrian' and len(notcar_features) < 5000:
                features = extract_features_row(row)
                if features is not None:
                    notcar_features.append(features)
            if len(car_features) == 5000 and len(notcar_features) == 5000:
                return car_features, notcar_features


bbox_list_buffer = []
def process(image):
    X_scaler, svc = load_model()
    image = image.astype(np.float32) / 255
    
    bbox_list = []
    bbox_list.extend(find_cars(image, 2, 400, 700, X_scaler, svc))
    bbox_list.extend(find_cars(image, 1.5, 400, 700, X_scaler, svc))

    # global bbox_list_buffer
    if len(bbox_list_buffer) == 3:
        bbox_list_buffer.pop(0)
    bbox_list_buffer.append(bbox_list)
    accumulated_bbox_list = list(itertools.chain.from_iterable(bbox_list_buffer))

    heatmap = build_heatmap(image, accumulated_bbox_list)
    heatmap_thresh = heatmap_threshold(heatmap, 12)

    bbox_list = extract_bbox(heatmap_thresh)
    annoted = draw_bbox(image, bbox_list)
    return (annoted*255).astype(np.uint8)


def test(X_scaler, svc):
    image = mpimg.imread('test_images/test11.jpg')
    assert(image.dtype == np.uint8)
    image = image.astype(np.float32) / 255
    
    bbox_list = []
    bbox_list.extend(find_cars(image, 2, 400, 700, X_scaler, svc))
    bbox_list.extend(find_cars(image, 1.5, 400, 700, X_scaler, svc))
    detection = draw_bbox(image, bbox_list)
    
    heatmap = build_heatmap(image, bbox_list)
    heatmap_thresh = heatmap_threshold(heatmap, 2)

    bbox_list = extract_bbox(heatmap_thresh)
    annoted = draw_bbox(image, bbox_list)

    f, ax = plt.subplots(2, 2)
    ax[0, 0].set_title('detection')
    ax[0, 0].imshow(detection)
    ax[0, 1].set_title('heatmap')
    ax[0, 1].imshow(heatmap, cmap='hot')
    ax[1, 0].set_title('heatmap_thresh')
    ax[1, 0].imshow(heatmap_thresh, cmap='hot')
    ax[1, 1].set_title('annoted')
    ax[1, 1].imshow(annoted)
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'train':
        X_scaler, svc = train_model()
    else:
        X_scaler, svc = load_model()

    test(X_scaler, svc)

    # clip = VideoFileClip("test_video.mp4")
    # clip.save_frame('test_images/test07.jpg', 0.7)
    # clip.save_frame('test_images/test08.jpg', 0.8)
    # clip.save_frame('test_images/test09.jpg', 0.9)
    # clip.save_frame('test_images/test10.jpg', 1.0)
    # clip.save_frame('test_images/test11.jpg', 1.1)
    # clip.save_frame('test_images/test12.jpg', 1.2)
    # clip.save_frame('test_images/test13.jpg', 1.3)
    # clip.save_frame('test_images/test14.jpg', 1.4)
    # clip.save_frame('test_images/test15.jpg', 1.5)
    # clip.save_frame('test_images/test16.jpg', 1.6)
    # clip.save_frame('test_images/test17.jpg', 1.7)

    # output_path  = 'test_videos_output/test_video.mp4'
    # input_clip = VideoFileClip("test_video.mp4")
    # output_clip = input_clip.fl_image(process)
    # output_clip.write_videofile(output_path, audio=False)
