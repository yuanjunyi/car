import csv
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D, Cropping2D


def read_data():
    lines = []
    with open('data/driving_log.csv') as f:

        # Use the first row of as keys
        reader = csv.DictReader(f)
        for line in reader:
            lines.append(line)

    images = []
    measurements = []
    for line in lines:
        source_path = line['center']
        filename = source_path.split('/')[-1]
        current_path = 'data/IMG/' + filename
        image = mpimg.imread(current_path)
        images.append(image)
        measurement = float(line['steering'])
        measurements.append(measurement)
        images.append(np.fliplr(image))
        measurements.append(-measurement)

    X_train = np.array(images)
    y_train = np.array(measurements)

    return X_train, y_train


def main():
    X_train, y_train = read_data()

    model = Sequential()
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: x/255.0-0.5))
    model.add(Conv2D(nb_filter=24, nb_row=5, nb_col=5, subsample=(2, 2), border_mode='valid'))
    model.add(Conv2D(nb_filter=36, nb_row=5, nb_col=5, subsample=(2, 2), border_mode='valid'))
    model.add(Conv2D(nb_filter=48, nb_row=5, nb_col=5, subsample=(2, 2), border_mode='valid'))
    model.add(Conv2D(nb_filter=64, nb_row=3, nb_col=3, subsample=(1, 1), border_mode='valid'))
    model.add(Conv2D(nb_filter=64, nb_row=3, nb_col=3, subsample=(1, 1), border_mode='valid'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    h = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
    model.save('model.h5')

    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


if __name__ == '__main__':
    main()
