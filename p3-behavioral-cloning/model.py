import csv
import cv2
import sys
import re
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras.backend import get_value, set_value


def generator(data_path, samples, batch_size=32):
    num_samples = len(samples)
    while True:
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            steerings = []
            for sample in batch_samples:
                # sample = [center, left, right, steering, throttle, brake, speed]
                path_center = data_path + '/IMG/' + re.split(r'/|\\', sample[0])[-1]
                path_left = data_path + '/IMG/' + re.split(r'/|\\', sample[1])[-1]
                path_right = data_path + '/IMG/' + re.split(r'/|\\', sample[2])[-1]
                steering = float(sample[3])

                images.append(mpimg.imread(path_center))
                steerings.append(steering)
                images.append(mpimg.imread(path_left))
                steerings.append(steering+0.3)
                images.append(mpimg.imread(path_right))
                steerings.append(steering-0.3)

                # Augment data by flipping images.
                flipped_images = [np.fliplr(x) for x in images[-3:]]
                flipped_steerings = [-x for x in steerings[-3:]]
                images.extend(flipped_images)
                steerings.extend(flipped_steerings)

            X = np.array(images)
            y = np.array(steerings)
            yield X, y


def read_data(data_path):
    lines = []
    with open(data_path+'/driving_log.csv') as f:
        reader = csv.reader(f)
        for line in reader:
            lines.append(line)
    return lines


def resize(image):
    import tensorflow as tf
    return tf.image.resize_images(image, (80,160))


def main(data_path, resume, epochs, learning_rate):
    samples = read_data(data_path)
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    train_generator = generator(data_path, train_samples, batch_size=64)
    validation_generator = generator(data_path, validation_samples, batch_size=64)

    model = None
    if resume:
        model = load_model('model.h5')
    else:
        model = Sequential()
        model.add(Lambda(resize, input_shape=(160,320,3)))
        model.add(Cropping2D(cropping=((20,10), (0,0))))
        model.add(Lambda(lambda x: x/255.0-0.5))

        # Convolution
        model.add(Conv2D(nb_filter=16, nb_row=3, nb_col=3, subsample=(1, 1), border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='valid'))

        # Convolution
        model.add(Conv2D(nb_filter=32, nb_row=3, nb_col=3, subsample=(1, 1), border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='valid'))

        # Convolution
        model.add(Conv2D(nb_filter=64, nb_row=3, nb_col=3, subsample=(1, 1), border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='valid'))

        # Convolution
        model.add(Conv2D(nb_filter=128, nb_row=3, nb_col=3, subsample=(1, 1), border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='valid'))

        # Convolution
        model.add(Conv2D(nb_filter=128, nb_row=3, nb_col=3, subsample=(1, 1), border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='valid'))

        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(100))
        model.add(Dense(50))
        model.add(Dense(10))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')

    print(model.summary())

    set_value(model.optimizer.lr, learning_rate)
    h = model.fit_generator(generator=train_generator, validation_data=validation_generator, samples_per_epoch=len(train_samples), nb_val_samples=len(validation_samples), nb_epoch=epochs)
    model.save('model.h5')

    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


if __name__ == '__main__':
    data_path, resume, epochs, learning_rate = 'data', False, 10, 0.001
    if len(sys.argv) == 2:
        data_path = sys.argv[1]
    if len(sys.argv) == 5:
        data_path = sys.argv[1]
        resume = sys.argv[2] == 'resume'
        epochs = int(sys.argv[3])
        learning_rate = float(sys.argv[4])
    main(data_path, resume, epochs, learning_rate)
