import csv
import cv2
import sys
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras.backend import get_value, set_value


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while True:
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            steerings = []
            for sample in batch_samples:
                source_path = sample['center']
                filename = source_path.split('/')[-1]
                current_path = 'data/IMG/' + filename
                image = mpimg.imread(current_path)
                images.append(image)
                steering = float(sample['steering'])
                steerings.append(steering)

                # Augment data by flipping images.
                images.append(np.fliplr(image))
                steerings.append(-steering)

            X = np.array(images)
            y = np.array(steerings)
            yield X, y


def read_data():
    lines = []
    with open('data/driving_log.csv') as f:
        reader = csv.DictReader(f) # Use the first row as keys
        for line in reader:
            lines.append(line)
    return lines


def resize(x):
    import tensorflow as tf
    return tf.image.resize_images(x, (80,160))


def main(resume, epochs, learning_rate):
    samples = read_data()
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    model = None
    if resume:
        model = load_model('model.h5')
    else:
        model = Sequential()
        model.add(Lambda(resize, input_shape=(160,320,3)))
        model.add(Cropping2D(cropping=((25,10), (0,0))))
        model.add(Lambda(lambda x: x/255.0-0.5))
        model.add(Conv2D(nb_filter=24, nb_row=5, nb_col=5, subsample=(2, 2), border_mode='valid'))
        model.add(Conv2D(nb_filter=36, nb_row=5, nb_col=5, subsample=(2, 2), border_mode='valid'))
        model.add(Conv2D(nb_filter=48, nb_row=5, nb_col=5, subsample=(2, 2), border_mode='valid'))
        model.add(Conv2D(nb_filter=64, nb_row=3, nb_col=3, subsample=(1, 1), border_mode='valid'))
        model.add(Flatten())
        model.add(Dense(100))
        model.add(Dense(50))
        model.add(Dense(10))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')

    print('learning rate = ' + str(get_value(model.optimizer.lr)))
    set_value(model.optimizer.lr, learning_rate)
    print('learning rate = ' + str(get_value(model.optimizer.lr)))
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
    resume, epochs, learning_rate = False, 5, 0.001
    if len(sys.argv) == 4:
        resume = sys.argv[1] == 'resume'
        epochs = int(sys.argv[2])
        learning_rate = float(sys.argv[3])
    main(resume, epochs, learning_rate)
