import csv
import cv2
import matplotlib.image as mpimg
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense

def read_data():
    lines = []
    with open('data/driving_log.csv') as f:
        reader = csv.DictReader(f) # Use the first row as keys
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
        
    X_train = np.array(images)
    y_train = np.array(measurements)
    
    return X_train, y_train
    
def main():
    X_train, y_train = read_data()
    
    model = Sequential()
    model.add(Flatten(input_shape=X_train.shape[1:]))
    model.add(Dense(1))
    
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)
    
    model.save('model.h5')

if __name__ == '__main__':
    main()