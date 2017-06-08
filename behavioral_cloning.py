# Load modules
import csv
import cv2
import numpy as np

# Load csv file:
lines = []
with open('./my_training_data_1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Load center(, left and right) images together with their steering angles
images = []
steering_angles = []
for line in lines:
    #for view in range(3):
        path = line[0]
        filename = path.split('/')[-1]
        new_path = './my_training_data_1/IMG/' + filename
        image = cv2.imread(new_path)
        image_flipped = np.fliplr(image)
        images.append(image)
        images.append(image_flipped)
        steering_angle = float(line[3])
        steering_angle_flipped = -steering_angle
        steering_angles.append(steering_angle)
        steering_angles.append(steering_angle_flipped)

X_train = np.array(images)
y_train = np.array(steering_angles)

# Load modules
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
#from keras.layers.pooling import MaxPooling2D

# Model architecture
model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(1164))
#model.add(Dropout(0.75))
model.add(Dense(100))
#model.add(Dropout(0.75))
model.add(Dense(50))
#model.add(Dropout(0.25))
model.add(Dense(10))
model.add(Dense(1))


# Train model
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

# Save model
model.save('behavioral_cloning.h5')
