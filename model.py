import csv
import os
import cv2
import numpy as np
import sklearn

samples = []
with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

images = []
measurements = []
for line in samples:
    for i in range(3):
        source_path = line[0]
        image = cv2.imread(source_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)

augumented_images,augumented_measurements = [],[]
for image, measurement in zip(images,measurements):
    augumented_images.append(image)
    augumented_measurements.append(measurement)
    augumented_images.append(cv2.flip(image,1))
    augumented_measurements.append(measurement*(-1.0))

X_train = np.array(augumented_images)
y_train = np.array(augumented_measurements)

"""
# Split the set to training set and validation set
from sklearn.model_selection import train_test_split
train_samples,validation_samples = train_test_split(samples,test_size = 0.2)

# Define the generator,
def generator(samples,batch_size = 32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples = samples[offset:offset+batch_size]
            # Read images and angles
            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[0]
                    image = cv2.imread(source_path)
                    images.append(image)
                    angle = float(batch_sample[3])
                    angles.append(angle)

            # Flip the images and then create augumented images
            augumented_images,augumented_angles = [],[]
            for image, angle in zip(images,angles):
                augumented_images.append(image)
                augumented_angles.append(angle)
                augumented_images.append(cv2.flip(image,1))
                augumented_angles.append(angle*(-1.0))

            # Convert images and labels to numpy arrays
            X_train = np.array(augumented_images)
            y_train = np.array(augumented_angles)
            yield sklearn.utils.shuffle(X_train,y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples,batch_size = 32)
validation_generator = generator(validation_samples,batch_size = 32)
"""
# Define the model
from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Construct a CNN
model = Sequential()
model.add(Lambda(lambda x:x/255.0 - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping = ((70,25),(0,0))))
model.add(Convolution2D(24,5,5, subsample = (2,2),activation = 'relu'))
model.add(Convolution2D(36,5,5, subsample = (2,2),activation = 'relu'))
model.add(Convolution2D(48,5,5, subsample = (2,2),activation = 'relu'))
model.add(Convolution2D(64,3,3,activation = 'relu'))
model.add(Convolution2D(64,3,3,activation = 'relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Compile the model and use the opitimizer to minimize the loss
model.compile(loss = 'mse', optimizer = 'adam')

model.fit(X_train,y_train,validation_split = 0.2,shuffle = True,nb_epoch = 5)

"""
model.fit_generator(train_generator, steps_per_epoch=len(train_samples),validation_data=validation_generator, validation_steps=len(validation_samples),epochs=2, verbose = 1)
"""
# Save the model
model.save('model.h5')
