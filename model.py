import csv
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle 
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from math import ceil

# Note: use keras version 2.0.8

# Parameters
data_path = "/opt/data" # input data
steering_correction = 0.1 # steering correction angle for left and right images
height, width, ch = 160, 320, 3 # image format
batch_size = 32
train_epochs = 15

samples = []
with open(data_path+"/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# define a generator for memory-efficient training
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # steering angle
                steering = float(batch_sample[3])
                
                # images
                for flipping in range(2):
                    for direction in range(3):
                        source_path = batch_sample[direction]
                        filename = source_path.split('/')[-1]
                        current_path = data_path+'/IMG/'+filename
                        image = cv2.imread(current_path)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        # direction - 0:center, 1:left, 2:right
                        if direction==0:
                            angle = steering
                        elif direction==1:
                            angle = steering+steering_correction
                        else:
                            angle = steering-steering_correction
                            
                        # flipping - 0: original, 1: flipped
                        if flipping==1:
                            image = np.fliplr(image)
                            angle = -angle
                        
                        images.append(image)
                        angles.append(angle)
                    
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# NVIDIA architecture for self driving cars
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(height, width, ch)))
model.add(Cropping2D(cropping=((75,20),(0,0))))
model.add(Conv2D(24,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(36,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(48,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(64,(3,3),strides=(1,1),activation="relu"))
model.add(Conv2D(64,(3,3),strides=(1,1),activation="relu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100,activation="relu"))
model.add(Dense(50,activation="relu"))
model.add(Dense(10,activation="relu"))
model.add(Dense(1))

# set model checkpoint to save model after each epoch
modelCheckpoint = ModelCheckpoint(filepath = 'output_data/model_{epoch:02d}_{val_loss:.4f}.h5',
                                  verbose=1)

# compile and train the model
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator,
    steps_per_epoch=ceil(len(train_samples)/batch_size),
    validation_data=validation_generator,
    validation_steps=ceil(len(validation_samples)/batch_size),
    epochs=train_epochs, verbose=1,
    callbacks=[modelCheckpoint])

# save the newest model
model.save('model.h5')

### print the keys contained in the history object
#print(history_object.history.keys())

### plot the training and validation loss for each epoch
import matplotlib.pyplot as plt
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('Model Mean Squared Error Loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig("./output_data/training_history")
