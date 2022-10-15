import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from emnist import extract_training_samples, extract_test_samples

I_TRAINING = np.array(extract_training_samples('letters')[0])
L_TRAINING = np.array(extract_training_samples('letters')[1])

I_TEST = np.array(extract_test_samples('letters')[0])
L_TEST = np.array(extract_test_samples('letters')[1])

# Normalization
I_TRAINING = I_TRAINING / 255
I_TEST = I_TEST / 255

# Reshape: (124800, 28, 28) -> (124800, 28, 28, 1)
IMG_SIZE = 28
I_TRAINING = I_TRAINING.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
I_TEST = I_TEST.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Creating the neural network

model = Sequential()

## 1st Convolutional Layer
model.add(Conv2D(64, (3,3), input_shape = I_TRAINING.shape[1:])) 
model.add(Activation("relu")) # remove values < 0
model.add(MaxPooling2D(pool_size = (2, 2))) # single maximum value of 2x2

## 2nd Convolutional Layer
model.add(Conv2D(64, (3,3))) 
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))

## 3rd Convolutional Layer
model.add(Conv2D(64, (3,3))) 
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))

## 1st Fully Connected Layer
model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

## 2nd Fully Connected Layer
model.add(Dense(32))
model.add(Activation("relu"))

## 3rd Fully Connected Layer
model.add(Dense(27))
model.add(Activation("softmax"))

#print(model.summary())

model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

# Training the model
model.fit(I_TRAINING, L_TRAINING, epochs = 5, validation_split = 0.3)

# Evaluating on test data
test_loss, test_acc = model.evaluate(I_TEST, L_TEST)
print("Test loss: ", test_loss)
print("Test accuracy: ", test_acc)

model.save("models/first_model")