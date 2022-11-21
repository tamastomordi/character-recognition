import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D

from emnist import extract_training_samples

x_letters = np.array(extract_training_samples('letters')[0])
y_letters = np.array(extract_training_samples('letters')[1]) - 1

x_digits = np.array(extract_training_samples('digits')[0])
y_digits = np.array(extract_training_samples('digits')[1]) + 26

x = np.concatenate((x_letters, x_digits), axis = 0)
y = np.concatenate((y_letters, y_digits), axis = 0)

# Normalization
x = x / 255

# Reshape: (124800, 28, 28) -> (124800, 28, 28, 1)
IMG_SIZE = 28
x = x.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Creating the neural network

model = Sequential()

## 1st Convolutional Layer
model.add(Conv2D(64, (3,3), input_shape = x.shape[1:])) 
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
model.add(Dense(48))
model.add(Activation("relu"))

## 3rd Fully Connected Layer
model.add(Dense(36))
model.add(Activation("softmax"))

#print(model.summary())

model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

# Training the model
model.fit(x, y, epochs = 5, validation_split = 0.3)

model.save("model1/")