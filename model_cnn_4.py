import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from emnist import extract_training_samples

x_letters = np.array(extract_training_samples('letters')[0])
y_letters = np.array(extract_training_samples('letters')[1]) - 1

x_digits = np.array(extract_training_samples('digits')[0])
y_digits = np.array(extract_training_samples('digits')[1]) + 26

x = np.concatenate((x_letters, x_digits), axis = 0)
y = np.concatenate((y_letters, y_digits), axis = 0)

x = x / 255

IMG_SIZE = 28
x = x.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', input_shape = x.shape[1:]))
model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
model.add(Conv2D(filters = 128, kernel_size = (4, 4), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(.25))

model.add(Flatten())

model.add(Dense(128, activation = 'relu'))
model.add(Dropout(.25))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(.25))
model.add(Dense(36, activation = 'softmax'))

print(model.summary())

model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

model.fit(x, y, epochs = 3, validation_split = 0.3)

model.save("model_3_2/")