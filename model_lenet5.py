import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, AveragePooling2D
from matplotlib import pyplot as plt
from data import x_train, y_train, x_val, y_val

# Normalization:
x_train = x_train / 255
x_val = x_val / 255

# Reshape:
IMG_SIZE = 28
x_train = x_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
x_val = x_val.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# The model:

model = Sequential()

model.add(Conv2D(filters = 6, kernel_size = (5,5), activation = 'sigmoid' , input_shape = x_train.shape[1:], padding = 'same'))
model.add(AveragePooling2D())
model.add(Conv2D(filters = 16, kernel_size = (5,5), activation = 'sigmoid'))
model.add(AveragePooling2D())
model.add(Conv2D(120, kernel_size = (5,5), activation='sigmoid'))

model.add(Flatten())

model.add(Dense(84, activation = 'sigmoid'))
model.add(Dense(36, activation = 'softmax'))

print(model.summary())

model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

history = model.fit(x_train, y_train, epochs = 50, validation_data = (x_val, y_val))

model.save("saved_models/model_lenet5/")

# Plot the training statistics:

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model LeNet5 Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model LeNet5 Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

"""
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_5 (Conv2D)           (None, 28, 28, 6)         156       
                                                                 
 average_pooling2d (AverageP  (None, 14, 14, 6)        0         
 ooling2D)                                                       
                                                                 
 conv2d_6 (Conv2D)           (None, 10, 10, 16)        2416      
                                                                 
 average_pooling2d_1 (Averag  (None, 5, 5, 16)         0         
 ePooling2D)                                                     
                                                                 
 conv2d_7 (Conv2D)           (None, 1, 1, 120)         48120     
                                                                 
 flatten_2 (Flatten)         (None, 120)               0         
                                                                 
 dense_4 (Dense)             (None, 84)                10164     
                                                                 
 dense_5 (Dense)             (None, 36)                3060      
                                                                 
=================================================================
Total params: 63,916
Trainable params: 63,916
Non-trainable params: 0
_________________________________________________________________
"""