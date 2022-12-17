import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
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

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', input_shape = x_train.shape[1:]))
model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
model.add(Conv2D(filters = 128, kernel_size = (4, 4), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(.25))

model.add(Flatten())

model.add(Dense(128, activation = 'relu'))
model.add(Dropout(.25))
model.add(Dense(36, activation = 'softmax'))

print(model.summary())

model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

history = model.fit(x_train, y_train, epochs = 50, validation_data = (x_val, y_val))

model.save("saved_models/model_3/")

# Plot the training statistics:

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

"""
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 26, 26, 32)        320       
                                                                 
 conv2d_1 (Conv2D)           (None, 24, 24, 64)        18496     
                                                                 
 conv2d_2 (Conv2D)           (None, 21, 21, 128)       131200    
                                                                 
 max_pooling2d (MaxPooling2D  (None, 10, 10, 128)      0         
 )                                                               
                                                                 
 dropout (Dropout)           (None, 10, 10, 128)       0         
                                                                 
 flatten (Flatten)           (None, 12800)             0         
                                                                 
 dense (Dense)               (None, 128)               1638528   
                                                                 
 dropout_1 (Dropout)         (None, 128)               0         
                                                                 
 dense_1 (Dense)             (None, 36)                4644      
                                                                 
=================================================================
Total params: 1,793,188
Trainable params: 1,793,188
Non-trainable params: 0
_________________________________________________________________
"""