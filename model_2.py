import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
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

model.add(Conv2D(32, (3,3), input_shape = x_train.shape[1:], activation='relu')) 
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(units = 128, activation='relu'))
model.add(Dense(units = 36, activation='softmax'))

print(model.summary())

model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

history = model.fit(x_train, y_train, epochs = 50, validation_data=(x_val, y_val))

model.save("saved_models/model_2/")

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
 conv2d_3 (Conv2D)           (None, 26, 26, 32)        320       
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 13, 13, 32)       0         
 2D)                                                             
                                                                 
 conv2d_4 (Conv2D)           (None, 11, 11, 32)        9248      
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 5, 5, 32)         0         
 2D)                                                             
                                                                 
 flatten_1 (Flatten)         (None, 800)               0         
                                                                 
 dense_3 (Dense)             (None, 128)               102528    
                                                                 
 dense_4 (Dense)             (None, 36)                4644      
                                                                 
=================================================================
Total params: 116,740
Trainable params: 116,740
Non-trainable params: 0
_________________________________________________________________
"""