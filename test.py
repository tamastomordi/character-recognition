import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import cv2
import numpy as np
from tensorflow import keras

model = keras.models.load_model('./saved_models/model_lenet5')

x, y = [], []

for filename in os.listdir('./letters/'):
   img = cv2.imread(os.path.join('./letters/', filename), cv2.IMREAD_GRAYSCALE)
   if img is not None:
      img = cv2.resize(img, (28, 28), interpolation = cv2.INTER_LINEAR)
      img = cv2.bitwise_not(img)
      img = np.array(img) / 255
      x.append(img)
      if ord(filename[0]) < 97:
         label = ord(filename[0]) - (48 - 26)
      else: 
         label = ord(filename[0]) - 97
      y.append(label)

x = np.array(x).reshape(-1, 28, 28, 1)
y = np.array(y)

loss, accuracy = model.evaluate(x, y)
print('Loss: ', loss)
print('Accuracy: ', accuracy)