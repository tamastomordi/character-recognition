import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
import numpy as np

model = keras.models.load_model('./saved_models/model_lenet5')

print("Type the name of the file: ")
file_name = input()

image = cv2.imread('./letters/' + file_name, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (28, 28), interpolation = cv2.INTER_LINEAR)
image = cv2.bitwise_not(image)

img = np.array(image)
img = img / 255
img = img.reshape(-1, 28, 28, 1)

predictions = model.predict(img)

if np.argmax(predictions) < 26:
   print("This is the letter: ", chr(97 + np.argmax(predictions)))
else:
   print("This is the digit: ", chr(48 + (np.argmax(predictions) - 26)))

plt.imshow(image, cmap='gray')
plt.show()