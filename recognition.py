import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
import numpy as np

model = keras.models.load_model('./model_4')

print("Type the name of the file: ")
file_name = input()

img = cv2.imread('./letters/' + file_name)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray, (28, 28), interpolation = cv2.INTER_AREA)

newimg = keras.utils.normalize(resized, axis = 1)
newimg = np.array(newimg).reshape(-1, 28, 28, 1)

predictions = model.predict(newimg)

if np.argmax(predictions) < 26:
   print("This is the letter: ", chr(97 + np.argmax(predictions)))
else:
   print("This is the digit: ", chr(48 + (np.argmax(predictions) - 26)))

plt.imshow(resized, cmap = plt.cm.binary)
plt.show()
