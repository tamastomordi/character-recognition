from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
import numpy as np

model = keras.models.load_model('models/test_model')

print("Type the name of the file: ")
file_name = input()

img = cv2.imread("letters/" + file_name)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray, (28, 28), interpolation = cv2.INTER_AREA)
newimg = keras.utils.normalize(resized, axis = 1)
newimg = np.array(newimg).reshape(-1, 28, 28, 1)

predictions = model.predict(newimg)

print("I think it's a letter: ", chr(96 + np.argmax(predictions)))

plt.imshow(resized, cmap = plt.cm.binary)
plt.show()