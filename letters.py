import numpy as np
import matplotlib.pyplot as plt

from emnist import extract_training_samples

x_letters = np.array(extract_training_samples('letters')[0])
y_letters = np.array(extract_training_samples('letters')[1]) - 1

x_digits = np.array(extract_training_samples('digits')[0])
y_digits = np.array(extract_training_samples('digits')[1]) + 26

x = np.concatenate((x_letters, x_digits), axis = 0)
y = np.concatenate((y_letters, y_digits), axis = 0)

curr = 200000

if y[curr] < 27:
   print("This is the letter: ", chr(97 + y[curr]))
else:
   print("This is the digit: ", chr(48 + (y[curr] - 26)))

plt.imshow(x[curr], cmap = plt.cm.binary)
plt.show()