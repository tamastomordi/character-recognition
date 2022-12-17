import numpy as np
import matplotlib.pyplot as plt
from emnist import extract_training_samples, extract_test_samples

# Training data:

x_letters = np.array(extract_training_samples('letters')[0])
y_letters = np.array(extract_training_samples('letters')[1]) - 1

x_digits = np.array(extract_training_samples('digits')[0])
y_digits = np.array(extract_training_samples('digits')[1]) + 26

x_train = np.concatenate((x_letters, x_digits), axis = 0)
y_train = np.concatenate((y_letters, y_digits), axis = 0)

# Validation data:

x_letters = np.array(extract_test_samples('letters')[0])
y_letters = np.array(extract_test_samples('letters')[1]) - 1

x_digits = np.array(extract_test_samples('digits')[0])
y_digits = np.array(extract_test_samples('digits')[1]) + 26

x_val = np.concatenate((x_letters, x_digits), axis = 0)
y_val = np.concatenate((y_letters, y_digits), axis = 0)

# Plot an image from the EMNIST dataset:
"""
show = 0

if y_train[show] < 26:
   print("This is the letter: ", chr(97 + y_train[show]))
else:
   print("This is the digit: ", chr(48 + (y_train[show] - 26)))

plt.imshow(x_train[show], cmap = plt.cm.binary)
plt.show()
"""