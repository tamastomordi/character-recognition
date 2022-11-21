import numpy as np
import matplotlib.pyplot as plt

from emnist import extract_training_samples, extract_test_samples

I_TRAINING = np.array(extract_training_samples('letters')[0])
L_TRAINING = np.array(extract_training_samples('letters')[1])

I_TEST = np.array(extract_test_samples('letters')[0])
L_TEST = np.array(extract_test_samples('letters')[1])

print("This is a letter: ", chr(96 + L_TRAINING[5]))

plt.imshow(I_TRAINING[100], cmap = plt.cm.binary)
plt.show()