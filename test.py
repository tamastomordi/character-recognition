import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
from tensorflow import keras

from emnist import extract_test_samples

x_letters = np.array(extract_test_samples('letters')[0])
y_letters = np.array(extract_test_samples('letters')[1]) - 1

x_digits = np.array(extract_test_samples('digits')[0])
y_digits = np.array(extract_test_samples('digits')[1]) + 26

x = np.concatenate((x_letters, x_digits), axis = 0)
y = np.concatenate((y_letters, y_digits), axis = 0)

x = keras.utils.normalize(x, axis = 1)
x = x.reshape(-1, 28, 28, 1)

model = keras.models.load_model('./model_4')

loss, accuracy = model.evaluate(x, y)
print('Loss: ', loss)
print('Accuracy: ', accuracy)

# On EMNIST test samples:
# Model1 - Loss: 2.22 Acc: 0.36
# Model2 - Loss: 1.16 Acc: 0.71
# Model3 - Loss: 0.78 Acc: 0.79
# Model4 - Loss: 0.49 Acc: 0.88