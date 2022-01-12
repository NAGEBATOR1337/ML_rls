from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels

i=0
train_images=np.zeros((10000,28,28,1))
with open('obychceli.npy', 'rb') as f:
       while i<10000:
              buf = np.load(f)
              train_images[i]=(buf)
              i=i+1

i=0
train_labels=np.zeros((10000,1))
with open('obychmetki.npy', 'rb') as f:
       while i<10000:
              buf = np.load(f)
              train_labels[i]=(buf)
              i=i+1
train_labels=np.squeeze(train_labels.astype(int), 1)

i=0

test_images=np.zeros((1000,28,28,1))
with open('provceli.npy', 'rb') as f:
       while i<1000:
              buf = np.load(f)
              test_images[i]=(buf)
              i=i+1

i=0
test_labels=np.zeros((1000,1))
with open('provmetki.npy', 'rb') as f:
       while i<1000:
              buf = np.load(f)
              test_labels[i]=(buf)
              i=i+1
test_labels=np.squeeze(test_labels.astype(int), 1)
  

class_names = ['0', '1', '2', '3']



train_images = train_images / 1400

test_images = test_images / 1400

model=keras.models.load_model('model.h5')

model.summary()

predictions = model.predict(test_images)
print(predictions[0])
print(np.argmax(predictions[0]))

print('prediction: ' + class_names[np.argmax(predictions[0])] + ' really: ' + class_names[test_labels[0]])

plt.figure(figsize=(10,10))

for i in range(50):
    plt.subplot(8,8,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow((test_images[i]))
    plt.grid(False)
    plt.xlabel('с: ' + class_names[np.argmax(predictions[i])] + ' / н: ' + class_names[test_labels[i]])

plt.show()
       
       




