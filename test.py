from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
"""
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels

with open('train-images-idx3-ubyte.gz', 'rb') as f:
    train_images = extract_images(f)

with open('train-labels-idx1-ubyte.gz', 'rb') as f:
    train_labels = extract_labels(f)

with open('t10k-images-idx3-ubyte.gz', 'rb') as f:
    test_images = extract_images(f)

with open('t10k-labels-idx1-ubyte.gz', 'rb') as f:
    test_labels = extract_labels(f)

"""
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

"""




class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
"""
train_images = train_images / 1400

test_images = test_images / 1400

"""
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[test_labels[i]])
plt.show()




"""
model = keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)
#model.fit(train_images, train_labels, epochs=10, steps_per_epoch=100)



test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

model.save('model.h5')

print('все ок')




