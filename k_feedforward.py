from typing import List, Callable
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

tf.reset_default_graph()

# Data
mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
dataset = tf.data.Dataset.from_tensor_slices((
    mnist.train.images.astype(np.float32),
    mnist.train.labels.astype(np.float32)
)).batch(100).repeat()

# Variables
layer_shapes = [500, 500, 500, 10]
activation = tf.nn.leaky_relu
learning_rate = 1
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
metrics = ['categorical_accuracy']
epochs = 10
steps_per_epoch = 1000


# Initialization
model = tf.keras.Sequential()

for i in range(len(layer_number)):
    model.add(tf.keras.layers.Dense(
        layer_shapes[i],
        activation=activation
    ))


# Compiling
print('Compiling model...')

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics
)

print('Model compiled.')
print('Starting training for', epochs, 'epochs...')

# Fitting
model.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch)
net.model.evaluate(mnist.test.images, mnist.test.labels)
