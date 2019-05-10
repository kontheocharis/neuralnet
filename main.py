import tensorflow as tf
from tensorflow import keras
# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image

tf.reset_default_graph()

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = \
    fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

# IMAGE GENERATION
# indices = [3, 40, 600, 800]
# class_names = ['Tshirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']

# for i in indices:
#     image = train_images[i]
#     image = np.array(image, dtype='float')
#     pixels = image.reshape((28, 28))
#     im = Image.fromarray(pixels).convert("L")
#     im.save("images/" + str(class_names[train_labels[i]]) + str(i) + ".jpeg")

train_labels = tf.one_hot(train_labels, 10)
test_labels = tf.one_hot(test_labels, 10)

model = keras.Sequential([
    keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
    keras.layers.Conv2D(8, (3, 3), activation=tf.nn.relu),
    keras.layers.Flatten(input_shape=()),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=20, steps_per_epoch=10)

test_loss, test_acc = model.evaluate(test_images, test_labels, steps=10)

print('Test accuracy:', test_acc)
