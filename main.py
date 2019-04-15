import tensorflow as tf
from data_helper import DataHelper

tf.reset_default_graph()

# Data
image_dim = 256
helper = DataHelper(image_dim=256)

total_size = helper.size
test_size = 400
train_size = total_size - test_size

dataset = helper.get_dataset()
train_dataset = dataset.take(train_size)

# Variables
layer_shapes = [500, 500, 500, len(helper.categories)]
activation = tf.nn.leaky_relu
learning_rate = 1
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
metrics = ['categorical_accuracy']
epochs = 10
steps_per_epoch = 10
loss = 'mse'


# Initialization
model = tf.keras.Sequential()

model.add(tf.keras.layers.Flatten(input_shape=[image_dim*image_dim*3]))

for i in range(len(layer_shapes)):
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
model.fit(
    train_dataset.batch(1),
    epochs=epochs,
    steps_per_epoch=steps_per_epoch
)
