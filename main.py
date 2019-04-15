import tensorflow as tf
from data_helper import DataHelper

tf.reset_default_graph()

# Data
image_dim = 256
helper = DataHelper(image_dim=256)

total_size = helper.size
# test_size = 400
# train_size = total_size - test_size

dataset = helper.get_dataset()

print(dataset.output_shapes)
print(dataset.output_types)

# it = dataset.batch(1).make_one_shot_iterator()
# with tf.Session() as sess:
#     items = []
#     for i in range(10):
#         items.append(it.get_next())
#     print(sess.run(items))    

# Variables
activation = tf.nn.leaky_relu
learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate)
metrics = ['categorical_accuracy']
epochs = 10
steps_per_epoch = 30
loss = 'mse'


# Initialization
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),activation='relu', input_shape=(image_dim**2*3,)))
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(helper.categories), activation='softmax'))

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
    dataset.batch(100),
    epochs=epochs,
    steps_per_epoch=steps_per_epoch
)
