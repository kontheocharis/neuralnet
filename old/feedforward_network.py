from typing import List, Callable
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class FeedForwardLayer:
    def __init__(self, weights: tf.Variable, biases: tf.Variable):
        weights.shape.assert_has_rank(2)
        biases.shape.assert_has_rank(1)
        assert weights.shape[1] == biases.shape[0]
        self.weights = weights
        self.biases = biases


CostFunction = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]

ActivationFunction = Callable[[tf.Tensor, str], tf.Tensor]


class FeedForwardNetwork:
    is_initialized: bool = False
    def __init__(self, layer_shapes: List[int]) -> None:
        # Note: layer_shapes includes x layer
        self.layer_shapes = layer_shapes

        self.layer_number: int = len(layer_shapes) - 1

        self.x_size = layer_shapes[0]
        self.y_size = layer_shapes[-1]

    def init_parameters(self) -> None:
        self.layers: List[FeedForwardLayer] = \
            [None for _ in range(self.layer_number)]

        print('Initializing parameters...')

        for i in range(self.layer_number):
            prev_shape = self.layer_shapes[i]
            curr_shape = self.layer_shapes[i + 1]

            weights = tf.Variable(tf.random_normal([prev_shape, curr_shape]))
            biases = tf.Variable(tf.random_normal([curr_shape]))

            self.layers[i] = FeedForwardLayer(weights, biases)

        print('Parameters initialized.')
        self.is_initialized = True

    def model(self, x: tf.Variable) -> tf.Variable:
        current_layer: tf.Variable = x

        for i in range(self.layer_number):
            current_layer = self.activation_function(tf.add(
                tf.tensordot(tf.transpose(self.layers[i].weights), current_layer, 1),
                self.layers[i].biases
            ))

        # current layer now is predicted y
        return current_layer

    def train(self,
              x_data: tf.Tensor,
              y_data: tf.Tensor,
              example_number: int,
              epochs: int,
              batch_size: int,
              activation_function: ActivationFunction,
              cost_function: CostFunction,
              optimizer_function: tf.train.Optimizer,
              ) -> None:

        assert x_data.shape[0] == y_data.shape[0] == example_number

        x_dataset = tf.data.Dataset.from_tensor_slices(x_data)
        y_dataset = tf.data.Dataset.from_tensor_slices(y_data)

        assert x_dataset.output_shapes == self.x_size
        assert y_dataset.output_shapes == self.y_size
        assert self.is_initialized

        self.activation_function = activation_function

        batched_x, batched_y = (i.batch(batch_size)
                                for i in (x_dataset, y_dataset))

        x_batch, y_batch = (
            tf.placeholder('float', shape=(batch_size, self.x_size)),
            tf.placeholder('float', shape=(batch_size, self.y_size)))

        pred_y_batch = tf.map_fn(self.model, x_batch)
        cost = tf.reduce_mean(cost_function(pred_y_batch, y_batch))
        optimizer = optimizer_function.minimize(cost)

        print('Training...')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(epochs):
                epoch_loss = 0
                x_it, y_it = (b.make_one_shot_iterator()
                              for b in (batched_x, batched_y))

                curr_x_init = x_it.get_next()
                curr_y_init = y_it.get_next()

                while True:
                    try:
                        curr_x = sess.run(curr_x_init)
                        curr_y = sess.run(curr_y_init)
                        _, c = sess.run([optimizer, cost], feed_dict={
                            x_batch: curr_x, y_batch: curr_y
                        })
                        epoch_loss += c
                    except tf.errors.OutOfRangeError:
                        break
                print('Epoch', epoch + 1, 'out of', epochs, 'completed.',
                      'Current loss:', epoch_loss)
            self.test_accuracy(sess, x_data, y_data, example_number)

    def test_accuracy(self,
                      sess: tf.Session,
                      x_data: tf.Tensor,
                      y_data: tf.Tensor,
                      example_number: int
                      ):
        print('Testing accuracy...')
        pred_y = tf.map_fn(self.model, x_data)
        correct = tf.equal(tf.argmax(pred_y, 1), tf.argmax(y_data, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print("Accuracy:", accuracy.eval(session=sess))


mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

net = FeedForwardNetwork([784, 500, 500, 500, 10])

net.init_parameters()
net.train(
    mnist.train.images,
    mnist.train.labels,
    mnist.train.num_examples,
    10, 100,
    tf.nn.relu,
    lambda pred_y, y: tf.nn.softmax_cross_entropy_with_logits_v2(
        logits = pred_y,
        labels = y
    ),
    tf.train.AdamOptimizer(learning_rate=0.001)
)
