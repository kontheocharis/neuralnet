import numpy as np
import math


class FeedForward:
    def __init__(self, shape, learning_rate, activation_func):
        self.shape = shape
        self.learning_rate = learning_rate
        self.activation_func = activation_func

        self.__randomize_weights()
        self.__generate_biases()

    def __shape_range(self):
        return range(1, len(self.shape))

    def __randomize_weights(self):
        self.weights = []
        for i in self.__shape_range():
            rows = self.shape[i]
            cols = self.shape[i-1]

            # arbitrary but common
            bound = 1 / math.sqrt(float(cols))

            self.weights[i] = np.random.uniform(-bound, bound, (rows, cols))

    def __generate_biases(self):
        self.biases = [
            np.zeros(self.shape[i]) for i in self.__shape_range()
        ]

    def train_once(self, inputs, targets):
        layers = [np.array(inputs, dtype=float)]

        for i in self.__shape_range():
            layers[i] = np.dot(self.weights[i], self.layers[i-1])
            layers[i] = np.add(layers[i], self.biases[i])
            layers[i] = np.vectorize(self.activation_func)(layers[i])

        net_output = layers[-1]
        target_output = np.array(targets, dtype=float)

        output_error = np.square(np.subtract(net_output, target_output))

        self.backpropagate(output_error)

    def backpropagate(error):
        pass
