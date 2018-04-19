from numpy.random import random
from numpy import matrix
import numpy


class Neural_Network:  # one layer

    number_of_neurons: int
    weights_matrix: matrix
    bias_vector: matrix
    activation_function: function

    def __init__(self, number_of_neurons, number_of_inputs, activation_function):
        self.number_of_neurons = number_of_neurons
        self.weights_matrix = matrix(-2 * random((number_of_neurons, number_of_inputs)) + 1)
        self.bias_vector = matrix(-2 * random(number_of_neurons) + 1)

    def NET(self, input: matrix) -> matrix:
        output = self.weights_matrix * input
        output += self.bias_vector
        return output

    def binary_activation_function(self, NET: float) -> int:
        if (NET < 0):
            return 0
        else:
            return 1
