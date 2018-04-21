from numpy.random import random
from numpy import matrix
from numpy import vectorize
import numpy


class Training_Set:
    feature_vector: matrix
    label: matrix

    def __init__(self, feature_vector: [], label: []):
        self.feature_vector = matrix(feature_vector)
        self.label = matrix(label)


class Neural_Network_Signle_Layer:
    epocs_passed: int = 0

    E_MIN: float = 0.2
    max_epocs: int = 1

    number_of_neurons: int
    number_of_inputs: int
    weights_matrix: matrix
    bias_vector: matrix
    activation_function = ()

    alpha = 0.5

    def __init__(self, number_of_neurons, number_of_inputs, activation_function):
        self.number_of_neurons = number_of_neurons
        self.number_of_inputs = number_of_inputs
        self.weights_matrix = matrix(-2 * random((number_of_neurons, number_of_inputs)) + 1)
        self.bias_vector = matrix(-2 * random(number_of_neurons) + 1)
        self.activation_function = activation_function

    def get_output(self, input: matrix):
        NET = self.NET(input)
        vfunc = vectorize(self.activation_function)
        output = vfunc(NET)
        return matrix(output)

    def NET(self, input: matrix) -> matrix:
        NET = self.weights_matrix * input
        NET += self.bias_vector
        return NET

    def binary_activation_function(self, x: float) -> int:
        if (x < 0):
            return 0
        else:
            return 1

    def learn(self, trainig_input: matrix) -> int:
        E = 0
        weight_didnt_changed = True
        for i in enumerate(trainig_input):
            trainig_set: Training_Set = trainig_input.A[0][0]
            X = trainig_set.feature_vector
            D = trainig_set.label
            Y = self.get_output(X)
            for i in range(self.number_of_neurons):
                if (Y != D):
                    weight_didnt_changed = False
                    d = D[i]
                    y = Y[i]
                    W = self.weights_matrix[i]
                    W = W + self.alpha * (d - y) * X
                    self.weights_matrix[i] = W
            E += self.calculate_partial_error(Y, D)
        self.epocs_passed += 1
        E *= E
        if (E < self.E_MIN or weight_didnt_changed or self.epocs_passed >= self.max_epocs):
            return self.epocs_passed
        else:
            return self.learn(trainig_input)

    def calculate_partial_error(self, Y: matrix, d_vector: matrix) -> float:
        sum = 0
        for i in range(self.number_of_neurons):
            d = d_vector[i]
            y = Y[i]
            sum += pow(d - y)
        return sum


X1 = Training_Set([[-1], [0], [3]], [-2])
X2 = Training_Set([[1], [0], [-2]], [1])
X3 = Training_Set([[-3], [-2], [-1]], [1])

neuron = Neural_Network_Signle_Layer(1, 3, Neural_Network_Signle_Layer.binary_activation_function)
neuron.weights_matrix = matrix([-1, 2, 1])
neuron.alpha = 0.5

neuron.learn(matrix([X1,X2,X3]))

