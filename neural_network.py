from numpy import matrix
from inputs import zeros, ones, twos

zeros_input = zeros.get_zeros_as_input()
ones_input = ones.get_ones_as_input()
twos_input = twos.get_twos_as_input()


class element_ciagu_treningowego:
    X: matrix
    D: matrix


ciag_treningowy = []

for zero in zeros_input:
    X = matrix(zero)
    D = matrix([[1], [0], [0]])
