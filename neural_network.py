from numpy import matrix

class element_ciagu_treningowego:
    X: matrix
    D: matrix

ciag_treningowy = []

# zero = matrix([[0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0]]).transpose()

from inputs import zeros as z

zeros = z.get_zeros_as_input()

# zeros.append([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0]) #