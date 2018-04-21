from numpy import matrix, vectorize
from numpy.random import random
from inputs import zeros, ones, twos

zeros_input = zeros.get_zeros_as_input()
ones_input = ones.get_ones_as_input()
twos_input = twos.get_twos_as_input()


class wektor_uczacy:
    X: matrix
    D: matrix

    def __init__(self, X, D):
        self.X = X
        self.D = D


ciag_treningowy = []

for zero in zeros_input:
    X = matrix(zero)
    D = matrix([[1], [0], [0]])
    ciag_treningowy.append(wektor_uczacy(X, D))

for one in ones_input:
    X = matrix(one)
    D = matrix([[0], [1], [0]])
    ciag_treningowy.append(wektor_uczacy(X, D))

for two in ones_input:
    X = matrix(two)
    D = matrix([[0], [0], [1]])
    ciag_treningowy.append(wektor_uczacy(X, D))

liczba_neuronow = 3
liczba_wejsc = 24

macierz_wag = matrix(-2 * random((liczba_neuronow, liczba_wejsc)) + 1)
wektor_odchylen = matrix(-2 * random(liczba_neuronow) + 1).transpose()
wektor_odchylen = matrix(wektor_odchylen)

print(wektor_odchylen)

alpha = 0.5
E_MIN = 0.5
MAX_EPOK = 15

ile_epok = 0

unipolarna_funkcja_aktywacji = vectorize(lambda n: 0 if n < 0 else 1)
aktualizacja_wag = vectorize(lambda w, X, alpha, d, y: w + alpha * (d - y) * X)
aktualizacja_odchylen = vectorize(lambda b, alpha, d, y: b + alpha * (d - y))


def ucz(ciag_uczacy: [wektor_uczacy], macierz_wag, wektor_odchylen, E, E_MIN, ile_epok, MAX_EPOK):
    nowe_wagi = macierz_wag
    for wektor in ciag_uczacy:
        X = wektor.X
        D = wektor.D
        nowe_wagi = macierz_wag
        Y = unipolarna_funkcja_aktywacji(matrix(macierz_wag) * matrix(X) + matrix(wektor_odchylen))
        for i in enumerate(nowe_wagi):
            Wi = nowe_wagi[i]
            bi = wektor_odchylen[i]
            d = D[i]
            y = Y[i]
            aktualizacja_wag(Wi, X, alpha, d, y)
            aktualizacja_odchylen(bi, alpha, d, y)
            Y = unipolarna_funkcja_aktywacji(matrix(macierz_wag) * matrix(wektor) + matrix(wektor_odchylen))
            y = Y[i]
            E += (d - y) ** 2
    E *= 0.5
    if (nowe_wagi == macierz_wag or E < E_MIN or ile_epok >= MAX_EPOK):
        return {"wagi": nowe_wagi, "wektor_odchylen": wektor_odchylen,
                "info": "Trening trwał {0} epok.".format(ile_epok)}
    else:
        return ucz(ciag_uczacy, nowe_wagi, wektor_odchylen, E, E_MIN, ile_epok, MAX_EPOK)


siec = ucz(ciag_treningowy,macierz_wag,wektor_odchylen,0,E_MIN,0,MAX_EPOK)
