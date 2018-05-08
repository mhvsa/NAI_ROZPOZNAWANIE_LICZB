from numpy import matrix, vectorize, array_equal
from numpy.random import random
from inputs import zeros, ones, twos
import matplotlib.pyplot as plt

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

for zero in zeros_input[0:][::1]:
    X = matrix(zero)
    D = matrix([[1], [0], [0]])
    ciag_treningowy.append(wektor_uczacy(X, D))

for one in ones_input[0:][::2]:
    X = matrix(one)
    D = matrix([[0], [1], [0]])
    ciag_treningowy.append(wektor_uczacy(X, D))

for two in twos_input[0:][::2]:
    X = matrix(two)
    D = matrix([[0], [0], [1]])
    ciag_treningowy.append(wektor_uczacy(X, D))
print(ciag_treningowy.__len__())
liczba_neuronow = 3
liczba_wejsc = 24

macierz_wag = matrix(-2 * random((liczba_neuronow, liczba_wejsc)) + 1)
wektor_odchylen = matrix(-2 * random(liczba_neuronow) + 1).transpose()
wektor_odchylen = matrix(wektor_odchylen)

# print(macierz_wag)
# print(wektor_odchylen)

alpha = 0.05
E_MIN = 0.01
MAX_EPOK = 100

ile_epok = 0

unipolarna_funkcja_aktywacji = vectorize(lambda n: 0 if n < 0 else 1)
aktualizacja_wag = vectorize(lambda w, X, alpha, d, y: w + (alpha * (d - y) * X))
aktualizacja_odchylen = vectorize(lambda b, alpha, d, y: b + alpha * (d - y))


def ucz(ciag_uczacy: [wektor_uczacy], macierz_wag, wektor_odchylen, E_MIN, ile_epok, MAX_EPOK, E_VALUES: []):
    E: float = 0.0
    nowe_wagi = matrix(macierz_wag)
    for wektor in ciag_uczacy:
        X = wektor.X
        D = wektor.D
        Y = unipolarna_funkcja_aktywacji(matrix(nowe_wagi) * matrix(X) + matrix(wektor_odchylen))
        i = 0  # tymczasowo
        if (not array_equal(D, Y)):
            for W in nowe_wagi:
                Wi = nowe_wagi[i]
                bi = wektor_odchylen[i]
                d: int = D.item(i)
                y: int = Y.item(i)
                Xt = X.transpose()
                Wi = aktualizacja_wag(Wi, Xt, alpha, d, y)
                bi = aktualizacja_odchylen(bi, alpha, d, y)
                nowe_wagi[i] = Wi
                wektor_odchylen[i] = bi
                Y = unipolarna_funkcja_aktywacji(matrix(nowe_wagi) * matrix(X) + matrix(wektor_odchylen))
                # y = Y.item(i)
                E = E + (d - y) ** 2
                i += 1  # tymczasowo
    E *= 0.5
    ile_epok += 1
    E_VALUES.append(E)
    brak_zmiany_wag = array_equal(nowe_wagi, macierz_wag)
    if (brak_zmiany_wag or E < E_MIN or ile_epok >= MAX_EPOK):
        print("brak zmiany wag: {0}, E = {1}, ile epok = {2}".format(brak_zmiany_wag, E, ile_epok))
        return {"wagi": nowe_wagi, "wektor_odchylen": wektor_odchylen,
                "info": "Trening trwał {0} epok. Blad wynosi {1}".format(ile_epok, E), "epoki": ile_epok, "blad": E}
    else:
        return ucz(ciag_uczacy, nowe_wagi, wektor_odchylen, E_MIN, ile_epok, MAX_EPOK,E_VALUES)


# siec = ucz(ciag_treningowy, macierz_wag, wektor_odchylen, E_MIN, 0, MAX_EPOK)
# print(siec["info"])
# print(siec["wagi"])
# print(siec["wektor_odchylen"])
#
# wagi = siec["wagi"]
# bias = siec["wektor_odchylen"]
# blad = siec["blad"];

bledne_testy = 100
blad = 100

while (blad > 1 or bledne_testy > 5):
    bledne_testy = 0
    macierz_wag = matrix(-2 * random((liczba_neuronow, liczba_wejsc)) + 1)
    wektor_odchylen = matrix(-2 * random(liczba_neuronow) + 1).transpose()
    wektor_odchylen = matrix(wektor_odchylen)
    E_VALUES = []
    siec = ucz(ciag_treningowy, macierz_wag, wektor_odchylen, E_MIN, 0, MAX_EPOK,E_VALUES)
    blad = siec["blad"]
    if(blad > 1): continue
    print("==== Testy dla zer ====")

    d0 = matrix([[1], [0], [0]])

    testy_zero = zeros_input[1:][::2]

    for wektor in testy_zero:
        Y = unipolarna_funkcja_aktywacji(matrix(siec["wagi"]) * matrix(wektor) + matrix(siec["wektor_odchylen"]))
        wynik_testu = array_equal(d0,Y)
        if (wynik_testu == False):
            bledne_testy += 1
        print(array_equal(d0, Y))

    print("==== Testy dla jedynek ====")

    testy_ones = ones_input[1:][::2]

    d1 = matrix([[0], [1], [0]])

    for wektor in testy_ones:
        Y = unipolarna_funkcja_aktywacji(matrix(siec["wagi"]) * matrix(wektor) + matrix(siec["wektor_odchylen"]))
        wynik_testu = array_equal(d1,Y)
        if (wynik_testu == False):
            bledne_testy += 1
        print(array_equal(d1, Y))

    print("=== Testy dla dwojek ===")

    testy_twos = twos_input[1:][::2]

    d2 = matrix([[0], [0], [1]])

    for wektor in testy_twos:
        Y = unipolarna_funkcja_aktywacji(matrix(siec["wagi"]) * matrix(wektor) + matrix(siec["wektor_odchylen"]))
        wynik_testu = array_equal(d2, Y)
        if (wynik_testu == False):
            bledne_testy += 1
        print(wynik_testu)

# print("==== Testy dla zer ====")
#
# d0 = matrix([[1], [0], [0]])
#
# testy_zero = zeros_input[1:][::2]
#
# for wektor in testy_zero:
#     Y = unipolarna_funkcja_aktywacji(matrix(siec["wagi"]) * matrix(wektor) + matrix(siec["wektor_odchylen"]))
#     print(array_equal(d0, Y))
#
# print("==== Testy dla jedynek ====")
#
# testy_ones = ones_input[1:][::2]
#
# d1 = matrix([[0], [1], [0]])
#
# for wektor in testy_ones:
#     Y = unipolarna_funkcja_aktywacji(matrix(siec["wagi"]) * matrix(wektor) + matrix(siec["wektor_odchylen"]))
#     print(array_equal(d1, Y))
#
# print("=== Testy dla dwojek ===")
#
# testy_twos = twos_input[1:][::2]
#
# d2 = matrix([[0], [0], [1]])
#
# for wektor in testy_twos:
#     Y = unipolarna_funkcja_aktywacji(matrix(siec["wagi"]) * matrix(wektor) + matrix(siec["wektor_odchylen"]))
#     wynik_testu = array_equal(d2, Y)
#     if(wynik_testu == False):
#         bledne_testy += 1
#     print(wynik_testu)

plt.plot(E_VALUES)
plt.ylabel("E")
plt.xlabel("epocs")
plt.text(5,5,"alpha = {0}".format(alpha))
plt.show()


def getAnswer(input: matrix):
    Y:matrix = unipolarna_funkcja_aktywacji(matrix(siec["wagi"]) * matrix(input) + matrix(siec["wektor_odchylen"]))
    # print(Y)
    if(array_equal(Y,d0)):
        print("To jest zero!")
    if(array_equal(Y,d1)):
        print("To jest jeden!")
    if(array_equal(Y,d2)):
        print("To jest dwa!")