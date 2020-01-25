#! usr/bin/env python2.7

import numpy as np
import matplotlib.pyplot as plt

dim = 2

def pairwise_hamiltonian(r, sigma_p, sigma_c):

    q = sigma_p / sigma_c
    R = 0.5*(sigma_p + sigma_c)

    if dim == 3:
        A = (-np.pi/6)*(sigma_p**3)*(((1 + q))**3/(q**3))
        B = 1- (3*r/sigma_c)/(2*(1+q))
        C = ((r/sigma_c)**3)/(2*(1+q)**3)

        return (-(A*(B+C)))

    else:
        A = 2*(R**2)*np.arccos(r/(2*R))
        B = -0.5*r*np.sqrt(4*(R**2) - r**2)

        return A + B


def full_pair_ham(r, sigma_p, sigma_c):

    if r <= sigma_p + sigma_c:
        overlap = pairwise_hamiltonian(r, sigma_p, sigma_c)
        return overlap

    else:
        return 0.0


def pairwise_plotter(sigma_p, sigma_c, plot = 0):

    X = np.linspace(sigma_c, sigma_c + sigma_p, 100)
    plt.plot(X, pairwise_hamiltonian(X, sigma_p, sigma_c), c = 'r')

    if plot:
        plt.show()

    return plt

if __name__ == '__main__':
    pairwise_plotter(sigma_p = 0.5, sigma_c = 0.5, plot = 1)


