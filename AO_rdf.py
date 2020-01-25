#! /usr/bin/env python2.7

import numpy as np
import matplotlib.pyplot as plt
import pickle
import AO_model_MC
import sys

L = 41
dim = 2

f_info = sys.argv[1]

with open(f_info, 'rb') as fp:
    colloid_centres = pickle.load(fp)


def get_all_distances(col_list_list):

    dist_list = []

    for col_list in col_list_list:
        for i in col_list:
            for j in col_list:
                x, y, z = AO_model_MC.vec_difference(i, j, L)
                dist = AO_model_MC.pythagoras(x, y, z)

                if dist != 0.0:
                    dist_list.append(dist)
            
                else:
                    pass

    col_list_list_len = len(col_list_list)
    col_list_len = len(col_list_list[0])

    return dist_list, col_list_len, col_list_list_len


def binner(num, col_list, col_list_len, col_list_list_len):

    print 'Colloid Count: {}'.format(col_list_len)
    print 'Colloid Configurations: {}'.format(col_list_list_len)
    X = np.linspace(0, 0.4*L, num)
    X_len = len(X)

    if dim == 2:
        ideal_density = float(col_list_len)/L**2 

    axis_list = []
    bin_density_list = []

    for i in range(0, X_len-1):
        axis_point = X[i]
        bin_count = 0
        for j in col_list:
            if j >= X[i] and j < X[i+1]:
                bin_count += 1

        if dim == 2:
            shell_vol = (np.pi)*(X[i+1])**2 - (np.pi)*(X[i])**2

        bin_density = float(bin_count)/(col_list_len*col_list_list_len*shell_vol*ideal_density)

        axis_list.append(axis_point)
        bin_density_list.append(bin_density)

    plt.scatter(axis_list, bin_density_list, s= 10)
    plt.xlabel('Distance')
    plt.ylabel('Density')
    plt.show()



if __name__ == "__main__":
    col_list, col_list_len, col_list_list_len = get_all_distances(colloid_centres)
    binner(70, col_list, col_list_len, col_list_list_len)    





















        







