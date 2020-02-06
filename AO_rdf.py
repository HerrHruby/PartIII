#! /usr/bin/env python2.7

import numpy as np
import matplotlib.pyplot as plt
import pickle
import AO_model_MC
import sys
import time


L = 41
dim = 2

def get_all_distances(col_configs):

    dist_list = []

    for col_list in col_configs:
        for i in col_list:
            for j in col_list:
                x, y, z = AO_model_MC.cyclic_vec_difference(i, j, L)
                dist = AO_model_MC.pythagoras(x, y, z)

                if dist != 0.0:
                    dist_list.append(dist)
            
                else:
                    pass

    col_configs_len = len(col_configs)
    col_list_len = len(col_configs[0])

    return dist_list, col_list_len, col_configs_len


def binner(num, plotrange, col_configs):

    dist_list, col_list_len, col_configs_len = get_all_distances(col_configs)
    
    print 'Colloid Count: {}'.format(col_list_len)
    print 'Colloid Configurations: {}'.format(col_configs_len)
    X = np.linspace(2, plotrange*L, num)
    X_len = len(X)

    if dim == 2:
        ideal_density = float(col_list_len)/L**2 

    axis_list = []
    bin_density_list = []

    for i in range(0, X_len-1):
        axis_point = X[i]
        bin_count = 0
        for j in dist_list:
            if j >= X[i] and j < X[i+1]:
                bin_count += 1

        if dim == 2:
            shell_vol = (np.pi)*(X[i+1])**2 - (np.pi)*(X[i])**2

        bin_density = float(bin_count)/(col_list_len*col_configs_len*shell_vol*ideal_density)

        axis_list.append(axis_point)
        bin_density_list.append(bin_density)

    return axis_list, bin_density_list


def plotter(num, plotrange, all_cols):

    trials = len(all_cols)

    axis_list, bin_density_list = binner(num, plotrange, all_cols[0])

    tic = time.time()
    col_dict = {i: [] for i in axis_list}
    for i, j in zip(axis_list, bin_density_list):
        col_dict[i].append(j)
    toc = time.time()

    for i in range(1, trials):
        tic = time.time()
        axis_list, bin_density_list = binner(num, plotrange, all_cols[i])
        for j, k in zip(axis_list, bin_density_list):
            col_dict[j].append(k)
        toc = time.time()

    avg_list = []
    std_list = []
    axis_list = []
    for i, j in col_dict.iteritems():
        std_list.append(np.std(j))
        avg_list.append(np.mean(j))
        axis_list.append(i)

    axis_list, avg_list, std_list = zip(*sorted(zip(axis_list, avg_list, std_list)))

    lower_bound = [i - j for i, j in zip(avg_list, std_list)]
    upper_bound = [i + j for i, j in zip(avg_list, std_list)]

    plt.scatter(axis_list, avg_list, s= 10)
    plt.fill_between(axis_list, lower_bound, upper_bound, alpha = 0.3)
    plt.xlabel('Distance')
    plt.ylabel('Density')
    plt.show()



if __name__ == "__main__":

    f_info = sys.argv[1]
    
    with open(f_info, 'rb') as fp:
        colloid_centres = pickle.load(fp)

    plotter(60, 0.3, colloid_centres)




















        







