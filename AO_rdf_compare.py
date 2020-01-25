#! /usr/bin/env python2.7

import numpy as np
import matplotlib.pyplot as plt
import pickle
import AO_model_MC
import sys
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel as C
from sklearn.model_selection import train_test_split


L = 41
dim = 2

great_list = []

for i in range(1,5):
    with open(sys.argv[i], 'rb') as fp:
        colloid_centres = pickle.load(fp)
        great_list.append(colloid_centres)

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
    X = np.linspace(0, 0.3*L, num)
    X_len = len(X)

    if dim == 2:
        ideal_density = float(col_list_len)/L**2 

    midpoint_list = []
    bin_density_list = []

    for i in range(0, X_len-1):
        midpoint = X[i]
        bin_count = 0
        for j in col_list:
            if j >= X[i] and j < X[i+1]:
                bin_count += 1

        if dim == 2:
            shell_vol = (np.pi)*(X[i+1])**2 - (np.pi)*(X[i])**2

        bin_density = float(bin_count)/(col_list_len*col_list_list_len*shell_vol*ideal_density)

        midpoint_list.append(midpoint)
        bin_density_list.append(bin_density)


    return midpoint_list, bin_density_list


def plotter(plotting_mode = 0):

    col_dict = {}

    if not plotting_mode:

        for index, i in enumerate(great_list):
            dict_list = []
            col_list, col_list_len, col_list_list_len = get_all_distances(i)
            mid_list, bin_list = binner(50, col_list, col_list_len, col_list_list_len)  
            dict_list.append(mid_list)
            dict_list.append(bin_list)

            col_dict[index] = dict_list

        for index, i in enumerate(col_dict.itervalues()):
            plt.scatter(i[0], i[1], s = 10, label = sys.argv[index+1])

    if plotting_mode:

        for index, i in enumerate(great_list):
            X = np.linspace(0, L*0.3, 300)
            col_list, col_list_len, col_list_list_len = get_all_distances(i)
            mid_list, bin_list = binner(80, col_list, col_list_len, col_list_list_len)          
            X_train, X_test, Y_train, Y_test = train_test_split(np.array(mid_list), np.array(bin_list), test_size = 0.2)

            X_train = X_train.reshape(-1,1)
            X_test = X_test.reshape(-1,1)
            X = X.reshape(-1,1)
            kernel = RBF(1, (1e-2, 1e3)) + C(noise_level = 1e-5, noise_level_bounds = (1e-10, 1e1))

            gp = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = 20, normalize_y = False)
            gp.fit(X_train, Y_train)

            Y_pred, std = gp.predict(X_test, return_std = True)
            MSE = ((Y_pred - Y_test)**2).mean()

            print 'RMSE:{}'.format(np.sqrt(MSE))

            plot_pred = gp.predict(X, return_std = False)

            col_dict[index] = plot_pred

        for index, i in enumerate(col_dict.itervalues()):
            plt.plot(X, i, label = sys.argv[index+1])

    plt.xlabel('Distance')
    plt.ylabel('Density')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    plotter(plotting_mode = 0)




















        







