#! usr/bin/env python2.7
"""
A script that builds an AO model using Gaussian processes. 

iyhw2, 02/10/2019
"""

import sys
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel as C
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import AO_pair_ham as PH


def AO_gauss_regressor(f):
    """Function that returns the GPR model"""

    f_csv = f + '.csv'
    f_info = f + '_info'

    # read in data
    df = pd.read_csv(f_csv)
    column_count = len(df.columns)-2
    cdf = df.iloc[:,:column_count:]
    dist_list = cdf.values.tolist()
    dist_arr = np.array(dist_list)
    vol_frac_arr = np.array(df.Vol)

    # train/test split
    mask = np.random.rand(len(dist_arr)) < 0.65
    X_train = dist_arr[mask]
    Y_train = vol_frac_arr[mask]
    X_test = dist_arr[~mask]
    Y_test = vol_frac_arr[~mask]

    # reshape if 1D
    if column_count == 1:
        X_train = X_train.reshape(-1, 1)
        Y_train = Y_train.reshape(-1, 1)
        X_test = X_test.reshape(-1, 1)
        Y_test = Y_test.reshape(-1, 1)

    else:
        pass

    # define the kernel - we use RBF
    if column_count == 1:
        # if 1D, we use isotropic kernel
        kernel = RBF(1, (1e-2, 1e3)) + C(noise_level = 1e-5, noise_level_bounds = (1e-10, 1e1))

    else:
        # if higher dimensionality, we use anisotropic kernel
        aniso_kernel = [1]*column_count
        kernel = RBF(aniso_kernel, (1e-2, 1e3)) + C(noise_level = 1e-5, noise_level_bounds = (1e-10, 1e1))

    # build model with training data
    gp = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = 20, normalize_y = False)
    gp.fit(X_train, Y_train)

    # test the model
    Y_pred, std = gp.predict(X_test, return_std = True)

    MSE = ((Y_pred - Y_test)**2).mean()
    
    return gp, X_test, Y_test, Y_pred, std, X_train, Y_train, MSE, column_count


def plotter(f):
    """Plot the data"""

    f_csv = f + '.csv'
    f_info = f + '_info'

    gp, X_test, Y_test, Y_pred, std, X_train, Y_train, MSE, column_count = AO_gauss_regressor(f)

    print gp
    print gp.kernel_
    print 'Mean Squared Error: {}'.format(MSE)
   
    with open(f_info, 'rb') as fp:
        info_list = pickle.load(fp)

    # plot if 2-body interaction
    if column_count == 1:
        # sort to ease plotting
        sorted_X_test, sorted_Y_pred, sorted_std = zip(*sorted(zip(X_test, Y_pred, std)))
        sorted_X_test = list(sorted_X_test)
        sorted_X_test = np.concatenate(sorted_X_test, axis = 0)
        sorted_Y_pred = list(sorted_Y_pred)
        sorted_Y_pred = np.concatenate(sorted_Y_pred, axis = 0)
        sorted_std = np.array(sorted_std)
        pos_error = sorted_Y_pred + sorted_std
        neg_error = sorted_Y_pred - sorted_std

        plt.figure()
        plt.scatter(X_train, Y_train, s = 12)
        plt.scatter(sorted_X_test, sorted_Y_pred, c = 'LightSalmon')
        plt.fill_between(sorted_X_test, neg_error, pos_error, facecolor = 'LightSalmon', alpha = 0.3)
        PH.pairwise_plotter(2*info_list[2], 2*info_list[1])
        plt.xlabel('Inter-colloid distance')
        plt.ylabel('Volume available for polymer')
        plt.show()

    elif column_count == 3:

        X_train_0 = []
        for i in X_train:
            X_train_0.append(i[0])

        X_train_1 = []
        for i in X_train:
            X_train_1.append(i[1])

        X_train_2 = []
        for i in X_train:
            X_train_2.append(i[2])

        plt.figure()
        plt.scatter(X_train_0, Y_train, s = 8, label = 'Short')
        plt.scatter(X_train_1, Y_train, s = 8, label = 'Medium')
        plt.scatter(X_train_2, Y_train, s = 8, label = 'Long')
        plt.xlabel('Inter-colloid Distance')
        plt.ylabel('Overlap Volume')
        plt.legend()
        plt.show()

    else:
        pass
        
    plt.figure()
    plt.scatter(Y_pred, Y_test, s = 12)
    plt.xlabel('Predictions')
    plt.ylabel('Test Data')
    plt.show()
    


if __name__ == '__main__':
    f = sys.argv[1]
    plotter(f)




