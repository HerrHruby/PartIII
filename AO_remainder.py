#! /usr/bin/env/ python2.7
"""
A script that builds a model for the n-body (n > 2) interaction

iyhw2, 06/10/2019
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel as C
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression
import AO_GP
import AO_model_MC
import AO_learning_curves
import AO_pair_ham
import random

model = 0 # = 0 for analytical AO, =1 for ML-2body

def AO_3body_remainder(f, plot = 0, shuffle = 0):

    f_csv = f + '.csv'
    f_info = f + '_info'
    f_csv_3 = f + '_3.csv'
    f_info_3 = f + '_3_info'

    with open(f_info, 'rb') as fp:
        info_list = pickle.load(fp)

    with open(f_info_3, 'rb') as fp:
        info_list_3 = pickle.load(fp)

    # read in 3-body data
    df = pd.read_csv(f_csv_3)
    column_count = len(df.columns) -2
    cdf = df.iloc[:,:column_count:]
    dist_list = cdf.values.tolist()
    dist_arr = np.array(dist_list)
    vol_arr = np.array(df.Vol)

    if model == 1:
        # return a GPR model for the two-body interaction
        gp_pair, X_test_pair, Y_test_pair, Y_pred_pair, std_pair, X_train_pair, Y_train_pair, MSE_pair, column_count_pair = AO_GP.AO_gauss_regressor(f)


    pair_sum_list = [] # sum of pair potentials
    pair_indiv_list = [] # individual predictions
    indiv_dist_list = [] # individual pairwise distances

    # shuffles the distance list, removing sorting
    if shuffle:
        for i in dist_arr:
            random.shuffle(i)

    # for each config, loop through the distances and sum the predicted two-body interactions. Also return the individual distances and individual pair predictions (for checking)
    for i in dist_arr:
        pair_sum = 0

        for j in i:
            indiv_dist_list.append(j)

            if model == 1:
                k = j.reshape(1, -1)
                pair_predict = gp_pair.predict(k)
                pair_sum += pair_predict[0]
                pair_indiv_list.append(pair_predict[0])
       
            else:
                pair_predict = AO_pair_ham.full_pair_ham(j, 2*info_list_3[2], 2*info_list_3[1])
                pair_sum += pair_predict
                pair_indiv_list.append(pair_predict)

        pair_sum_list.append(pair_sum)

    # summed distances (for checking)
    sumlist = []
    for i in dist_arr:
        summed = 0
        for j in i:
            summed += j
        sumlist.append(summed)

    if plot == 1:
        # plot of pairwise potentials against pairwise individual distances (for checking)
        plt.scatter(indiv_dist_list, pair_indiv_list, s = 8)
        plt.xlabel('Pairwise Distance')
        plt.ylabel('Overlap volume')
        plt.show()

        # plot of summed distance against sum of pairwise overlap 
        plt.scatter(sumlist, pair_sum_list, s = 8)
        plt.xlabel('Total Intercolloid Distance')
        plt.ylabel('Sum of two-particle overlaps')
        plt.show()

        reg = LinearRegression()
        reg.fit(pair_sum_list, vol_arr)
        print 'R2 Score: {}'.format(reg.score(pair_sum_list, vol_arr))
        coefs = reg.coef_
        intercept = reg.intercept_
        print 'Coefficients: {}'.format(coefs)
        print 'Intercept: {}'.format(intercept)

        # plot of summed two-particle overlaps against total n-body overlapping volume
        X_ = np.linspace(min(pair_sum_list), max(pair_sum_list), 100) 
        plt.scatter(pair_sum_list, vol_arr, s = 8)
        plt.plot(X_, X_*coefs[0] + intercept)
        plt.xlabel('Sum of two-particle overlaps')
        plt.ylabel('Total n-body overlapping volume')
        plt.show()

    else:
        pass

    # clean up arrays
    pair_sum_list = list(pair_sum_list)

    if model == 1:
        pair_sum_list = np.concatenate(pair_sum_list, axis = 0)
        pair_sum_list = pair_sum_list.flatten()

    # find the difference between the total interaction and the two-body interaction 
    if model == 1:
        n_body_list = -0.5*vol_arr + 0.5*pair_sum_list

    else:
        n_body_list = [-0.5*i + 0.5*j for i, j in zip(vol_arr, pair_sum_list)]

    if plot == 1:
        # plot of summed distance against 3-body volume
        plt.scatter(sumlist, n_body_list, s = 8)
        plt.xlabel('Total Intercolloid Distance')
        plt.ylabel('3-body interaction')
        plt.show()

        print vol_arr
        print pair_sum_list

        print 'n_body_list: {}'.format(n_body_list)

    #train-test split
    X_train_n, X_test_n, Y_train_n, Y_test_n = train_test_split(dist_arr, n_body_list, test_size = 0.1)

    # set up GPR
    aniso_kernel = [1]*column_count
    kernel = RBF(aniso_kernel, (1e-2, 1e3)) + C(noise_level = 1e-3, noise_level_bounds = (1e-10, 1e1))

    gp_n_interact = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = 20, normalize_y = False)

    # fit data
    gp_n_interact.fit(X_train_n, Y_train_n)

    # predict test data
    Y_pred, Y_std = gp_n_interact.predict(X_test_n, return_std = True)

    print gp_n_interact.kernel_

    MSE = ((Y_pred - Y_test_n)**2).mean()
    print 'Mean Squared Error: {}'.format(MSE)

    return gp_n_interact, X_train_n, X_test_n, Y_train_n, Y_test_n, dist_arr, n_body_list, Y_pred, Y_std


def plot_learning(f, learning = 0, shuffle = 0):

    f_csv = f + '.csv'
    f_info = f + '_info'
    f_csv_3 = f + '_3.csv'
    f_info_3 = f + '_3_info'

    with open(f_info, 'rb') as fp:
        info_list = pickle.load(fp)

    with open(f_info_3, 'rb') as fp:
        info_list_3 = pickle.load(fp)
        
    gp_n_interact, X_train_n, X_test_n, Y_train_n, Y_test_n, dist_arr, n_body_list, Y_pred, Y_std = AO_3body_remainder(f, plot = 0, shuffle = shuffle)

    if learning == 1:

        # optional, to plot learning curves
        cv = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)

        AO_learning_curves.plot_learning_curve(gp_n_interact, dist_arr, n_body_list, cv = cv, n_jobs = 1)
        plt.show()

    else:
        pass

    # extracting shortest, middle and longest distances
    X_train_0 = []
    for i in X_train_n:
        X_train_0.append(i[0])

    X_train_1 = []
    for i in X_train_n:
        X_train_1.append(i[1])

    X_train_2 = []
    for i in X_train_n:
        X_train_2.append(i[2])

    plt.scatter(X_train_0, Y_train_n, s = 8, label = 'Short')
    plt.scatter(X_train_1, Y_train_n, s = 8, label = 'Medium')
    plt.scatter(X_train_2, Y_train_n, s = 8, label = 'Long')
    plt.xlabel('Intercolloid Distance')
    plt.ylabel('3-body overlap volume')
    plt.show()

    if model == 0:
        Y_pred = np.asarray(Y_pred)
        Y_test_n = np.asarray(Y_test_n)

    Y_pred_shaped = Y_pred.reshape(-1, 1)
    Y_test_n_shaped = Y_test_n.reshape(-1, 1)

    reg = LinearRegression()
    reg.fit(Y_pred_shaped, Y_test_n_shaped)
    print 'R2 Score: {}'.format(reg.score(Y_test_n_shaped, Y_pred_shaped))
    coefs = reg.coef_
    intercept = reg.intercept_
    print 'Coefficients: {}'.format(coefs)
    print 'Intercept: {}'.format(intercept)

    fig, ax = plt.subplots()

    # plot of prediction value against test value
    X_ = np.linspace(min(Y_test_n), max(Y_test_n), 100)
    textstr = 'R2 score: {}'.format(coefs[0])
    ax.scatter(Y_test_n, Y_pred, s = 8, c = 'g', label = 'r = {}'.format(info_list_3[2]))
    # plt.errorbar(Y_test_n, Y_pred, yerr = Y_std, fmt = 'o', capsize = 2)
    ax.plot(X_, X_*coefs[0] + intercept, c = 'r', label = 'Best fit line')
    ax.plot(X_, X_, c = 'b', label = 'y = x')
    ax.text(0.7, 0.1, textstr, transform=ax.transAxes, fontsize=8,
        verticalalignment='top')
    ax.legend()
    ax.set_xlabel('Test')
    ax.set_ylabel('Prediction')

    plt.show()


if __name__ == '__main__':
    f = sys.argv[1]
    plot_learning(f, learning = 0, shuffle = 0)








