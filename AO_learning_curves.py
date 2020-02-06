#! /usr/bin/env/ python2.7

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel as C
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(estimator, X, y, outfile, cv = None, n_jobs = None, train_size = np.linspace(0.1, 1.0, 20)):

    train_size, train_score, test_score = learning_curve(estimator, X, y, cv = cv, n_jobs = n_jobs, train_sizes = train_size) 

    train_score_mean = np.mean(train_score, axis = 1)
    train_score_std = np.std(train_score, axis = 1)
    test_score_mean = np.mean(test_score, axis = 1)
    test_score_std = np.std(test_score, axis = 1)

    df_size = pd.DataFrame(train_size, columns = ['TrainSize'])
    df_train_mean = pd.DataFrame(train_score_mean, columns = ['TrainMean'])
    df_train_std = pd.DataFrame(train_score_std, columns = ['TrainStd'])
    df_test_mean = pd.DataFrame(test_score_mean, columns = ['TestMean'])
    df_test_std = pd.DataFrame(test_score_std, columns = ['TestStd'])
    df_learn = pd.concat([df_size, df_train_mean, df_train_std, df_test_mean, df_test_std], axis = 1, sort = False)
    df_learn.to_csv(outfile + '_learn.csv', index = False)

    plt.scatter(train_size, train_score_mean, s = 8, label = 'Training Score')
    plt.scatter(train_size, test_score_mean, s = 8, label = 'CV Score' )
    plt.fill_between(train_size, train_score_mean - train_score_std, train_score_mean + train_score_std, alpha = 0.3)
    plt.fill_between(train_size, test_score_mean - test_score_std, test_score_mean + test_score_std, alpha = 0.3)
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.legend()

    return plt


if __name__ == '__main__':

    f = sys.argv[1]
    f_csv = f + '.csv'
    f_info = f + '_info'

    df = pd.read_csv(f_csv)
    column_count = len(df.columns) - 2
    cdf = df.iloc[:, :column_count:]
    dist_list = cdf.values.tolist()
    dist_arr = np.array(dist_list)
    vol_arr = np.array(df.Vol)

    if column_count == 1:
        dist_arr = dist_arr.reshape(-1, 1)
        vol_arr = vol_arr.reshape(-1, 1)

    cv = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)

    if column_count == 1:
        kernel = RBF(1, (1e-2, 1e3)) + C(noise_level = 1e-5, noise_level_bounds = (1e-10, 1e1))

    else:
        aniso_kernel = [1]*column_count
        kernel = RBF(aniso_kernel, (1e-2, 1e3)) + C(noise_level = 1e-5, noise_level_bounds = (1e-10, 1e1))

    estimator = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = 20, normalize_y = False)

    estimator.fit(dist_arr, vol_arr)
    print estimator.kernel_

    plot_learning_curve(estimator, dist_arr, vol_arr, cv = cv, n_jobs = 1, outfile = f)
    plt.show()








