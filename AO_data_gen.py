#! usr/bin/env python2.7

"""
A script that generates Monte Carlo data for the AO model, returning the volume availble to polymers and a list of inter-colloid distances. Can also plot the data for the two-colloid system. 

iyhw2, 23/09/2019
"""

import sys
import AO_model_MC as MC
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dim = 2

FILE = sys.argv[1]
FILE_csv = FILE + '.csv'
FILE_info = FILE + '_info'


def generate_data(L, R, n, r, precision, N, sims, plot = None, constrained = 0):

    if dim == 3:
        sphere_vol = (float(4)/3)*np.pi*(R+r)**3
        tot_vol = L**3

    else:
        sphere_vol = (np.pi*(R+r)**2)
        tot_vol = L**2

    occ_vol = sphere_vol*n
    vol_av = tot_vol - occ_vol

    info_list = [vol_av, R, r]

    with open(FILE_info, 'wb') as fp:
        pickle.dump(info_list, fp)

    vol_frac_list = []
    col_dist_list = []
    err_list = []

    # iterate through the desired number of simulations
    for i in range(0, sims):
        print 'Currently performing simulation number {}...'.format(i)

        vol_frac, dist_list, MC_error = MC.monte_carlo(L, R, n, r, precision, N, constrained)
        vol_frac_list.append(vol_frac)
        col_dist_list.append(dist_list)
        err_list.append(MC_error)

    vol_list = [i - vol_av for i in vol_frac_list]

    dist_dic = {}
    dist_list_len = len(col_dist_list[0])

    # split inter-colloid distance list of lists into seperate dictionary entries
    for i in range(0, dist_list_len):
        dist_dic['distance_{}'.format(i)] = [j[i] for j in col_dist_list]

    # if n == 2 system and if argument passed for plot, then plot the data
    if n == 2 and plot:
        #plt.scatter(dist_dic['distance_0'], vol_list)
        plt.errorbar(dist_dic['distance_0'], vol_list, yerr = err_list,  fmt = 'o')
        plt.xlabel('Inter-colloid distance')
        plt.ylabel('Volume available for polymers')
        plt.show()
        
    elif n > 2 and plot:
        print "Cannot plot! Dimensionality too high to visualise"

    else:
        pass

    df_dist = pd.DataFrame.from_dict(dist_dic)
    df_vol = pd.DataFrame(vol_list, columns = ['Vol'])
    df_err = pd.DataFrame(err_list, columns = ['Err'])
    df = pd.concat([df_dist, df_vol, df_err], axis = 1, sort = False)
    df.to_csv(FILE_csv, index = False)

    return dist_dic, vol_list, err_list


if __name__ == '__main__':
    generate_data(L = 8, R = 1, n = 2, r = 0.5, precision = 10000000000000, N = 1000000, sims = 100, plot = 1)















