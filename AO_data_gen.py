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


def generate_data(L, R, n, r, precision, N, sims, plot = None, constrained = 0):

    FILE = sys.argv[1]
    FILE_csv = FILE + '.csv'
    FILE_info = FILE + '_info'

    if dim == 3:
        sphere_vol = (float(4)/3)*np.pi*(R+r)**3
        tot_vol = L**3

    else:
        sphere_vol = (np.pi*(R+r)**2)
        tot_vol = L**2

    occ_vol = sphere_vol*n

    info_list = [occ_vol, R, r]

    with open(FILE_info, 'wb') as fp:
        pickle.dump(info_list, fp)

    vol_frac_list = []
    col_dist_list = []
    err_list = []
    col_list = []

    # iterate through the desired number of simulations
    for i in range(0, sims):
        print 'Currently performing simulation number {}...'.format(i)

        vol_frac, dist_list, MC_error, colloid_centres = MC.monte_carlo(L, R, n, r, precision, N, constrained = constrained)
        vol_frac_list.append(vol_frac)
        col_dist_list.append(dist_list)
        err_list.append(MC_error)
        col_list.append(colloid_centres)

    vol_list = [occ_vol - i for i in vol_frac_list]

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
    df_col = pd.DataFrame(col_list, columns = ['Coord_1', 'Coord_2', 'Coord_3'])
    df = pd.concat([df_dist, df_vol, df_err, df_col], axis = 1, sort = False)
    df.to_csv(FILE_csv, index = False)

    return dist_dic, vol_list, err_list


def generate_specific_data(L, R, n, r, precision, N, lengths = [], input_coords = []):

    sphere_vol = (np.pi*(R+r)**2)
    tot_vol = L**2

    occ_vol = sphere_vol*n

    if lengths and not input_coords:
        vol_frac, dist_list, MC_error, colloid_centres = MC.monte_carlo(L, R, n, r, precision, N, lengths = lengths)

    elif not lengths and input_coords:
        vol_frac, dist_list, MC_error, colloid_centres = MC.monte_carlo(L, R, n, r, precision, N, input_coords = input_coords)

    else:
        raise ValueError('Check inputs!')

    vol = occ_vol - vol_frac

    print vol

    return vol, dist_list, MC_error


if __name__ == '__main__':
    #generate_specific_data(L = 8, R = 1, n = 3, r = 1, precision = 10000000000000, N = 5000000, input_coords = [[0,0,0], [2,3,0], [5,5,0]])
    generate_data(L = 10, R = 1, n = 3, r = 1, precision = 10000000000000000, N = 5000000, sims = 500, plot = 1, constrained = 1)


