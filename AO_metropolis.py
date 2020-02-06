#!/usr/bin/env/python2.7

"""
Improved version of the AO_metropolis program, which performs MC on the colloid system using either the AO/2-bodyML/3-bodyML potential. Implements a grid system. Call with AO_metropolis_grid.py [input data] [outfile]

iyhw2 14/01/20
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.collections
import AO_model_MC
import AO_GP
import AO_pair_ham
import AO_remainder
import sys
import random
import copy
import pickle
import time


n = 108  # number of colloids simulated
L = 41  # length of sides of box
R = 1  # colloid radius
r = 1  # polymer radius
dim = 2

model = 0  # model = 0 for analytical AO, = 1 for 2-body ML, = 2 for 3-body ML

def generate_colloid_dict(colloid_centres):
    """A function that generates a dictionary from a list of colloids"""

    colloid_dict = {}

    for index, i in enumerate(colloid_centres):
        colloid_dict[index] = i

    return colloid_dict


def full_interaction(colloid_dict):
    """A function that computes all pairwise interactions and returns a dictionary"""

    vol_dict = {}

    for index, i in colloid_dict.iteritems():
        init_dist_list = []

        for j in colloid_dict.itervalues():

            # compute vec difference between the vectors
            x, y, z = AO_model_MC.cyclic_vec_difference(i, j, L)
        
            # get distance and append
            init_dist_list.append(AO_model_MC.pythagoras(x, y, z))

        # remove all zeros 
        dist_list = [w for w in init_dist_list if w != 0.0]
 
        pair_vol = 0

        if model == 0 or model == 2:
        # fit each distance to analytical soln and sum
            for l in dist_list:
                predict = AO_pair_ham.full_pair_ham(l, 2*r, 2*R)
                pair_vol += predict

        elif model == 1:
        # fit each distance to the ML model and sum
            for l in dist_list:
                if l <= 2*R + 2*r:
                    m = l.reshape(-1, 1)
                    predict = gp.predict(m)
                    pair_vol += predict[0]
       
                else:
                    pass

        # update dictionary to contain volumes
        vol_dict[index] = [pair_vol]

    if model == 2:

        for index, i in colloid_dict.iteritems():
            dist_list = []
            for j in colloid_dict.itervalues():
                    for k in colloid_dict.itervalues():
                        
                        init_dist_list = []

                        x_1, y_1, z_1 = AO_model_MC.cyclic_vec_difference(i, j, L)
                        x_2, y_2, z_2 = AO_model_MC.cyclic_vec_difference(i, k, L)
                        x_3, y_3, z_3 = AO_model_MC.cyclic_vec_difference(j, k, L)
               
                        init_dist_list.append(AO_model_MC.pythagoras(x_1, y_1, z_1)) 
                        init_dist_list.append(AO_model_MC.pythagoras(x_2, y_2, z_2))
                        init_dist_list.append(AO_model_MC.pythagoras(x_3, y_3, z_3))

                        init_dist_list = sorted(init_dist_list)

                        if 0.0 in init_dist_list:
                            pass

                        else:
                            dist_list.append(init_dist_list)
            
            n_vol = 0

            for l in dist_list:
                if any(u >= 2*R + 2*r for u in l):
                    pass
                else:
                    m = np.array(l)
                    m = m.reshape(1, -1)
                    predict = gp_n_interact.predict(m)
                    n_vol += predict[0]

            vol_dict[index].append(n_vol)

    tot_pair_vol = 0
    tot_n_vol = 0

    # compute total overlap vol
    for i in vol_dict.itervalues():
        tot_pair_vol += i[0]
    
    tot_pair_vol = 0.5*tot_pair_vol
    
    if model == 2:
        for i in vol_dict.itervalues():
            tot_n_vol += i[1]
    
    tot_n_vol = (1.0/3)*tot_n_vol

    tot_vol = tot_pair_vol - tot_n_vol

    # return dictionary of volumes and total overlap volume
    return vol_dict, tot_vol


def make_grid(colloid_dict):
    """Generates a grid and a book using colloid_dict"""

    # define grid parameters
    cell_width = math.ceil(2*R + 2*r)
    cell_axis = np.linspace(0, L, int(L)/cell_width)

    # generate empty grid
    grid = [ [[] for i in range(0, len(cell_axis)-1)] for j in range(0, len(cell_axis)-1)]

    book = {}

    # populate grid
    for key_i, i in colloid_dict.iteritems():
        for index_j, j in enumerate(cell_axis):
            if i[0] <= j:
                x_key = index_j
                break
        for index_j, j in enumerate(cell_axis):
            if i[1] <= j:
                y_key = index_j
                break

        grid[y_key-1][x_key-1].append(key_i)
        
        # populate book
        book[key_i] = [y_key-1, x_key-1]

    return grid, book


def refresh_grid(colloid_dict, grid, book, index):
    """Function to update grid and book"""

    # find colloid location and remove from grid
    location_x = book[index][0]
    location_y = book[index][1]
    grid[location_x][location_y].remove(index)
 
    # define grid parameters
    cell_width = math.ceil(2*R + 2*r)
    cell_axis = np.linspace(0, L, int(L)/cell_width)

    # insert new info into grid and book
    for index_i, i in enumerate(cell_axis):
        if colloid_dict[index][0] <= i:
            x_key = index_i
            break
    for index_i, i in enumerate(cell_axis):
        if colloid_dict[index][1] <= i:
            y_key = index_i
            break

    grid[y_key-1][x_key-1].append(index)
    book[index] = [y_key-1, x_key-1]

    return grid, book


#init_colloid_centres, init_dist_list = AO_model_MC.generate_system(L, R, n, r, 10000000000, dim)
#colloid_dict = generate_colloid_dict(init_colloid_centres)
#vol_dict, tot_vol = full_interaction(colloid_dict)
#grid, book = make_grid(colloid_dict)
#single_interaction(grid, book, colloid_dict, vol_dict, 1)


def single_interaction(grid, book, colloid_dict, vol_dict, index):
    """Function that computes the interactions of a single colloid"""

    neighbour_list = []
    location_x = book[index][0]
    location_y = book[index][1]
    grid_len = len(grid)

    # find keys of indexed colloid and all adjacent colloids
    for i in range(0, 3):
        for j in range(0, 3):
            if (location_x+(i-1) < grid_len) and (location_y+(j-1) < grid_len):
                neighbour_list.append(grid[location_x+(i-1)][location_y+(j-1)])
            # ensure periodicity
            elif (location_x+(i-1) >= grid_len) and (location_y+(j-1) < grid_len):
                neighbour_list.append(grid[location_x+(i-1) - grid_len][location_y+(j-1)])

            elif (location_x+(i-1) < grid_len) and (location_y+(j-1) >= grid_len):
                neighbour_list.append(grid[location_x+(i-1)][location_y+(j-1) - grid_len])

            else:
                neighbour_list.append(grid[location_x+(i-1) - grid_len][location_y+(j-1) - grid_len])
    
    # flatten neighbour list
    flat_list = []
    for i in neighbour_list:
        for j in i:
            flat_list.append(j)

    dist_list = []

    for i in flat_list:
        # compute vec difference between specified colloid and all colloids in adjacent and self cell
        x, y, z = AO_model_MC.cyclic_vec_difference(colloid_dict[i], colloid_dict[index], L)
        dist_list.append(AO_model_MC.pythagoras(x, y, z))

    dist_list = [w for w in dist_list if w != 0.0]
    if dist_list:
        min_dist = min(dist_list)
    else:
        # if dist_list empty, consider min_dist as...
        min_dist = 2*R + 2*r

    pair_vol = 0

    if model == 0 or model == 2:
    # fit to analytical soln and sum 
        for l in dist_list:
            predict = AO_pair_ham.full_pair_ham(l, 2*r, 2*R)
            pair_vol += predict

    elif model == 1:
    # fit to ML model, and sum volumes
        for i in dist_list:
            if i <= 2*R + 2*r:
                j = i.reshape(-1, 1)
                predict = gp.predict(j)
                pair_vol += predict[0]
            else:
                pass

    # update the relevant volume
    vol_dict[index][0] = pair_vol

    if model == 2:
        dist_list = []

        # compute three-way interactions
        for i in flat_list:
            for j in flat_list:

                init_dist_list = []

                x_1, y_1, z_1 = AO_model_MC.cyclic_vec_difference(colloid_dict[index], colloid_dict[i], L)
                x_2, y_2, z_2 = AO_model_MC.cyclic_vec_difference(colloid_dict[i], colloid_dict[j], L)
                x_3, y_3, z_3 = AO_model_MC.cyclic_vec_difference(colloid_dict[index], colloid_dict[j], L)

                init_dist_list.append(AO_model_MC.pythagoras(x_3, y_3, z_3))
                init_dist_list.append(AO_model_MC.pythagoras(x_1, y_1, z_1))
                init_dist_list.append(AO_model_MC.pythagoras(x_2, y_2, z_2))

                init_dist_list = sorted(init_dist_list)

                if 0.0 in init_dist_list:
                    pass

                else:
                    dist_list.append(init_dist_list)

        n_vol = 0
        # only fit to ML model for small triangles
        for l in dist_list:
            if any(u > 2*R + 2*r for u in l):
                pass
            else:
                m = np.array(l)
                m = m.reshape(1, -1)
                predict = gp_n_interact.predict(m)
                n_vol += predict[0]

        vol_dict[index][1] = n_vol

    tot_pair_vol = 0
    tot_n_vol = 0

    # compute total overlap vol
    for i in vol_dict.itervalues():
        tot_pair_vol += i[0]
    
    tot_pair_vol = 0.5*tot_pair_vol
    
    if model == 2:
        for i in vol_dict.itervalues():
            tot_n_vol += i[1]
    
    tot_n_vol = (1.0/3)*tot_n_vol

    tot_vol = tot_pair_vol - tot_n_vol

    # return the new vol_dict, the new total volume and the minimum distance between selected colloid and all other colloids
    return vol_dict, tot_vol, min_dist


def system_plot(colloid_centres, L, form = 'l'):
    """Function to plot and save snapshots of system"""

    x_data = []
    y_data = []
    z_data = []
    
    # for a colloid list
    if form == 'l':
        for i in colloid_centres:
            x_data.append(i[0])
            y_data.append(i[1])
            z_data.append(i[2])

    # for a colloid dict
    elif form == 'd':
        for i in colloid_centres.itervalues():
            x_data.append(i[0])
            y_data.append(i[1])
            z_data.append(i[2])

    if dim == 3:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(x_data, y_data, z_data)
        ax.set_xlim3d(0, L)
        ax.set_ylim3d(0, L)
        ax.set_zlim3d(0, L)
        plt.savefig('Init_' + outfile + '.png')

    if dim == 2:
        coords = zip(x_data, y_data)
        patches = [plt.Circle(center, R) for center in coords]
        fig, ax = plt.subplots()
        coll = matplotlib.collections.PatchCollection(patches, facecolor = 'black')  # use patches to generate accurately sized circles
        ax.add_collection(coll)
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)

        ax.set_aspect('equal')

    return plt


def metropolis(cycles, amplitude, temperature, dim, plotting = None, periodic_save = None):
    """Function to carry out the Metropolis algorithm"""

    init_colloid_centres, init_dist_list = AO_model_MC.generate_system(L, R, n, r, 10000000000)
    count = 0
    accept_count = 0
    # plot initial configuration, save as png
    if plotting:
        init_plt = system_plot(init_colloid_centres, L)
        init_plt.title('Count: {}, R: {}, r: {}, n: {}, L: {}, Amplitude: {}, Temperature: {}'.format(count, R, r, n, L, amplitude, temperature), fontsize = 8, verticalalignment = 'bottom')
        #init_plt.savefig('Init_' + outfile + '.png')

    else:
        pass

    # make a colloid_dict and corresponding vol_dict
    colloid_dict = generate_colloid_dict(init_colloid_centres)
    vol_dict, vol = full_interaction(colloid_dict)
    grid, book = make_grid(colloid_dict)

    final_vol_list = []
    count_list = []

    # select configs periodically for saving - for g(r) calculations
    savepoint = 0.5*cycles
    T = np.linspace(savepoint, cycles, 41)
    saved_colloid_list = []

    start_time = time.time()
    
    print '------------------------------------'

    while count < cycles:

        # save config if selected - for g(r) calculations
        if count in T:
            saved_colloid = []
            for i in colloid_dict.itervalues():
                saved_colloid.append(i)
            
            saved_colloid_list.append(copy.deepcopy(saved_colloid))

        count += 1
        # only save vol data for every n steps
        if count % 1000 == 0:
            final_vol_list.append(vol)
            count_list.append(count)

        else:
            pass

        # save snapshots at specific counts
        if (count <= 40000 and count % 4000 == 0 and periodic_save) or (count > 40000 and count % 1000000 == 0 and periodic_save):

            mid_plt = system_plot(colloid_dict, L, form = 'd')
            mid_plt.title('Count: {}, R: {}, r: {}, n: {}, L: {}, Amplitude: {}, Temperature: {}'.format(count, R, r, n, L, amplitude, temperature), fontsize = 8, verticalalignment = 'bottom')
            mid_plt.savefig('{}_{}.png'.format(count, outfile))

        else:
            pass

        print 'Cycle Number: {}'.format(count)
        print 'Initial Volume: {}'.format(vol)

        # make a random move, scaled by amplitude
        move = (-0.5*amplitude) + amplitude*(random.uniform(0,1))
        # select a random colloid to move
        random_colloid = random.randint(0, n-1)
        
        if dim == 3:
            random_coordinate = random.randint(0,2)

        else:
            # if 2D, never move z-coordinate away from 0
            random_coordinate = random.randint(0,1)

        print 'Move: {}'.format(move)
 
        # find initial value to update
        update_value = colloid_dict[random_colloid][random_coordinate]
        
        # generate updated value
        update_move = update_value + move
        
        # ensure movement is also periodic
        if update_move >= L:
            update_move = update_move - L
        elif update_move < 0:
            update_move = update_move + L
        else:
            pass

        # update colloid_dict with perturbation
        colloid_dict[random_colloid][random_coordinate] = update_move
       
        print 'Perturbed Colloid Index: {}'.format(random_colloid)
        print 'Perturbed Coordinate: {}'.format(random_coordinate)

        # update grid and book with perturbation
        update_grid, update_book = refresh_grid(colloid_dict, grid, book, random_colloid)

        # generate new vol_dict from new colloid_dict
        update_vol_dict, update_vol, min_dist = single_interaction(update_grid, update_book, colloid_dict, vol_dict, random_colloid)
        
        print 'Update Volume: {}'.format(update_vol)
 
        # if total overlap volume is now higher, always accept move as long as move is legal
        if update_vol >= vol and min_dist >= 2*R: 
            vol = update_vol
            vol_dict = update_vol_dict
            grid = update_grid
            book = update_book
            accept_count += 1
            print 'Update Accepted'

        # if total overlap volume is now lower and move is legal, make move depending on Metropolis condition
        elif update_vol < vol and min_dist >= 2*R:
            P = np.exp((update_vol - vol) / temperature)
            uniform_no = random.uniform(0,1)

            if P > uniform_no:
                vol = update_vol
                vol_dict = update_vol_dict
                grid = update_grid
                book = update_book
                accept_count += 1
                print 'Update Accepted with P = {}'.format(P)

            else:
                colloid_dict[random_colloid][random_coordinate] = update_value
                
                grid, book = refresh_grid(colloid_dict, update_grid, update_book, random_colloid) 
                print 'Update Rejected by Metropolis'

        # if move is illegal, always reject
        else:
            colloid_dict[random_colloid][random_coordinate] = update_value
            
            grid, book = refresh_grid(colloid_dict, update_grid, update_book, random_colloid) 
            
            print 'Update Rejected: move into infinite potential attempted'


        print '------------------------------------'


    accept_ratio = float(accept_count)/count
    print 'Run completed in {} seconds'.format(time.time() - start_time)

    # pickle saved configs - to use for g(r) calculations
    #outfile_col_info = outfile + '_col_info'
    #with open(outfile_col_info, 'wb') as fp:
        #pickle.dump(saved_colloid_list, fp)

    if plotting:
    
    # save plot of total volume vs cycles as png
        MC_plt = plt.scatter(count_list, final_vol_list, s=8)
        MC_plt.xlabel('Cycles')
        MC_plt.ylabel('Total Overlap Volume')
        MC_plt.savefig('MC_' + outfile + '.png')

    # plot final config, save as png
    if plotting:

        final_plt = system_plot(colloid_dict, L, form = 'd')
        final_plt.title('R: {}, r: {}, n: {}, L: {}, Amplitude: {}, Temperature: {}, Accept Ratio: {}'.format(R, r, n, L, amplitude, temperature, accept_ratio), fontsize = 8, verticalalignment = 'bottom')
        final_plt.savefig('Final_' + outfile + '.png')

    else:
        pass

    print '------------------------------------'

    return saved_colloid_list


def metropolis_repeat(trials, cycles, amplitude, temperature, plotting, dim):

    final_save_list = []
    for i in range(0, trials):
        saved_colloid_list = metropolis(cycles = cycles, amplitude = amplitude, temperature = temperature, plotting = plotting, dim = dim)
        final_save_list.append(saved_colloid_list)
 
    outfile_col_info = outfile + '_col_info'
    with open(outfile_col_info, 'wb') as fp:
        pickle.dump(final_save_list, fp)


if __name__ == '__main__':
    # input file parser
    f = sys.argv[1]

    # input output file name
    outfile = sys.argv[2]

    gp, X_test, Y_test, Y_pred, std, X_train, Y_train, MSE, column_count = AO_GP.AO_gauss_regressor(f)

    gp_n_interact, X_train_n, X_test_n, Y_train_n, Y_test_n, dist_arr, n_body_list, Y_pred_n, Y_std = AO_remainder.AO_3body_remainder(f) 

    metropolis_repeat(trials = 10, cycles = 1000000, amplitude = 1, temperature = 5, plotting = 0, dim = 2)





