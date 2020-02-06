#! usr/bin/env python2.7

"""
A script to generate a system of n hard-sphere colloids in an L-sided cube box, and to then perform Monte Carlo to compute the volume available to polymers. Configured for the AO model.

Import AO_model_MC.monte_carlo(L, R, N, precision, n, r) to generate data. When run as the main program, a test run will occur and plot describing the exclusion zones will appear.

iyhw2 20/09/2019
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dim = 2

def vec_difference(a, b, L):

    x = abs(a[0] - b[0])
    y = abs(a[1] - b[1])
    z = abs(a[2] - b[2])

    return x, y, z


def cyclic_vec_difference(a, b, L):
    """A function to compute the difference between two shortest vectors in a periodic system"""

    x_d = abs(a[0] - b[0])
    
    if a[0] < b[0]:
        x_p = a[0] + (L-b[0])
    else:
        x_p = b[0] + (L-a[0])

    if x_p > x_d:
        x = x_d
    else:
        x = x_p

    y_d = abs(a[1] - b[1])
 
    if a[1] < b[1]:
        y_p = a[1] + (L-b[1])
    else:
        y_p = b[1] + (L-a[1])

    if y_p > y_d:
        y = y_d
    else:
        y = y_p

    z_d = abs(a[2] - b[2])
 
    if a[2] < b[2]:
        z_p = a[2] + (L-b[2])
    else:
        z_p = b[2] + (L-a[2])

    if z_p > z_d:
        z = z_d
    else:
        z = z_p

    return x, y, z


def pythagoras(x, y, z):
    """A function to calculate 3D pythagoras"""
    
    return np.sqrt(x**2 + y**2 + z**2)


def generate_system(L, R, n, r, precision):
    """Generates the specified system of colloids. Returns coordinates of colloids and their pairwise distances"""

    colloid_centres = []

    # iterate through the different colloids
    for i in range(0, n):
        dist = 0
        
        # remain in loop if the smallest dist. between colloids is < diameter of colloids 
        while dist <= 2*R:
            single_colloid = []
            
            if dim == 3:
                for j in range(0, 3):
                    # generate random 3-D xyz coordinates for the colloid
                    unscaled_coord = random.randint(R + r, precision*(L-R-r))
                    coord = float(unscaled_coord)/precision
                    single_colloid.append(coord)
            
            else:
                for j in range(0, 2):
                    # generate random 2_D xy coordinates for the colloid
                    unscaled_coord = random.randint(R + r, precision*(L-R-r))
                    coord = float(unscaled_coord)/precision
                    single_colloid.append(coord) 
                
                single_colloid.append(0)    # z-coord is 0

            if colloid_centres:
                indiv_dist = []

                # compute distance between new colloid and all previously generated colloids
                for k in colloid_centres:
                    x, y, z = vec_difference(k, single_colloid, L)
                    indiv_dist.append(pythagoras(x, y, z))
                
                # take the smallest distance
                dist = min(indiv_dist)
            
            # exit while loop if this is the first colloid generated
            else:
                break

        colloid_centres.append(single_colloid)

    # generate pairwise distances between colloids
    init_dist_list = []
    for i in colloid_centres:
        for j in colloid_centres:
            x, y, z = vec_difference(i, j, L)
            init_dist_list.append(pythagoras(x, y, z))

    dist_set = list(set(init_dist_list))
    dist_list = [i for i in dist_set if i != 0.0]
    dist_list = sorted(dist_list)

    return colloid_centres, dist_list


def constrained_generate_system(L, R, n, r, precision):

    colloid_centres = []

    # iterate through the different colloids
    for i in range(0, n):
        min_dist = 0
        max_dist = 0
        
        # remain in loop if the smallest dist. between colloids is < diameter of colloids 
        while min_dist <= 2*R or max_dist > 2*R + 2*r:
            single_colloid = []
            
            if dim == 3:
                for j in range(0, 3):
                    # generate random 3-D xyz coordinates for the colloid
                    unscaled_coord = random.randint(precision*(R + r), precision*(L-R-r))
                    coord = float(unscaled_coord)/precision
                    single_colloid.append(coord)
            
            else:
                for j in range(0, 2):
                    # generate random 2_D xyz coordinates for the colloid
                    unscaled_coord = random.randint(precision*(R + r), precision*(L-R-r))
                    coord = float(unscaled_coord)/precision
                    single_colloid.append(coord)
                
                single_colloid.append(0)    

            if colloid_centres:
                indiv_dist = []

                # compute distance between new colloid and all previously generated colloids
                for k in colloid_centres:
                    x, y, z = vec_difference(k, single_colloid, L)
                    indiv_dist.append(pythagoras(x, y, z))
                
                # take the smallest distance
                min_dist = min(indiv_dist)
                max_dist = max(indiv_dist)
            
            # exit while loop if this is the first colloid generated
            else:
                break

        colloid_centres.append(single_colloid)

    # generate pairwise distances between colloids
    init_dist_list = []
    for i in colloid_centres:
        for j in colloid_centres:
            x, y, z = vec_difference(i, j, L)
            init_dist_list.append(pythagoras(x, y, z))

    dist_set = list(set(init_dist_list))
    dist_list = [i for i in dist_set if i != 0.0]
    dist_list = sorted(dist_list)

    return colloid_centres, dist_list


def specific_generate_system(L, R, r, precision, lengths):
    
    l_1 = lengths[0]
    l_2 = lengths[1]
    l_3 = lengths[2]

    colloid_centres = []
    colloid_centres.append([0,0,0])
    colloid_centres.append([0,l_1,0])

    y_2 = (l_1**2 + l_2**2 - l_3**2)/(2*l_1)
    x_2 = np.sqrt(l_2**2 - y_2**2)

    colloid_centres.append([x_2, y_2, 0])

    dist_list = [l_1, l_2, l_3]

    return colloid_centres, dist_list


def monte_carlo(L, R, n, r, precision, N, plotting = None, constrained = 0, lengths = [], input_coords = []):
    """Performs Monte Carlo to compute the volume fraction that is accessible to polymer"""

    if plotting:
        x_coords = []
        y_coords = []
        z_coords = []
    
    else:
        pass

    # generate random system of colloids
    if lengths and not input_coords:
        colloid_centres, dist_list = specific_generate_system(L, R, r, precision, lengths)

    elif lengths and input_coords:
        raise ValueError('Check inputs!')
    else:
        if input_coords:
            colloid_centres = input_coords
            
            init_dist_list = []
            for i in colloid_centres:
                for j in colloid_centres:
                    x, y, z = vec_difference(i, j, L)
                    init_dist_list.append(pythagoras(x, y, z))
            
            dist_set = list(set(init_dist_list))
            dist_list = [i for i in dist_set if i != 0.0]

        else:
            if constrained:
                colloid_centres, dist_list = constrained_generate_system(L, R, n, r, precision)

            else:
                colloid_centres, dist_list = generate_system(L, R, n, r, precision)
        
    hit_count = 0
    
    # iterate through desired no. of steps
    for i in range(0, N):
        coords = []
        
        # generate coordinates for a random point in the box
        if dim == 3:
            for j in range(0,3):
                unscaled_point = random.randint(0, precision*L)
                point = float(unscaled_point)/precision
                coords.append(point)
            
        else:
            for j in range(0,2):
                unscaled_point = random.randint(0, precision*L)
                point = float(unscaled_point)/precision
                coords.append(point)
            
            coords.append(0)

        indiv_dist = []
    
        # compute distance between this random point and all colloids
        for k in colloid_centres:
            x, y, z = vec_difference(coords, k, L)
            indiv_dist.append(pythagoras(x, y, z))
        
        # take the smallest such distance
        dist = min(indiv_dist)

        # if the point lies outside the exclusion zone, count as hit
        if dist <  R + r:
            hit_count += 1
            
            if plotting:
                x_coords.append(coords[0])
                y_coords.append(coords[1])
                z_coords.append(coords[2])

            else:
                pass

    hit_rate = float(hit_count)/N
    if dim == 3:
        tot_vol = L**3

    else:
        tot_vol = L**2

    vol_frac = hit_rate*tot_vol

    MC_error = np.sqrt((hit_rate*(1-hit_rate))/N)
    MC_error = tot_vol*MC_error

    print colloid_centres

    if plotting:
        if dim == 3:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.set_aspect('equal')
            ax.scatter(x_coords, y_coords, z_coords, c = 'r')

            ax.set_xlim3d(0, L)
            ax.set_ylim3d(0, L)
            ax.set_zlim3d(0, L)

            plt.show()

        else:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
            ax.scatter(x_coords, y_coords, c = 'r')

            plt.show()

    else:
        pass

    # returns the free volume, inter-colloid distances, MC error and colloid centres
    return vol_frac, dist_list, MC_error, colloid_centres


if __name__ == '__main__':
    #monte_carlo(L = 8, R = 1, n = 3, r = 1, precision = 10000000000000, N = 10000, plotting = 1, input_coords = [[0.822216, 2.977126, 0],[0.665972, 0.15291024, 0],[0.9777810, 5.751146,0]])
    monte_carlo(L = 10, R = 1, n = 3, r = 1, precision = 10000000000000, N = 10000, plotting = 1, constrained = 1)











