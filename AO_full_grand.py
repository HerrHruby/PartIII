#! usr/bin/env python2.7

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections
import AO_model_MC
import sys
import random
import math
import copy
import pickle

L = 33
R = 1
r = 1
dim = 2
precision = 1000000000000
delta = r + R


def generate_polymers(L, R, n_p, r, precision, colloid_centres):

    polymer_centres = []

    if colloid_centres:
        for i in range(0, n_p):
            dist = 0

            while dist <= r + R:
                single_polymer = []

                if dim == 3:
                    for j in range(0, 3):
                        unscaled_coord = random.randint(0, precision*L)
                        coord = float(unscaled_coord)/precision
                        single_polymer.append(coord)

                elif dim == 2:
                    for j in range(0, 2):
                        unscaled_coord = random.randint(0, precision*L)
                        coord = float(unscaled_coord)/precision
                        single_polymer.append(coord)

                    single_polymer.append(0)

                else:
                    print 'Check Dimensionality'
                    break

                indiv_dist = []

                for k in colloid_centres:
                    x, y, z = AO_model_MC.vec_difference(k, single_polymer, L)
                    indiv_dist.append(AO_model_MC.pythagoras(x, y, z))

                dist = min(indiv_dist)

            polymer_centres.append(single_polymer)

    else:
        for i in range(0, n_p):

            single_polymer = []

            if dim == 3:
                for j in range(0, 3):
                    unscaled_coord = random.randint(0, precision*L)
                    coord = float(unscaled_coord)/precision
                    single_polymer.append(coord)

            elif dim == 2:
                for j in range(0, 2):
                    unscaled_coord = random.randint(0, precision*L)
                    coord = float(unscaled_coord)/precision
                    single_polymer.append(coord)

                single_polymer.append(0)

            else:
                print 'Check Dimensionality'
                break

            polymer_centres.append(single_polymer)

    return polymer_centres


def compute_packing(n_c):

    if dim == 2:
        colloid_vol = n_c*(np.pi)*(R**2)
        box_vol = L**2
        col_frac = float(colloid_vol)/box_vol

    return col_frac


def GC_monte_carlo(cycles, n_c, n_p, M, z_c, res_pack_frac, plotting_mode = 0):
    
    init_colloid_centres, init_dist_list = AO_model_MC.generate_system(L, R, n_c, r, precision, dim)
    init_polymer_centres = generate_polymers(L, R, n_p, r, precision, init_colloid_centres)

    init_plt = system_plot(init_colloid_centres, init_polymer_centres)
    init_plt.savefig('Init_' + outfile + '.png')
    init_plt.show()

    if dim == 2:
        V_r = (np.pi)*r**2
        z_p = res_pack_frac/V_r

    count = 0

    V_delta = 4*(np.pi)*delta**2
    m = V_delta*z_p + 1
    m = int(round(m))

    print 'delta: {}'.format(delta)
    print 'm: {}'.format(m)
    print 'Polymer Fugacity: {}'.format(z_p)

    insert_count = 0
    remove_count = 0

    col_frac_list = []
    count_list = []

    while count < cycles:

        if n_c == 0:
            print 'All colloids removed...'
            break

        else:
            pass

        count += 1
        print 'Cycle: {}'.format(count)

        print 'Insert Count: {}'.format(insert_count)
        print 'Remove Count: {}'.format(remove_count)

        n_r = random.randint(0, m)
        print 'n_r: {}'.format(n_r)
        random_bool = random.random()

        col_frac = compute_packing(n_c)
        print 'Colloid Packing Fraction: {}'.format(col_frac)

        if count % M == 0:
            col_frac_list.append(col_frac)
            count_list.append(count)

        else:
            pass

        if random_bool < 0.5:
            random_coord = []

            unscaled_coord_x = random.randint(0, L*precision)
            unscaled_coord_y = random.randint(0, L*precision)
            random_coord_x = float(unscaled_coord_x)/precision
            random_coord_y = float(unscaled_coord_y)/precision

            random_coord.append(random_coord_x)
            random_coord.append(random_coord_y)
            random_coord.append(0)

            print 'Attempting to insert colloid at {}'.format(random_coord)
            
            selected_polymers = []

            for i in init_polymer_centres:
                x, y, z = AO_model_MC.vec_difference(i, random_coord, L)
                indiv_dist = AO_model_MC.pythagoras(x, y, z)

                if indiv_dist <= delta:
                    selected_polymers.append(i)

            print 'Selected polymers: {}'.format(selected_polymers)
            selected_polymer_count = len(selected_polymers)
            print 'Selected polymer count: {}'.format(selected_polymer_count)
            
            if n_r > selected_polymer_count:
                print 'Move rejected: selected polymer count is smaller than n_r'

            else:

                P = (((z_c*(L**2))/(n_c + 1))*((math.factorial(selected_polymer_count))/(math.factorial(selected_polymer_count - n_r)))*(1/((z_p*V_delta)**n_r)))

                uniform_no = random.uniform(0, 1)

                if P > uniform_no:
                    print 'Move accepted with P = {}'.format(P)
                    
                    switch = True

                    update_colloid_centres = copy.deepcopy(init_colloid_centres)
                    update_polymer_centres = copy.deepcopy(init_polymer_centres)

                    sampled_selected_polymers = random.sample(selected_polymers, n_r)

                    update_polymer_centres = [i for i in update_polymer_centres if i not in sampled_selected_polymers]

                    for i in update_colloid_centres:
                        x, y, z = AO_model_MC.vec_difference(i, random_coord, L)
                        indiv_dist = AO_model_MC.pythagoras(x, y, z)

                        if indiv_dist <= 2*R:
                            switch = False
                            print 'Colloid insertion failed due to overlap with colloid'
                            break
                        else:
                            pass

                    if switch:
                        for i in update_polymer_centres:
                            x, y, z = AO_model_MC.vec_difference(i, random_coord, L)
                            indiv_dist = AO_model_MC.pythagoras(x, y, z)

                            if indiv_dist <= R + r:
                                switch = False
                                print 'Colloid insertion failed due to overlap with polymer'
                                break
                            else:
                                pass

                    if switch:
                        print 'Colloid successfully inserted'

                        update_colloid_centres.append(random_coord)
                        init_colloid_centres = update_colloid_centres
                        init_polymer_centres = update_polymer_centres

                        n_c = n_c + 1
                        n_p = n_p - n_r

                        insert_count += 1
                        
                        if plotting_mode:
                            test_plt = system_plot(init_colloid_centres, init_polymer_centres)

                else:
                    print 'Move rejected with P = {}'.format(P)
            
        elif random_bool >= 0.5:
            random_colloid = random.randint(0, n_c-1)

            print 'Attempting to remove colloid {}'.format(random_colloid)
            rand_colloid_coord = init_colloid_centres[random_colloid]

            polymer_count = 0
            for i in init_polymer_centres:
                x, y, z = AO_model_MC.vec_difference(rand_colloid_coord, i, L)
                indiv_dist = AO_model_MC.pythagoras(x, y, z)
                if indiv_dist <= delta:
                    polymer_count += 1
            
            print 'No. of polymers in region: {}'.format(polymer_count)

            poly_insert = []

            for i in range(0, n_r):
                indiv_coord = []
                in_circle = False

                while not in_circle:
                    random_coord_x = random.uniform(rand_colloid_coord[0] - delta, rand_colloid_coord[0] + delta)
                    random_coord_y = random.uniform(rand_colloid_coord[1] - delta, rand_colloid_coord[1] + delta)
                    if (random_coord_x - rand_colloid_coord[0])**2 + (random_coord_y - rand_colloid_coord[1])**2 <= delta**2:
                        in_circle = True
                    else:
                        pass

                indiv_coord.append(random_coord_x)
                indiv_coord.append(random_coord_y)
                indiv_coord.append(0)

                poly_insert.append(indiv_coord)

            update_colloid_centres = copy.deepcopy(init_colloid_centres)
            update_polymer_centres = copy.deepcopy(init_polymer_centres)
            
            update_colloid_centres.pop(random_colloid)
            
            col_pol_dist_list = []

            for i in poly_insert:
                for j in update_colloid_centres:
                    x, y, z = AO_model_MC.vec_difference(i, j, L)
                    indiv_dist = AO_model_MC.pythagoras(x, y, z)
                    col_pol_dist_list.append(indiv_dist)

            if not col_pol_dist_list:
                pass
            else:
                col_pol_min_dist = min(col_pol_dist_list)

            P = ((n_c/(z_c*(L**2)))*((math.factorial(polymer_count)*((z_p*V_delta)**n_r))/(math.factorial(polymer_count + n_r))))

            uniform_no = random.uniform(0, 1)

            print 'Uniform no: {}'.format(uniform_no)

            if P > uniform_no and (col_pol_min_dist >= r + R or not col_pol_min_dist):
                print 'Move accepted with P = {}'.format(P)
                init_colloid_centres = update_colloid_centres

                for i in poly_insert:
                    update_polymer_centres.append(i)

                init_polymer_centres = update_polymer_centres
                
                n_c = n_c - 1
                n_p = n_p + n_r

                remove_count += 1

                if plotting_mode:
                    test_plt = system_plot(init_colloid_centres, init_polymer_centres)

            else:
                print 'Move rejected with P = {}'.format(P)
                pass

        print '-----------------------------'

    print 'Task Complete!'

    outfile_col_info = outfile + '_CGcol_info'

    with open(outfile_col_info, 'wb') as fp:
        pickle.dump(init_colloid_centres, fp)

    final_plt = system_plot(init_colloid_centres, init_polymer_centres)
    final_plt.savefig('Final_' + outfile + '.png')
    final_plt.show()

    plt.scatter(count_list, col_frac_list, s = 8)
    plt.xlabel('Cycle')
    plt.ylabel('Colloid Packing Fraction')
    plt.savefig('PackFrac_' + outfile + '.png')
    plt.show()

    bin_plt = binner(1000, col_frac_list, cycles, M)
    bin_plt.savefig('PPack_' + outfile + '.png')
    bin_plt.show()


def system_plot(colloid_centres, polymer_centres, L):

    x_col_data = []
    y_col_data = []
    z_col_data = []

    x_pol_data = []
    y_pol_data = []
    z_pol_data = []

    for i in colloid_centres:
        x_col_data.append(i[0])
        y_col_data.append(i[1])
        z_col_data.append(i[2])

    for i in polymer_centres:
        x_pol_data.append(i[0])
        y_pol_data.append(i[1])
        z_pol_data.append(i[2])

    if dim == 2:
        col_coords = zip(x_col_data, y_col_data)
        col_patches = [plt.Circle(center, R) for center in col_coords]

        pol_coords = zip(x_pol_data, y_pol_data)
        pol_patches = [plt.Circle(center, r) for center in pol_coords]

        fig, ax = plt.subplots()
        col_coll = matplotlib.collections.PatchCollection(col_patches, facecolor = 'black')
        pol_coll = matplotlib.collections.PatchCollection(pol_patches, facecolor = 'red')
        ax.add_collection(col_coll)
        ax.add_collection(pol_coll)
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_aspect('equal')

        return plt


def binner(num, frac_list, N, M):

    X = np.linspace(0, 1, num)
    X_len = len(X)

    midpoint_list = []
    bin_list = []

    for i in range(0, X_len-1):
        midpoint = (X[i] + X[i+1])/2
        bin_count = 0
        for j in frac_list:
            if j >= X[i] and j < X[i+1]:
                bin_count += 1

        normalised_bin = float(bin_count)/(N/M)
        midpoint_list.append(midpoint)
        bin_list.append(normalised_bin)

    plt.scatter(midpoint_list, bin_list, s = 8)
    plt.xlabel('Colloid Packing Fraction')
    plt.ylabel('P of Colloid Packing Fraction Appearing')

    return plt




if __name__ == '__main__':

    outfile = sys.argv[1]
    GC_monte_carlo(cycles = 5000000, n_c = 100, n_p = 100, res_pack_frac = 0.4, z_c = 5.0, M = 100, plotting_mode = 0)












