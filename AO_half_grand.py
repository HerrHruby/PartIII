#! /usr/bin/env python2.7

import numpy as np
import sys
import random
import copy
import matplotlib.pyplot as plt
import AO_model_MC
import AO_full_grand
import time
import pickle

R = 1
r = 1
n_c = 108
n_p = 108
dim = 2
L = 41
precision = 1000000000000
amplitude = 0.5

if dim == 2:
    V = L**2


def canonical_move(col_list, pol_list):

    move = (-0.5*amplitude) + amplitude*(random.uniform(0,1))
    random_colloid = random.randint(0, n_c-1)

    if dim == 3:
        random_coordinate = random.randint(0,2)

    else:
        random_coordinate = random.randint(0,1)

    print 'Move: {}'.format(move)

    accept_count = 0

    update_col_list = copy.deepcopy(col_list)
    new_coordinate = update_col_list[random_colloid][random_coordinate] + move 

    if new_coordinate >= L:
        new_coordinate = new_coordinate - L
    elif new_coordinate < 0:
        new_coordinate = new_coordinate + L
    else:
        pass

    update_col_list[random_colloid][random_coordinate] = new_coordinate

    switch = True

    if new_coordinate > L or new_coordinate < 0:
        print 'Move rejected: attempt to move colloid outside the box'
        switch = False

    if switch:
        for i in update_col_list:
            x, y, z = AO_model_MC.vec_difference(i, update_col_list[random_colloid], L)
            dist = AO_model_MC.pythagoras(x, y, z)
            if dist < 2*R and dist != 0.0:
                print 'Move rejected: moved colloid overlaps with another colloid'
                switch = False
                break

    if switch:
        for i in pol_list:
            x, y, z = AO_model_MC.vec_difference(i, update_col_list[random_colloid], L)
            dist = AO_model_MC.pythagoras(x, y, z)
            if dist < R + r:
                print 'Move rejected: inserted colloid overlaps with a polymer'
                switch = False
                break

    if switch:
        print 'Move accepted'
        accept_count += 1
        return update_col_list, accept_count

    else:
        return col_list, accept_count


def grand_move(col_list, pol_list, beta_mu):

    random_bool = random.random()
    N = len(pol_list)

    insert_count = 0
    remove_count = 0

    if random_bool < 0.5: 
        print 'Attempting to insert a polymer'
        P = (V*(np.exp(beta_mu)))/(N+1) 
        uniform_no = random.uniform(0,1)

        if P >= uniform_no:
            # insert polymer
            random_coord = []

            unscaled_coord_x = random.randint(0, L*precision)
            unscaled_coord_y = random.randint(0, L*precision)
            random_coord_x = float(unscaled_coord_x)/precision
            random_coord_y = float(unscaled_coord_y)/precision

            random_coord.append(random_coord_x)
            random_coord.append(random_coord_y)
            random_coord.append(0)

            update_pol_list = copy.deepcopy(pol_list)
            update_pol_list.append(random_coord)

            if col_list:
                switch = True

                for i in col_list:
                    x, y, z = AO_model_MC.vec_difference(i, random_coord, L)
                    dist = AO_model_MC.pythagoras(x, y, z)
                    if dist < R + r:
                        switch = False
                        print 'Move rejected: inserted polymer overlaps with colloid'
                        break

                if switch:
                    print 'Move accepted with P = {}'.format(P)
                    insert_count += 1
                    return update_pol_list, insert_count, remove_count
                else:
                    return pol_list, insert_count, remove_count

            else:
                print 'Move accepted with P = {}'.format(P)
                insert_count += 1
                return update_pol_list, insert_count, remove_count

        else:
            print 'Move rejected with P = {}'.format(P)
            return pol_list, insert_count, remove_count

    else:
        print 'Attempting to remove a polymer'
        P = (N*(np.exp(-beta_mu)))/V
        uniform_no = random.uniform(0,1)
        
        if P >= uniform_no:
            # remove polymer
            random_polymer = random.randint(0, N-1)
            update_pol_list = copy.deepcopy(pol_list)
            update_pol_list.pop(random_polymer)
            print 'Move accepted with P = {}'.format(P)
            remove_count += 1
            return update_pol_list, insert_count, remove_count
        
        else:
            print 'Move rejected with P = {}'.format(P)
            return pol_list, insert_count, remove_count


def simulation(cycles, beta_mu, test_plot = 0):

    colloid_centres, init_dist_list = AO_model_MC.generate_system(L, R, n_c, r, precision, dim)
    polymer_centres = AO_full_grand.generate_polymers(L, R, n_p, r, precision, colloid_centres)

    init_plt = AO_full_grand.system_plot(colloid_centres, polymer_centres, L)
    init_plt.savefig('Init_' + outfile + '.png')
    init_plt.show()

    total_accept_count = 0
    total_insert_count = 0
    total_remove_count = 0

    total_pols_list = []
    count_list = []

    savepoint = 0.7*cycles
    T = np.linspace(savepoint, cycles, 31)
    saved_colloid_list = []

    start_time = time.time()

    for i in range(0, cycles):

        print 'Cycle: {}'.format(i)

        if i % 100 == 0:
            total_pols = n_p + total_insert_count - total_remove_count
            total_pols_list.append(total_pols)
            count_list.append(i)

        if i in T:
            print 'Colloid configuration saved'
            saved_colloid_list.append(colloid_centres)

        if i % 2 == 0:
            print 'Attempting a colloid move'
            colloid_centres, accept_count = canonical_move(colloid_centres, polymer_centres)
            total_accept_count += accept_count
            print 'Colloid move accept count: {}'.format(total_accept_count)
            if test_plot:
                test_plt = AO_full_grand.system_plot(colloid_centres, polymer_centres, L)
                test_plt.show()

        else:
            polymer_centres, insert_count, remove_count = grand_move(colloid_centres, polymer_centres, beta_mu)
            total_insert_count += insert_count
            total_remove_count += remove_count

            print 'Polymer insert count: {}'.format(total_insert_count)
            print 'Polymer remove count: {}'.format(total_remove_count)
            if test_plot: 
                test_plt = AO_full_grand.system_plot(colloid_centres, polymer_centres, L)
                test_plt.show()

        print '--------------------------------------------------'

    print 'Run completed in {} seconds'.format(time.time() - start_time)

    final_pol_count = n_p + total_insert_count - total_remove_count
    print 'Final Polymer Count: {}'.format(final_pol_count)

    outfile_col_info = outfile + '_col_info'
    with open(outfile_col_info, 'wb') as fp:
        pickle.dump(saved_colloid_list, fp)

    final_plt = AO_full_grand.system_plot(colloid_centres, polymer_centres, L)
    final_plt.savefig('Final_' + outfile + '.png')
    final_plt.show()

    plt.scatter(count_list, total_pols_list, s = 8)
    plt.xlabel('Cycles')
    plt.ylabel('Total polymers in system')
    plt.savefig('Polycount_' + outfile + '.png')
    plt.show()


def ideal_simulation(cycles, beta_mu, test_plot = 0):

    colloid_centres = []
    polymer_centres = AO_full_grand.generate_polymers(L, R, n_p, r, precision, colloid_centres)

    if test_plot:
        init_plt = AO_full_grand.system_plot(colloid_centres, polymer_centres, L)
        init_plt.savefig('Init_' + outfile + '.png')
        init_plt.show()

    total_accept_count = 0
    total_insert_count = 0
    total_remove_count = 0

    total_pols_list = []
    count_list = []

    savepoint = 0.8*cycles
    T = np.linspace(savepoint, cycles, 41)
    saved_count_list = []

    start_time = time.time()

    for i in range(0, cycles):

        print 'Cycle: {}'.format(i)

        total_pols = n_p + total_insert_count - total_remove_count
        
        if i in T:
            print 'Colloid configuration saved'
            saved_count_list.append(total_pols)

        if i % 100 == 0 and test_plot:
            total_pols_list.append(total_pols)
            count_list.append(i)

        polymer_centres, insert_count, remove_count = grand_move(colloid_centres, polymer_centres, beta_mu)
        total_insert_count += insert_count
        total_remove_count += remove_count

        print 'Polymer insert count: {}'.format(total_insert_count)
        print 'Polymer remove count: {}'.format(total_remove_count)

        print '-------------------------------------------------'

    print 'Run completed in {} seconds'.format(time.time() - start_time)

    avg_pol_count = float(sum(saved_count_list))/len(saved_count_list)
    print 'Final Polymer Count: {}'.format(avg_pol_count)

    if test_plot:
        final_plt = AO_full_grand.system_plot(colloid_centres, polymer_centres, L)
        final_plt.savefig('Final_' + outfile + '.png')
        final_plt.show()

        plt.scatter(count_list, total_pols_list, s = 8)
        plt.xlabel('Cycles')
        plt.ylabel('Total polymers in system')
        plt.savefig('Polycount_' + outfile + '.png')
        plt.show()

    return avg_pol_count


def run_ideal_simulation(cycles, trials, start, end, steps):

    X = np.linspace(start, end, steps+1)

    avg_list = []
    std_list = []
    true_list = []

    for i in X:
        true_count = i*V
        true_list.append(true_count)

        beta_mu = np.log(i)
        pol_count_list = []
        
        for j in range(0, trials):
            avg_pol_count = ideal_simulation(cycles, beta_mu, test_plot = 0)
            pol_count_list.append(avg_pol_count)

        avg_count = float(sum(pol_count_list))/len(pol_count_list)
        avg_list.append(avg_count)
        std_count = np.std(pol_count_list)
        std_list.append(std_count)
    
    plt.errorbar(X, avg_list, yerr = std_list, fmt = 'o', label = "Simulation")
    plt.scatter(X, true_list, marker = "x", c = "r", label = "Calculation")
    plt.xlabel('Fugacity')
    plt.ylabel('Polymer Count')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    outfile = sys.argv[1]
    #run_ideal_simulation(40000, 10, 0, 1.2, 6)
    simulation(4000000, -0.693) 


