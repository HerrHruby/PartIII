#! /usr/bin/env python2.7

import numpy as np
import pandas as pd
import AO_remainder
import matplotlib.pyplot as plt
import sys
import pickle
from mpl_toolkits.mplot3d import Axes3D



f = sys.argv[1]

f_csv_3 = f + '_3.csv'
f_pred = f + '_pred'
f_test = f + '_test'
f_X = f + '_X'
f_Y = f + '_Y'
f_Xtrain = f + '_Xtrain'

df = pd.read_csv(f_csv_3)
column_count = len(df.columns) -2
cdf = df.iloc[:,:column_count:]
dist_list = cdf.values.tolist()
dist_arr = np.array(dist_list)
vol_arr = np.array(df.Vol)

"""
gp_n_interact, X_train_n, X_test_n, Y_train_n, Y_test_n, dist_arr, n_body_list, Y_pred, Y_std = AO_remainder.AO_3body_remainder(f, plot = 0)


with open(f_pred, 'wb') as fp:
    pickle.dump(Y_pred, fp)

with open(f_test, 'wb') as fp:
    pickle.dump(Y_test_n, fp)

with open(f_X, 'wb') as fp:
    pickle.dump(X_test_n, fp)

with open(f_Y, 'wb') as fp:
    pickle.dump(Y_train_n, fp)

with open(f_Xtrain, 'wb') as fp:
    pickle.dump(X_train_n, fp)
"""


with open(f_pred, 'rb') as fp:
    Y_pred = pickle.load(fp)

with open(f_test, 'rb') as fp:
    Y_test_n = pickle.load(fp)

with open(f_X, 'rb') as fp:
    X_test_n = pickle.load(fp)

with open(f_Y, 'rb') as fp:
    Y_train_n = pickle.load(fp)

with open(f_Xtrain, 'rb') as fp:
    X_train_n = pickle.load(fp)


big_susp = []

susp_x = []
susp_y = []
susp_z = []
susp_vol_pred = []
susp_vol_test = []

for i, j, k in zip(Y_pred, Y_test_n, X_test_n):
    if i >= j + 0.08:
        k_list = list(k)
        susp_x.append(k_list[0])
        susp_y.append(k_list[1])
        susp_z.append(k_list[2])
        susp_vol_test.append(j)
        susp_vol_pred.append(i)

x_coords = []
y_coords = []
z_coords = []

for i in X_train_n:
    x_coords.append(i[0])
    y_coords.append(i[1])
    z_coords.append(i[2])

fig = plt.figure()
ax = Axes3D(fig)


ax.scatter(x_coords, y_coords, z_coords, alpha = 0.2)
ax.scatter(susp_x, susp_y, susp_z, c = 'r')

ax.set_xlabel('short')
ax.set_ylabel('medium')
ax.set_zlabel('long')

plt.show()

plt.scatter(x_coords, Y_train_n, alpha = 0.3, s = 8)
plt.scatter(y_coords, Y_train_n, alpha = 0.3, s = 8)
plt.scatter(z_coords, Y_train_n, alpha = 0.3, s = 8)
plt.scatter(susp_x, susp_vol_pred)
plt.scatter(susp_y, susp_vol_pred)
plt.scatter(susp_z, susp_vol_pred)

plt.show()








