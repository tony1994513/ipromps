#!/usr/bin/python
import numpy as np
import os
import ConfigParser
import matplotlib.pyplot as plt
from sklearn.externals import joblib

# the current file path
file_path = os.path.dirname(__file__)

# read models cfg file
cp_models = ConfigParser.SafeConfigParser()
cp_models.read(os.path.join(file_path, '../cfg/models.cfg'))
# read models params
datasets_path = os.path.join(file_path, cp_models.get('datasets', 'path'))
len_norm = cp_models.getint('datasets', 'len_norm')
num_basis = cp_models.getint('basisFunc', 'num_basisFunc')
sigma_basis = cp_models.getfloat('basisFunc', 'sigma_basisFunc')
datasets_norm_preproc = joblib.load(os.path.join(datasets_path, 'pkl/datasets_norm_preproc.pkl'))

num_basis = 12

y_t = datasets_norm_preproc[0][0]['left_joints'][:,1]

x_grid = np.linspace(0, 1, len_norm)
c_grid = np.arange(0,num_basis)/(num_basis-1.0)

basis_fuc = np.exp(-.5*(np.array(map(lambda x: x-c_grid, np.tile(x_grid, (num_basis, 1)).T)).T**2 / (sigma_basis**2)))

w = np.dot(np.linalg.inv(np.dot(basis_fuc, basis_fuc.T)), np.dot(basis_fuc, y_t.T)).T

y_reg = np.dot(w, basis_fuc)

fig = plt.figure(0)
plt.plot(x_grid, y_t, 'o', markersize=5, label='motion trajectory', color='r')
plt.plot(x_grid, y_reg, linewidth=3, alpha=1, label='regression curve', color='black')
plt.bar(c_grid, w, width=0.05, edgecolor='w', alpha=0.5)
# x_grid = np.linspace(0, 1.5, len_norm)
# c_grid = np.arange(0,num_basis)/(num_basis-1.0)
# basis_fuc = np.exp(-.5*(np.array(map(lambda x: x-c_grid, np.tile(x_grid, (num_basis, 1)).T)).T**2 / (sigma_basis**2)))
plt.plot(x_grid, basis_fuc.T, linewidth=2, alpha=0.5, color='black')

plt.ylim((0, 1.5))
plt.xlim((0, 1.1))
plt.xlabel('t(s)')
plt.ylabel('y_reg(m)')

plt.legend()
plt.show()