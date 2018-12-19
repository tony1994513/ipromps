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

num_basis = 8

ex = datasets_norm_preproc[0][0]['left_joints'][:,1]


fig = plt.figure(0)
x_grid = np.linspace(0, 1.5, len_norm)
c_grid = np.arange(0,num_basis)/(num_basis-1.0)
h = np.exp(-.5*(np.array(map(lambda x: x-c_grid, np.tile(x_grid, (num_basis, 1)).T)).T**2 / (sigma_basis**2)))
plt.plot(x_grid, h.T, linewidth=2, color='black')
plt.ylim((0, 1.3))
plt.xlim((0, 1.2))
plt.xlabel('t(s)')
plt.ylabel('y(m)')
plt.grid(True)
plt.legend()

x_grid = np.linspace(0, 2.0, len_norm)
c_grid = np.arange(0,num_basis)/(num_basis-1.0)*1.3
fig = plt.figure(1)
h = np.exp(-.5*(np.array(map(lambda x: x-c_grid, np.tile(x_grid/1.3, (num_basis, 1)).T)).T**2 / (sigma_basis**2)))
plt.plot(x_grid, h.T, linewidth=2, color='black')
plt.ylim((0, 1.3))
plt.xlim((0, 2.0))
plt.xlabel('t(s)')
plt.ylabel('y(m)')
plt.grid(True)
plt.legend()

plt.show()