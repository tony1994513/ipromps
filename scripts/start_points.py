#!/usr/bin/python
# coding:utf-8
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import scipy.stats as stats
import os
import ConfigParser
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties
from sklearn import mixture

# read conf file
file_path = os.path.dirname(__file__)
cp_models = ConfigParser.SafeConfigParser()
cp_models.read(os.path.join(file_path, '../cfg/models.cfg'))
# the datasets path
datasets_path = os.path.join(file_path, cp_models.get('datasets', 'path'))

# read datasets cfg file
cp_datasets = ConfigParser.SafeConfigParser()
cp_datasets.read(os.path.join(datasets_path, 'info/cfg/datasets.cfg'))
# read datasets params
data_index_sec = cp_datasets.items('index_17')
data_index = [map(int, task[1].split(',')) for task in data_index_sec]

# load datasets
datasets_raw = joblib.load(os.path.join(datasets_path, 'pkl/datasets_raw.pkl'))

task_label = ['gummed_paper', 'screw_driver', 'pencil_box', 'measurement_tape']

temp = []
task_marker = ['o', '*', '^', 'd']
for task_idx, demo_idx in enumerate(data_index):
    b = [datasets_raw[task_idx][x]['left_hand'][0,0:3] for x in demo_idx]
    b= np.array(b)
    temp.append(b)

# for task_idx in range(len(data_index)):
    fig = plt.figure(0)
    ax = fig.gca(projection='3d')
    ax.plot(temp[task_idx][:, 0], temp[task_idx][:, 1], temp[task_idx][:, 2], task_marker[task_idx], markersize=12,
            # label='training sets about human '+str(demo_idx), alpha=0.3)
            alpha=1.0, label='Starts point of %s' % task_label[task_idx] + ' task')
    ax.set_xlabel('X(m)')
    ax.set_ylabel('Y(m)')
    ax.set_zlabel('Z(m)')

    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # ax.set_zticks([-2, 0, 2])
    # plt.
    # ax.set_xlabel('X(m)', fontsize=20, rotation=150)
    # ax.set_ylabel('X(m)', fontsize=20)
    # ax.set_zlabel('X(m)', fontsize=30, rotation=60)

    ax.legend(loc=2)
plt.show()
#
# clf = mixture.GaussianMixture(n_components=1, covariance_type='full')
# clf.fit(temp[1])
# print clf.means_
# print clf.covariances_
