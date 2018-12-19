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


# read conf file
file_path = os.path.dirname(__file__)
cp_models = ConfigParser.SafeConfigParser()
cp_models.read(os.path.join(file_path, '../cfg/models.cfg'))
# the datasets path
datasets_path = os.path.join(file_path, cp_models.get('datasets', 'path'))

datasets_raw = joblib.load(os.path.join(datasets_path, 'pkl/datasets_raw.pkl'))
datasets_norm = joblib.load(os.path.join(datasets_path, 'pkl/datasets_norm.pkl'))
datasets_filtered = joblib.load(os.path.join(datasets_path, 'pkl/datasets_filtered.pkl'))
datasets_norm_preproc = joblib.load(os.path.join(datasets_path, 'pkl/datasets_norm_preproc.pkl'))
task_name = joblib.load(os.path.join(datasets_path, 'pkl/task_name_list.pkl'))


# read datasets cfg file
cp_datasets = ConfigParser.SafeConfigParser()
cp_datasets.read(os.path.join(datasets_path, 'info/cfg/datasets.cfg'))
# read datasets params
data_index_sec = cp_datasets.items('index_17')
data_index = [map(int, task[1].split(',')) for task in data_index_sec]

# after processed human/robot index
human_index = cp_models.get('processed_index', 'human')
robot_index = cp_models.get('processed_index', 'robot')
human_index =  [int(x.strip()) for x in human_index.split(',')]
robot_index =  [int(x.strip()) for x in robot_index.split(',')]

info_n_idx = {
            'left_hand': human_index,
            'left_joints': robot_index,
            }
# the info to be plotted
info = cp_models.get('visualization', 'info')
joint_num = info_n_idx[info][1] - info_n_idx[info][0]

# plot the 3d raw traj
def plot_3d_raw_traj(num=0):
    for task_idx, demo_list in enumerate(data_index):
        fig = plt.figure(task_idx + num)
        ax = fig.gca(projection='3d')
        for demo_idx in demo_list:
            data = datasets_filtered[task_idx][demo_idx][info]
            ax.plot(data[:, 0], data[:, 1], data[:, 2], label=str(demo_idx))
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            ax.legend()

def main():
    plot_3d_raw_traj(10)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.show()


if __name__ == '__main__':
    main()