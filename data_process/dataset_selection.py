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
num_demo = cp_models.getint('datasets', 'num_demo')

# load datasets
ipromps_set = joblib.load(os.path.join(datasets_path, 'pkl/ipromps_set.pkl'))
datasets_raw = joblib.load(os.path.join(datasets_path, 'pkl/datasets_raw.pkl'))
datasets_filtered = joblib.load(os.path.join(datasets_path, 'pkl/datasets_filtered.pkl'))
task_name = joblib.load(os.path.join(datasets_path, 'pkl/task_name_list.pkl'))



# read datasets cfg file
cp_datasets = ConfigParser.SafeConfigParser()
cp_datasets.read(os.path.join(datasets_path, 'info/cfg/datasets.cfg'))
# read datasets params
data_index_sec = cp_datasets.items('index_15')
data_index = [map(int, task[1].split(',')) for task in data_index_sec]

# the idx of interest info in data structure
info_n_idx = {
            'left_hand': [0, 3],
            'left_joints': [3, 6]
            }
# the info to be plotted
info = cp_models.get('visualization', 'info')
joint_num = info_n_idx[info][1] - info_n_idx[info][0]
num_obs = cp_models.getint('visualization', 'num_obs')


# zh config
def conf_zh(font_name):
    from pylab import mpl
    mpl.rcParams['font.sans-serif'] = [font_name]
    mpl.rcParams['axes.unicode_minus'] = False


# plot the raw data index
def plot_raw_data_index(num=0):
    for task_idx, demo_list in enumerate(data_index):
        for demo_idx in demo_list:
            fig = plt.figure(num + task_idx)
            fig.suptitle('the raw data of ' + info)
            for joint_idx in range(joint_num):
                ax = fig.add_subplot(joint_num, 1, 1 + joint_idx)
                data = datasets_raw[task_idx][demo_idx][info][:, joint_idx]
                plt.plot(range(len(data)), data, label=str(demo_idx))
                plt.legend()


# plot the filter data index
def plot_filter_data_index(num=0):
    for task_idx, demo_list in enumerate(data_index):
        for demo_idx in demo_list:
            fig = plt.figure(num + task_idx)
            fig.suptitle('the raw data of ' + info)
            for joint_idx in range(joint_num):
                ax = fig.add_subplot(joint_num, 1, 1 + joint_idx)
                data = datasets_filtered[task_idx][demo_idx][info][:, joint_idx]
                plt.plot(range(len(data)), data, label=str(demo_idx))
                ax.legend()


def main():
    conf_zh("Droid Sans Fallback")
    # plt.close('all')
    # plot_alpha()
    # plot_robot_traj()
    # plot_raw_data_index()
    plot_filter_data_index(20)

    plt.show()


if __name__ == '__main__':
    main()
