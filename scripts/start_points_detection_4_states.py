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
from sklearn import svm

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

task_list = []
task_marker = ['o', '*', '^', 'd']
demo_data_full = np.array([]).reshape(0, 3)
for task_idx, demo_idx in enumerate(data_index):
    b = [datasets_raw[task_idx][x]['left_hand'][0,0:3] for x in demo_idx]
    b= np.array(b)
    task_list.append(b)
    # demo_data_full = np.vstack([demo_data_full, b])
# print demo_data_full
test_data = np.array( datasets_raw[0][0]['left_hand'][:,0:3])
clf = svm.OneClassSVM( kernel="rbf", gamma=0.1)
clf.fit(task_list[0])
y_pred_train = clf.predict(task_list[0])
test_pred = clf.predict(test_data)
n_error_train = y_pred_train[y_pred_train == -1].size
print n_error_train