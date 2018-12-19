#!/usr/bin/python
# data structure is: list(task1,2...)-->list(demo1,2...)-->dict(emg,imu,tf...)
# left_hand_csv_idx = [207,208,209]
# left_joints_csv_idx = [99,100,101,102,103,104,105]
# left_joints_csv_idx = [317,318,319,320,321,322,323]

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from sklearn.externals import joblib
import glob
import os
import ConfigParser
from sklearn import preprocessing
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import griddata
import ipdb
# the current file path
file_path = os.path.dirname(__file__)

# read models cfg file
cp_models = ConfigParser.SafeConfigParser()
cp_models.read(os.path.join(file_path, '../cfg/models.cfg'))
# read datasets params
datasets_path = os.path.join(file_path, cp_models.get('datasets', 'path'))
len_norm = cp_models.getint('datasets', 'len_norm')
num_demo = cp_models.getint('datasets', 'num_demo')
num_dim = cp_models.getint('datasets', 'num_dim')

# read filter params
sigma = cp_models.getint('filter', 'sigma')

# read csv params
emg_index = cp_models.get('csv_parse', 'emg')
leftHand_index = cp_models.get('csv_parse', 'left_hand')
leftJoint_index = cp_models.get('csv_parse', 'left_joints')

# process csv params
emg_index =  [int(x.strip()) for x in emg_index.split(',')]
leftHand_index =  [int(x.strip()) for x in leftHand_index.split(',')]
leftJoint_index =  [int(x.strip()) for x in leftJoint_index.split(',')]

# read datasets cfg file
cp_datasets = ConfigParser.SafeConfigParser()
cp_datasets.read(os.path.join(datasets_path, './info/cfg/datasets.cfg'))
#read datasets params
data_index_sec = cp_datasets.items('index_17')
data_index = [map(int, task[1].split(',')) for task in data_index_sec]

# after processed human/robot index
human_index = cp_models.get('processed_index', 'human')
robot_index = cp_models.get('processed_index', 'robot')
human_index =  [int(x.strip()) for x in human_index.split(',')]
robot_index =  [int(x.strip()) for x in robot_index.split(',')]


human_dim = leftHand_index[-1] - leftHand_index[0]
robot_dim = leftJoint_index[-1] - leftJoint_index[0]


print("dataset path : %s " %datasets_path)
print("length of norm : %s " %len_norm)
print("number of demo : %s " %num_demo)
print("number of dimension : %s" %num_dim)
print("filter sigma : %s" %sigma)
print("-----------")
print("emg index : %s" %emg_index)
print("human csv index : %s" %leftHand_index)
print("robot csv  index : %s" %leftJoint_index)
print("human dimension : %s" %human_dim)
print("robot dimension : %s" %robot_dim)
print("-----------")
print("human index: %s" %human_index )
print("robot index: %s" %robot_index )
print("-----------")

def filter_static_points(mat):
    last = mat[0] # for refercence
    new_mat = [last]
    for idx in range(mat.shape[0]):
        if np.linalg.norm(mat[idx]-last)<0.005:
            pass
        else:
            new_mat.append(mat[idx])
            last = mat[idx]
    return np.array(new_mat)


def main():
    # datasets-related info
    
    task_path_list = glob.glob(os.path.join(datasets_path, 'raw/*'))
    task_path_list =sorted(task_path_list)
    task_name_list = [task_path.split('/')[-1] for task_path in task_path_list]

    # load raw datasets
    datasets_raw = []
    for task_path in task_path_list:
        left_hand_path = os.path.join(task_path, 'left_hand')
        left_joints_path = os.path.join(task_path, 'left_joints')
        left_hand_csv_path = os.path.join(left_hand_path, 'csv')
        left_joints_csv_path = os.path.join(left_joints_path, 'csv')
        print('Loading data from: ' + left_hand_csv_path)
        print('Loading data from: ' + left_joints_csv_path)
        
        left_hand_path_list = glob.glob(os.path.join(left_hand_csv_path, '201*'))  
        left_hand_path_list = sorted(left_hand_path_list)
        left_joints_path_list = glob.glob(os.path.join(left_joints_csv_path, '201*'))  
        left_joints_path_list = sorted(left_joints_path_list)

        demo_temp = []

        for left_hand_demo_path, left_joints_demo_path in zip(left_hand_path_list,left_joints_path_list):
            
            left_hand_csv = pd.read_csv(os.path.join(left_hand_demo_path, 'multiModal_states.csv'))  
            left_joints_csv = pd.read_csv(os.path.join(left_joints_demo_path, 'multiModal_states.csv')) 

            left_hand_csv_t = np.array((left_hand_csv.values[:,2] - left_hand_csv.values[0,2])*1e-9).astype("float")
            left_joints_csv_t = np.array((left_joints_csv.values[:,2] - left_joints_csv.values[0,2])*1e-9).astype("float")

            left_hand = left_hand_csv.values[:, leftHand_index[0]:leftHand_index[-1]].astype(float)
            left_joints = left_joints_csv.values[:, leftJoint_index[0]:leftJoint_index[-1]].astype(float)

            lenth = len(left_hand)

            t_end = left_hand_csv_t[-1]
 

            resample_t = np.linspace(0.0, t_end, lenth)
            left_hand_resampled = [None] * human_dim
            left_joints_resampled = [None] * robot_dim
            for idx  in range(left_hand.shape[1]):
                left_hand_resampled[idx] = np.interp(resample_t, left_hand_csv_t, left_hand[:,idx])

            for idx in range(left_joints.shape[1]):
                left_joints_resampled[idx] = np.interp(resample_t, left_joints_csv_t, left_joints[:,idx])
            
            demo_temp.append({
                    'stamp': resample_t,
                    'left_hand': np.array(left_hand_resampled).T,
                    'left_joints': np.array(left_joints_resampled).T
                    })
        datasets_raw.append(demo_temp)
    ### load raw data ###
    for 