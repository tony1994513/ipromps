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
from sklearn import preprocessing
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import griddata
import ipdb

def main(num_dim=7, sigma=5, len_norm=101, data_index=None,method="promp",num_demo=19,
         leftHand_index=None, leftJoint_index=None,task_path_list=None, datasets_path=None):
    # datasets-related info

    task_name_list = [task_path.split('/')[-1] for task_path in task_path_list]

    human_dim = leftHand_index[-1] - leftHand_index[0]
    robot_dim = leftJoint_index[-1] - leftJoint_index[0]
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
            # ipdb.set_trace()
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
            

 # filter the datasets: gaussian_filter1d
    datasets_filtered = []
    for task_idx, task_data in enumerate(datasets_raw):
        print('Filtering data of task: ' + task_name_list[task_idx])
        demo_norm_temp = []
        for demo_data in task_data:
            time_stamp = demo_data['stamp']
            # filter the datasets
            left_hand_filtered = gaussian_filter1d(demo_data['left_hand'].T, sigma=sigma).T
            left_joints_filtered = gaussian_filter1d(demo_data['left_joints'].T, sigma=sigma).T
            # append them to list
            demo_norm_temp.append({
                'alpha': time_stamp[-1],
                'left_hand': left_hand_filtered,
                'left_joints': left_joints_filtered
            })
        datasets_filtered.append(demo_norm_temp)

    # resample the datasets
    datasets_norm = []
    for task_idx, task_data in enumerate(datasets_raw):
        print('Resampling data of task: ' + task_name_list[task_idx])
        demo_norm_temp = []
        for demo_data in task_data:
            time_stamp = demo_data['stamp']
            grid = np.linspace(0, time_stamp[-1], len_norm)
            # filter the datasets
            left_hand_filtered = gaussian_filter1d(demo_data['left_hand'].T, sigma=sigma).T
            left_joints_filtered = gaussian_filter1d(demo_data['left_joints'].T, sigma=sigma).T

            # normalize the datasets
            left_hand_norm = griddata(time_stamp, left_hand_filtered, grid, method='linear')
            left_joints_norm = griddata(time_stamp, left_joints_filtered, grid, method='linear')
            # append them to list
            demo_norm_temp.append({
                                    'alpha': time_stamp[-1],
                                    'left_hand': left_hand_norm,
                                    'left_joints': left_joints_norm
                                    })
        datasets_norm.append(demo_norm_temp)

    # preprocessing for the norm data
    datasets4train = []
    for task_idx, demo_list in enumerate(data_index):
        data = [datasets_norm[task_idx][i] for i in demo_list]
        datasets4train.append(data)
    y_full = np.array([]).reshape(0, num_dim)
    for task_idx, task_data in enumerate(datasets4train):
        print('Preprocessing data for task: ' + task_name_list[task_idx])
        for demo_data in task_data:
            if method == "promp":
                h = demo_data['left_joints']
                y_full = np.vstack([y_full, h])
            elif method == "ipromp" or "emg_ipromp":
                h = np.hstack([demo_data['left_hand'], demo_data['left_joints']])
                y_full = np.vstack([y_full, h])
    min_max_scaler = preprocessing.MinMaxScaler()
    datasets_norm_full = min_max_scaler.fit_transform(y_full)
    # construct a data structure to train the model
    datasets_norm_preproc = []
    for task_idx in range(len(datasets4train)):
        datasets_temp = []
        # ipdb.set_trace()
        for demo_idx in range(num_demo):
            temp = datasets_norm_full[(task_idx * num_demo + demo_idx) * len_norm:
            (task_idx * num_demo + demo_idx) * len_norm + len_norm, :]
            if method == "ipromp":
                datasets_temp.append({
                                        'left_hand': temp[:, human_index[0]:human_index[-1]],
                                        'left_joints': temp[:, robot_index[0]:robot_index[-1]],
                                        'alpha': datasets4train[task_idx][demo_idx]['alpha']})
            if method =="promp":
                datasets_temp.append({'left_joints': temp,
                                      'alpha': datasets4train[task_idx][demo_idx]['alpha']})                                        
        datasets_norm_preproc.append(datasets_temp)

    # save all the datasets
    print('Saving the datasets as pkl ...')
    joblib.dump(task_name_list, os.path.join(datasets_path, 'pkl/task_name_list.pkl'))
    joblib.dump(datasets_raw, os.path.join(datasets_path, 'pkl/datasets_raw.pkl'))
    joblib.dump(datasets_filtered, os.path.join(datasets_path, 'pkl/datasets_filtered.pkl'))
    joblib.dump(datasets_norm, os.path.join(datasets_path, 'pkl/datasets_norm.pkl'))
    joblib.dump(datasets_norm_preproc, os.path.join(datasets_path, 'pkl/datasets_norm_preproc.pkl'))
    joblib.dump(min_max_scaler, os.path.join(datasets_path, 'pkl/min_max_scaler.pkl'))

    # the finished reminder
    print('Loaded, filtered, normalized, preprocessed and saved the datasets successfully!!!')


if __name__ == '__main__':
    main()
