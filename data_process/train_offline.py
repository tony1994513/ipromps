#!/usr/bin/python
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

import load_data
import train_models
import noise_cov_cal

# the current file path
file_path = os.path.dirname(__file__)

method = "promp"
# method = "emg_ipromp"
# method = "ipromp"

# read models cfg file
cp_models = ConfigParser.SafeConfigParser()
cp_models.read(os.path.join(file_path, '../cfg/models.cfg'))

# read models params
datasets_path = os.path.join(file_path, cp_models.get('datasets', 'path'))
if method == "promp":
    num_dim = cp_models.getint('promp_param', 'num_dim')
    num_obs_dim = cp_models.getint('promp_param', 'num_obs_dim')
elif method == "ipromp":
    num_dim = cp_models.getint('ipromp_param', 'num_dim')
    num_obs_dim = cp_models.getint('ipromp_param', 'num_obs_dim')
elif method == "emg_ipromp" :
    num_dim = cp_models.getint('emg_ipromp_param', 'num_dim')
    num_obs_dim = cp_models.getint('ipromp_param', 'num_obs_dim')

# read csv params
emg_index = cp_models.get('csv_parse', 'emg')
leftHand_index = cp_models.get('csv_parse', 'left_hand')
leftJoint_index = cp_models.get('csv_parse', 'left_joints')
emg_index =  [int(x.strip()) for x in emg_index.split(',')]
leftHand_index =  [int(x.strip()) for x in leftHand_index.split(',')]
leftJoint_index =  [int(x.strip()) for x in leftJoint_index.split(',')]

len_norm = cp_models.getint('datasets', 'len_norm')
num_basis = cp_models.getint('basisFunc', 'num_basisFunc')
sigma_basis = cp_models.getfloat('basisFunc', 'sigma_basisFunc')
num_alpha_candidate = cp_models.getint('phase', 'num_phaseCandidate')
num_demo = cp_models.getint('datasets', 'num_demo')
sigma = cp_models.getint('filter', 'sigma')

# read datasets cfg file
cp_datasets = ConfigParser.SafeConfigParser()
cp_datasets.read(os.path.join(datasets_path, './info/cfg/datasets.cfg'))
#read datasets params
data_index_sec = cp_datasets.items('index_17')
data_index = [map(int, task[1].split(',')) for task in data_index_sec]

# the pkl data path
datasets_pkl_path = os.path.join(datasets_path, 'pkl')
min_max_scaler_path = os.path.join(datasets_pkl_path, 'min_max_scaler.pkl')
min_max_scaler = joblib.load(min_max_scaler_path)

task_path_list = sorted(glob.glob(os.path.join(datasets_path, 'raw/*')))

def main():
    print("-----------")
    print('## Running the %s' % load_data.__name__)
    load_data.main(num_dim=num_dim, sigma=sigma, method=method, num_demo=num_demo,data_index=data_index, 
                   len_norm=len_norm, leftHand_index=leftHand_index, leftJoint_index=leftJoint_index,
                    task_path_list=task_path_list, datasets_path=datasets_path)
    print("-----------")
    print('## Running the %s' % noise_cov_cal.__name__)
    noise_cov_cal.main(method=method,min_max_scaler=min_max_scaler,emg_index=emg_index,
                       leftHand_index=leftHand_index,leftJoint_index=leftJoint_index, datasets_path=datasets_path)
    print("-----------")
    print('## Running the %s' % train_models.__name__)
    train_models.main(num_dim=num_dim, num_obs_dim=num_obs_dim, num_basis=num_basis,
                      sigma_basis=sigma_basis, len_norm=len_norm, min_max_scaler=min_max_scaler,
                      num_alpha_candidate=num_alpha_candidate,
                      method=method, datasets_pkl_path=datasets_pkl_path)

if __name__ == '__main__':
    main()
