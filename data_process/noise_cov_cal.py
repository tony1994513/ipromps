#!/usr/bin/python
import numpy as np
import pandas as pd
import os
from sklearn.externals import joblib
import ConfigParser
import ipdb
# read conf file
file_path = os.path.dirname(__file__)
cp_models = ConfigParser.SafeConfigParser()
cp_models.read(os.path.join(file_path, '../cfg/models.cfg'))
# the datasets path
datasets_path = os.path.join(file_path, cp_models.get('datasets', 'path'))
datasets_pkl_path = os.path.join(datasets_path, 'pkl')
min_max_scaler_path = os.path.join(datasets_pkl_path, 'min_max_scaler.pkl')
min_max_scaler = joblib.load(min_max_scaler_path)
# read csv params
emg_index = cp_models.get('csv_parse', 'emg')
leftHand_index = cp_models.get('csv_parse', 'left_hand')
leftJoint_index = cp_models.get('csv_parse', 'left_joints')

# process csv params
emg_index =  [int(x.strip()) for x in emg_index.split(',')]
leftHand_index =  [int(x.strip()) for x in leftHand_index.split(',')]
leftJoint_index =  [int(x.strip()) for x in leftJoint_index.split(',')]

promp = True
ipromp = False
emg_ipromp = False

def main():
    # read csv file
    csv_path = os.path.join(datasets_path, 'info/noise/multiModal_states.csv')
    data = pd.read_csv(csv_path)
    # extract the all signals data ORIENTATION
    emg = data.values[:, 7:15].astype(float)
    left_hand = data.values[:, leftHand_index[0]:leftHand_index[-1]].astype(float)
    left_joints = data.values[:, leftJoint_index[0]:leftJoint_index[-1]].astype(float)  # robot ee actually

    if promp:
        full_data = left_joints[1200:, :]
        full_data = min_max_scaler.transform(left_joints)

    elif emg_ipromp:
        full_data = np.hstack([left_hand, emg, left_joints])[1200:, :]
        full_data = min_max_scaler.transform(full_data)

    elif ipromp:
        full_data = np.hstack([left_hand, left_joints])[1200:, :]
        full_data = min_max_scaler.transform(full_data)

    # compute the noise observation covariance matrix
    noise_cov = np.cov(full_data.T)
    # save it in pkl
    joblib.dump(noise_cov, os.path.join(datasets_path, 'pkl/noise_cov.pkl'))
    print('Saved the noise covariance matrix successfully!')


if __name__ == '__main__':
    main()
