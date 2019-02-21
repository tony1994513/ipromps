#!/usr/bin/python
import numpy as np
import pandas as pd
import os
from sklearn.externals import joblib
import ConfigParser
import ipdb


def main(method="promp",min_max_scaler=None,emg_index=None,
         leftHand_index=None,leftJoint_index=None, datasets_path=None):
    
    # read csv file
    csv_path = os.path.join(datasets_path, 'info/noise/multiModal_states.csv')
    data = pd.read_csv(csv_path)
    # extract the all signals data ORIENTATION
    emg = data.values[:, 7:15].astype(float)
    left_hand = data.values[:, leftHand_index[0]:leftHand_index[-1]].astype(float)
    left_joints = data.values[:, leftJoint_index[0]:leftJoint_index[-1]].astype(float)  # robot ee actually

    if method == "promp":
        full_data = left_joints[1200:, :]
        full_data = min_max_scaler.transform(left_joints)

    elif method == "emg_ipromp":
        full_data = np.hstack([left_hand, emg, left_joints])[1200:, :]
        full_data = min_max_scaler.transform(full_data)

    elif method =="ipromp":
        full_data = np.hstack([left_hand, left_joints])[1200:, :]
        full_data = min_max_scaler.transform(full_data)
        
    # compute the noise observation covariance matrix
    noise_cov = np.cov(full_data.T)
    # save it in pkl
    joblib.dump(noise_cov, os.path.join(datasets_path, 'pkl/'+method+'_noise_cov.pkl'))
    print('Saved the noise covariance matrix successfully!')


if __name__ == '__main__':
    main()
