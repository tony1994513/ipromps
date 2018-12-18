#!/usr/bin/python
import numpy as np
import ipromps_lib
from sklearn.externals import joblib
import os
import ConfigParser


# the current file path
file_path = os.path.dirname(__file__)

# read models cfg file
cp_models = ConfigParser.SafeConfigParser()
cp_models.read(os.path.join(file_path, '../cfg/models.cfg'))
# read models params
datasets_path = os.path.join(file_path, cp_models.get('datasets', 'path'))
num_dim = cp_models.getint('datasets', 'num_dim')
num_obs_dim = cp_models.getint('datasets', 'num_obs_dim')
len_norm = cp_models.getint('datasets', 'len_norm')
num_basis = cp_models.getint('basisFunc', 'num_basisFunc')
sigma_basis = cp_models.getfloat('basisFunc', 'sigma_basisFunc')
num_alpha_candidate = cp_models.getint('phase', 'num_phaseCandidate')

# the pkl data
datasets_pkl_path = os.path.join(datasets_path, 'pkl')
task_name_path = os.path.join(datasets_pkl_path, 'task_name_list.pkl')
datasets_norm_preproc_path = os.path.join(datasets_pkl_path, 'datasets_norm_preproc.pkl')
min_max_scaler_path = os.path.join(datasets_pkl_path, 'min_max_scaler.pkl')
noise_cov_path = os.path.join(datasets_pkl_path, 'noise_cov.pkl')


def main():
    # load the data from pkl
    task_name = joblib.load(task_name_path)
    datasets_norm_preproc = joblib.load(datasets_norm_preproc_path)
    min_max_scaler = joblib.load(min_max_scaler_path)
    noise_cov = joblib.load(noise_cov_path)

    # create iProMPs sets
    ipromps_set = [ipromps_lib.IProMP(num_joints=num_dim, num_obs_joints=num_obs_dim, num_basis=num_basis,
                                      sigma_basis=sigma_basis, num_samples=len_norm, sigmay=noise_cov,
                                      min_max_scaler=min_max_scaler, num_alpha_candidate=num_alpha_candidate)
                   for x in datasets_norm_preproc]

    # add demo for each IProMPs
    for idx, ipromp in enumerate(ipromps_set):
        print('Training the IProMP for task: %s...' % task_name[idx])
        # for demo_idx in datasets4train[idx]:
        for demo_idx in datasets_norm_preproc[idx]:
            demo_temp = np.hstack([demo_idx['left_hand'], demo_idx['left_joints']])
            ipromp.add_demonstration(demo_temp)   # spatial variance demo
            ipromp.add_alpha(demo_idx['alpha'])   # temporal variance demo

    # save the trained models
    print('Saving the trained models...')
    joblib.dump(ipromps_set, os.path.join(datasets_pkl_path, 'ipromps_set.pkl'))

    print('Trained the IProMPs successfully!!!')


if __name__ == '__main__':
    main()
