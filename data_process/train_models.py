#!/usr/bin/python
import numpy as np
from ipromps import ipromps_lib
from sklearn.externals import joblib
import os
import ConfigParser
import ipdb


# the current file path
file_path = os.path.dirname(__file__)
promp = True
ipromp = False
emg_ipromp = False

# read models cfg file
cp_models = ConfigParser.SafeConfigParser()
cp_models.read(os.path.join(file_path, '../cfg/models.cfg'))
# read models params
datasets_path = os.path.join(file_path, cp_models.get('datasets', 'path'))
if promp:
    num_dim = cp_models.getint('promp_param', 'num_dim')
    num_obs_dim = cp_models.getint('promp_param', 'num_obs_dim')
elif ipromp:
    num_dim = cp_models.getint('ipromp_param', 'num_dim')
    num_obs_dim = cp_models.getint('ipromp_param', 'num_obs_dim')
elif emg_ipromp:
    num_dim = cp_models.getint('emg_ipromp_param', 'num_dim')
    num_obs_dim = cp_models.getint('ipromp_param', 'num_obs_dim')

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
    global promp
    # load the data from pkl
    task_name = joblib.load(task_name_path)
    datasets_norm_preproc = joblib.load(datasets_norm_preproc_path)
    min_max_scaler = joblib.load(min_max_scaler_path)
    noise_cov = joblib.load(noise_cov_path)
    if promp:
        promps_set = [ipromps_lib.IProMP(num_joints=num_dim, num_obs_joints=num_obs_dim, num_basis=num_basis,
                                        sigma_basis=sigma_basis, num_samples=len_norm, sigmay=noise_cov,
                                        min_max_scaler=min_max_scaler, num_alpha_candidate=num_alpha_candidate,
                                        method="promp")
                    for x in datasets_norm_preproc]
        # add demo for each ProMPs
        for idx, promp in enumerate(promps_set):
            print('Training the ProMP for task: %s...' % task_name[idx])
            for demo_idx in datasets_norm_preproc[idx]:
                # ipdb.set_trace()
                promp.add_demonstration(demo_idx['left_joints'])   # spatial variance demo
                promp.add_alpha(demo_idx['alpha'])   # temporal variance demo
        # save the trained models
        print('Saving the trained models...')
        joblib.dump(promps_set, os.path.join(datasets_pkl_path, 'promps_set.pkl'))
        print('Trained the ProMPs successfully!!!')

    elif ipromp or emg_ipromp:
        ipromps_set = [ipromps_lib.IProMP(num_joints=num_dim, num_obs_joints=num_obs_dim, num_basis=num_basis,
                                        sigma_basis=sigma_basis, num_samples=len_norm, sigmay=noise_cov,
                                        min_max_scaler=min_max_scaler, num_alpha_candidate=num_alpha_candidate,
                                         method="ipromp")
                    for x in datasets_norm_preproc]
        # add demo for each IProMPs
        for idx, ipromp in enumerate(ipromps_set):
            print('Training the IProMP for task: %s...' % task_name[idx])
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
