#!/usr/bin/python
import numpy as np
from ipromps import ipromps_lib
from sklearn.externals import joblib
import os
import ConfigParser
import ipdb

def main(num_dim=7, num_obs_dim=7, num_basis=31,min_max_scaler=None, 
         sigma_basis=None, len_norm=101, sigmay=None,
         num_alpha_candidate=6, datasets_pkl_path=None,
         method="promp"):
    
    task_name_path = os.path.join(datasets_pkl_path, 'task_name_list.pkl')
    datasets_norm_preproc_path = os.path.join(datasets_pkl_path, 'datasets_norm_preproc.pkl')
    noise_cov_path = os.path.join(datasets_pkl_path, 'noise_cov.pkl')

    task_name = joblib.load(task_name_path)
    noise_cov = joblib.load(noise_cov_path)
    datasets_norm_preproc = joblib.load(datasets_norm_preproc_path)
    
    global promp
    if method == "promp":
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

    elif method =="ipromp" or method =="emg_ipromp":
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
