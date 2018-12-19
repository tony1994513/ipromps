#!/usr/bin/python
import train_model_func
from sklearn.model_selection import LeaveOneOut
import numpy as np
from ipromps import ipromps_lib
from sklearn.externals import joblib
import os
import ConfigParser
import error_compute

# the current file path
file_path = os.path.dirname(__file__)

# read models cfg file
cp_models = ConfigParser.SafeConfigParser()
cp_models.read(os.path.join(file_path, '../cfg/models.cfg'))
# read models params
datasets_path = os.path.join(file_path, cp_models.get('datasets', 'path'))

# the pkl data
datasets_pkl_path = os.path.join(datasets_path, 'pkl')
datasets_norm_preproc_path = os.path.join(datasets_pkl_path, 'datasets_norm_preproc.pkl')


def main(obs_ratio, task_id, num_alpha_candidate):
    # load the data from pkl
    datasets_norm_preproc = joblib.load(datasets_norm_preproc_path)

    X = datasets_norm_preproc[0]
    loo = LeaveOneOut()
    loo.get_n_splits(X)

    ee_error_full = []
    phase_error_full = []
    i = 0.0
    for train_index, test_index in loo.split(X):
        ipromps_set = train_model_func.main(train_index)
        [idx_max_prob, positioning_error, phase_error] = error_compute.main(ipromps_set, test_index, obs_ratio, task_id, num_alpha_candidate)
        ee_error_full.append(positioning_error)
        phase_error_full.append(phase_error)
        if idx_max_prob == task_id:
            i += 1.0
    acc = i/len(datasets_norm_preproc[0])
    return [ee_error_full, acc, phase_error_full]


if __name__ == '__main__':
    main()
