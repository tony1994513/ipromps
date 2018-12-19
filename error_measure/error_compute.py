#!/usr/bin/python
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import os
import ConfigParser
from sklearn.externals import joblib

# read the current file path
file_path = os.path.dirname(__file__)
# read model cfg file
cp_models = ConfigParser.SafeConfigParser()
cp_models.read(os.path.join(file_path, '../cfg/models.cfg'))

# load param
datasets_path = os.path.join(file_path, cp_models.get('datasets', 'path'))
num_alpha_candidate = cp_models.getint('phase', 'num_phaseCandidate')
task_name_path = os.path.join(datasets_path, 'pkl/task_name_list.pkl')
task_name = joblib.load(task_name_path)
sigma = cp_models.get('filter', 'sigma')
num_joints = cp_models.getint('datasets', 'num_dim')
num_obs_dim = cp_models.getint('datasets', 'num_obs_dim')
datasets_pkl_path = os.path.join(datasets_path, 'pkl')
datasets_raw_path = os.path.join(datasets_pkl_path, 'datasets_raw.pkl')
datasets_filtered_path = os.path.join(datasets_pkl_path, 'datasets_filtered.pkl')
datasets_norm_preproc_path = os.path.join(datasets_pkl_path, 'datasets_norm_preproc.pkl')


# read datasets cfg file
cp_datasets = ConfigParser.SafeConfigParser()
cp_datasets.read(os.path.join(datasets_path, './info/cfg/datasets.cfg'))
# read datasets params

data_index_sec = cp_datasets.items('index_17')
data_index = [map(int, task[1].split(',')) for task in data_index_sec]


def main(ipromps_set, test_index, obs_ratio, task_id, num_alpha_candidate):
    datasets_raw = joblib.load(datasets_raw_path)
    datasets_filtered = joblib.load(datasets_filtered_path)
    datasets_norm_preproc = joblib.load(datasets_norm_preproc_path)

    obs_data_dict = datasets_raw[task_id][data_index[task_id][test_index[0]]]

    left_hand = obs_data_dict['left_hand']
    left_joints = obs_data_dict['left_joints']
    obs_data = np.hstack([left_hand, left_joints])
    timestamp = obs_data_dict['stamp']

    # filter the data
    # obs_data = gaussian_filter1d(obs_data.T, sigma=sigma).T
    # preprocessing for the data
    obs_data_post_arr = ipromps_set[0].min_max_scaler.transform(obs_data)
    # consider the unobserved info
    obs_data_post_arr[:, num_obs_dim:] = 0.0

    # choose the data
    ratio = 3
    num_obs = int(len(timestamp)*obs_ratio)
    num_obs -= num_obs % ratio
    obs_data_post_arr = obs_data_post_arr[0:num_obs:ratio, :]
    timestamp = timestamp[0:num_obs:ratio]
    obs_data_post_arr = obs_data_post_arr
    timestamp = timestamp

    # phase estimation
    alpha_max_list = []
    for ipromp in ipromps_set:
        alpha_temp = ipromp.alpha_candidate(num_alpha_candidate)
        idx_max = ipromp.estimate_alpha(alpha_temp, obs_data_post_arr, timestamp)
        alpha_max_list.append(alpha_temp[idx_max]['candidate'])
        ipromp.set_alpha(alpha_temp[idx_max]['candidate'])
    # task recognition
    # print('Adding via points in each trained model...')
    for task_idx, ipromp in enumerate(ipromps_set):
        for idx in range(len(timestamp)):
            ipromp.add_viapoint(timestamp[idx] / alpha_max_list[task_idx], obs_data_post_arr[idx, :])
        ipromp.param_update(unit_update=True)
    # print('Computing the likelihood for each model under observations...')

    prob_task = []
    for ipromp in ipromps_set:
        prob_task_temp = ipromp.prob_obs()
        prob_task.append(prob_task_temp)
    idx_max_prob = np.argmax(prob_task)
    # idx_max_prob = task_id # a trick for testing
    print('The max fit model index is task %s' % task_name[idx_max_prob])

    # robot motion generation
    [traj_time, traj] = ipromps_set[task_id].gen_real_traj(alpha_max_list[task_id])
    traj = ipromps_set[task_id].min_max_scaler.inverse_transform(traj)

    robot_traj = traj[:, num_obs_dim:]
    ground_truth = datasets_raw[task_id][data_index[task_id][test_index[0]]]['left_joints']
    positioning_error = np.sqrt(np.sum(np.square(robot_traj[-1,:] - ground_truth[-1,:])))
    phase_error = alpha_max_list[task_id] - datasets_norm_preproc[task_id][test_index[0]]['alpha']

    return [idx_max_prob, positioning_error, np.abs(phase_error)]




if __name__ == '__main__':
    main()