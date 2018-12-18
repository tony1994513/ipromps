#!/usr/bin/python
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import os
import ConfigParser
from sklearn.externals import joblib
import ipdb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
sigma = cp_models.getint('filter', 'sigma')
ipromps_set = joblib.load(os.path.join(datasets_path, 'pkl/ipromps_set.pkl'))
datasets_raw = joblib.load(os.path.join(datasets_path, 'pkl/datasets_raw.pkl'))
num_obs_dim = cp_models.getint('datasets', 'num_obs_dim')

# read datasets cfg file
cp_datasets = ConfigParser.SafeConfigParser()
cp_datasets.read(os.path.join(datasets_path, './info/cfg/datasets.cfg'))
# read datasets params
data_index_sec = cp_datasets.items('index_17')
data_index = [map(int, task[1].split(',')) for task in data_index_sec]

# after processed human/robot index
human_index = cp_models.get('processed_index', 'human')
robot_index = cp_models.get('processed_index', 'robot')
human_index =  [int(x.strip()) for x in human_index.split(',')]
robot_index =  [int(x.strip()) for x in robot_index.split(',')]


def main():
    task_id = 0
    test_index = 8
    obs_ratio = 0.4
    plot = True

    # read test data
    obs_data_dict = datasets_raw[task_id][test_index]

    left_hand = obs_data_dict['left_hand']
    left_joints = obs_data_dict['left_joints']
    obs_data = np.hstack([left_hand, left_joints])
    timestamp = obs_data_dict['stamp']

    # filter the data
    obs_data = gaussian_filter1d(obs_data.T, sigma=sigma).T
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
    print('Phase estimating...')
    alpha_max_list = []
    for ipromp in ipromps_set:
        alpha_temp = ipromp.alpha_candidate(num_alpha_candidate)
        idx_max = ipromp.estimate_alpha(alpha_temp, obs_data_post_arr, timestamp)
        alpha_max_list.append(alpha_temp[idx_max]['candidate'])
        ipromp.set_alpha(alpha_temp[idx_max]['candidate'])

    # task recognition
    print('Adding via points in each trained model...')
    for task_idx, ipromp in enumerate(ipromps_set):
        for idx in range(len(timestamp)):
            ipromp.add_viapoint(timestamp[idx] / alpha_max_list[task_idx], obs_data_post_arr[idx, :])
        ipromp.param_update(unit_update=True)
    print('Computing the likelihood for each model under observations...')

    prob_task = []
    for ipromp in ipromps_set:
        prob_task_temp = ipromp.prob_obs()
        prob_task.append(prob_task_temp)
    idx_max_prob = np.argmax(prob_task)
    print('The max fit model index is task %s' % task_name[idx_max_prob])

    # robot motion generation
    traj_full = []
    for ipromp_id, ipromp in enumerate(ipromps_set):
        [traj_time, traj] = ipromp.gen_real_traj(alpha_max_list[ipromp_id])
        traj = ipromp.min_max_scaler.inverse_transform(traj)
        human_traj= traj[:, human_index[0]:human_index[-1]]
        robot_traj = traj[:, robot_index[0]:robot_index[-1]]   
        traj_full.append([human_traj, robot_traj])

    # if plot != None:
    #     for task_idx, task in traj_full:
    #         fig = plt.figure()
    #         ax = fig.add_subplot( projection='3d')
    #         human = task[0]
    #         robot = task[1]
    #         ipdb.set_trace()
    #         ax.plot(human[:,0],human[:,1],human[:,2])
    #         ax.plot(robot[:,0],robot[:,1],robot[:,2])
    #     plt.show()




    # save the conditional result
    print('Saving the post IProMPs...')
    joblib.dump(ipromps_set, os.path.join(datasets_path, 'pkl/ipromps_set_post_offline.pkl'))
    # save the robot traj
    print('Saving the robot traj...')
    joblib.dump([traj_full, obs_data_dict, num_obs], os.path.join(datasets_path, 'pkl/robot_traj_offline.pkl'))


if __name__ == '__main__':
    main()