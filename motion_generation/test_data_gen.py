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
datasets_filtered = joblib.load(os.path.join(datasets_path, 'pkl/datasets_filtered.pkl'))

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
    obs_ratio = 0.7
    # plot = True
    plot_distribution = None
    plot_3d = None
    plot_phase = True

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
    start_idx = 0
    ratio = 10
    num_obs = int(len(timestamp)*obs_ratio)
    num_obs -= num_obs % ratio
    obs_data_post_arr = obs_data_post_arr[start_idx:num_obs:ratio, :]
    timestamp = timestamp[start_idx:num_obs:ratio]
    obs_data_post_arr = obs_data_post_arr
    timestamp = timestamp
    # ipdb.set_trace()
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
        ipromp_id = 3
        [traj_time, traj] = ipromp.gen_real_traj(alpha_max_list[ipromp_id])
        traj = ipromp.min_max_scaler.inverse_transform(traj)
        human_traj= traj[:, human_index[0]:human_index[-1]]
        robot_traj = traj[:, robot_index[0]:robot_index[-1]]   
        traj_full.append([human_traj, robot_traj])
    # ipdb.set_trace()

    if plot_3d:
        task_idx = 0
        human_pred = traj_full[task_idx][0] 
        robot_pred = traj_full[task_idx][1] 
        human_gt = obs_data[:,human_index[0]:human_index[1]]
        robot_gt = obs_data[:,robot_index[0]:robot_index[1]]
        plot_3d_raw_traj(human_pred=human_pred,human_gt=human_gt,robot_pred=robot_pred,robot_gt=robot_gt,task_id=task_idx,
                        datasets_filtered=datasets_filtered,obs_data_dict=obs_data_dict,num_obs=num_obs)    

    if plot_distribution != None:
        plot_ipromps_set = [ipromps_set[1],ipromps_set[2]]
        color_list = ['b', 'g']
        label_name = ['Demonstration 1','Demonstration 2']
        fig = plt.figure(figsize=(8,5),dpi=300,facecolor='w',edgecolor='w')
        for task_idx, ipromp in enumerate(plot_ipromps_set):      
            promp_instance = ipromp.promps[1]
            promp_instance.plot_prior(color=color_list[task_idx],b_regression=False,b_dataset=False,legend=label_name[task_idx])
        plt.xlabel('Time (s)')
        plt.ylabel('y-axis [m]')
        plt.legend(loc=2)
        plt.savefig('fig1',format='eps')
        # plt.show()

    if plot_phase != None:
        plot_ipromps_set = [ipromps_set[2]]
        fig = plt.figure(figsize=(8,5),dpi=300,facecolor='w',edgecolor='w')
        for task_idx, ipromp in enumerate(plot_ipromps_set):      
            promp_instance = ipromp.promps[1]
            promp_instance.plot_prior(color='b',b_regression=False,b_dataset=False)
            promp_instance.plot_nUpdated(color='g')
        plt.xlabel('Time (s)')
        plt.ylabel('y-axis [m]')
        plt.legend(loc=2)
        plt.savefig('fig_phase_1.pdf',format='pdf')
        # plt.show()


def plot_3d_raw_traj(human_pred,human_gt,robot_pred,robot_gt,task_id,datasets_filtered,obs_data_dict,num_obs):
    start_idx = 20
    ratio = 5
    fig = plt.figure(figsize=(8,6),dpi=300,facecolor='w',)
    ax = fig.gca(projection='3d')
    ax.grid(True)
    for demo_idx in data_index[task_id]:
        human = datasets_filtered[task_id][demo_idx]['left_hand']
        ax.plot(human[:, 0], human[:, 1], human[:, 2], linewidth=1, linestyle='-', alpha=0.5, color="grey")
        robot = datasets_filtered[task_id][demo_idx]['left_joints']
        ax.plot(robot[:, 0], robot[:, 1], robot[:, 2], linewidth=1, linestyle='-', alpha=0.5, color="grey")

    ax.plot(human_pred[:, 0], human_pred[:, 1], human_pred[:, 2], linestyle='-',label='Motion prediction',color="b",linewidth=1.5,)
    ax.plot(human_gt[:, 0], human_gt[:, 1], human_gt[:, 2], linestyle='-',label='Motion ground truth',color="r",linewidth=1.5,)
    ax.plot(robot_pred[:, 0], robot_pred[:, 1], robot_pred[:, 2], linestyle='-' ,color='b',linewidth=1.5,)
    ax.plot(robot_gt[:, 0], robot_gt[:, 1], robot_gt[:, 2],linestyle='-',color='r',linewidth=1.5,)
    obs = obs_data_dict['left_hand']
    ax.plot(obs[start_idx:num_obs:ratio, 0], obs[start_idx:num_obs:ratio, 1], obs[start_idx:num_obs:ratio, 2],
        'o', markersize=5, label='Human motion observations', alpha=1.0,
        markerfacecolor='none', markeredgewidth=1.0, markeredgecolor='g')    
    
    ticklines = ax.get_xticklines() + ax.get_yticklines() + ax.get_zticklines()
    gridlines = ax.get_xgridlines() + ax.get_ygridlines() 

    for line in ticklines:
        line.set_linewidth(1)

    for line in gridlines:
        line.set_linestyle('-')


    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 16,
            }
    '''
    fig2_1 
    1.0, 0.20, 0.12
    0.5, 0.30, 0.35
    '''        
    ax.text(1.0, 0.20, 0.12, "Human", color='red')
    ax.text(0.5, 0.30, 0.35, "Robot", color='red')
    ax.set_xlabel('x-axis [m]',fontsize=8,)
    ax.set_ylabel('y-axis [m]',fontsize=8,)
    ax.set_zlabel('z-axis [m]',fontsize=8,)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.zaxis.set_tick_params(labelsize=8)
    ax.legend(loc=2,fontsize=10)
    plt.savefig('fig2_1.eps',format='eps')
    plt.show()



    # save the conditional result
    # print('Saving the post IProMPs...')
    # joblib.dump(ipromps_set, os.path.join(datasets_path, 'pkl/ipromps_set_post_offline.pkl'))
    # save the robot traj
    # print('Saving the robot traj...')
    # joblib.dump([traj_full, obs_data_dict, num_obs], os.path.join(datasets_path, 'pkl/robot_traj_offline.pkl'))


if __name__ == '__main__':
    main()