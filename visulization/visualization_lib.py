#!/usr/bin/python
# coding:utf-8
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import scipy.stats as stats
import os
import ConfigParser
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties


# read conf file
file_path = os.path.dirname(__file__)
cp_models = ConfigParser.SafeConfigParser()
cp_models.read(os.path.join(file_path, '../cfg/models.cfg'))
# the datasets path
datasets_path = os.path.join(file_path, cp_models.get('datasets', 'path'))
method = "promp"
# load datasets
MPs_set = joblib.load(os.path.join(datasets_path, 'pkl/'+method+'_set.pkl'))
MPs_set_post = joblib.load(os.path.join(datasets_path, 'pkl/'+method+'_post_offline.pkl'))
datasets_raw = joblib.load(os.path.join(datasets_path, 'pkl/'+method+'_datasets_raw.pkl'))
datasets_norm = joblib.load(os.path.join(datasets_path, 'pkl/'+method+'_datasets_norm.pkl'))
datasets_filtered = joblib.load(os.path.join(datasets_path, 'pkl/'+method+'_datasets_filtered.pkl'))
datasets_norm_preproc = joblib.load(os.path.join(datasets_path, 'pkl/'+method+'_datasets_norm_preproc.pkl'))
task_name = joblib.load(os.path.join(datasets_path, 'pkl/task_name_list.pkl'))
[MP_traj_offline, ground_truth, viapoint] = joblib.load(os.path.join(datasets_path, 'pkl/'+method+'_traj_offline.pkl'))

# robot_traj_online = joblib.load(os.path.join(datasets_path, 'pkl/robot_traj_online.pkl'))
# obs_data_online = joblib.load(os.path.join(datasets_path, 'pkl/obs_data_online.pkl'))


# read datasets cfg file
cp_datasets = ConfigParser.SafeConfigParser()
cp_datasets.read(os.path.join(datasets_path, 'info/cfg/datasets.cfg'))
# read datasets params
data_index_sec = cp_datasets.items('index_17')
data_index = [map(int, task[1].split(',')) for task in data_index_sec]

# after processed human/robot index
human_index = cp_models.get('processed_index', 'human')
robot_index = cp_models.get('processed_index', 'robot')
human_index =  [int(x.strip()) for x in human_index.split(',')]
robot_index =  [int(x.strip()) for x in robot_index.split(',')]

info_n_idx = {
            'left_hand': human_index,
            'left_joints': robot_index,
            }
# the info to be plotted
if method == "promp":
    info = "left_joints"
    joint_num = info_n_idx[info][1] - info_n_idx[info][0]
elif method == "ipromp":
    info = cp_models.get('visualization', 'info')
    joint_num = info_n_idx[info][1] - info_n_idx[info][0]


# # zh config
# def conf_zh(font_name):
#     from pylab import mpl
#     mpl.rcParams['font.sans-serif'] = [font_name]
#     mpl.rcParams['axes.unicode_minus'] = False


# plot the raw data
def plot_raw_data(num=0):
    for task_idx, MPs in enumerate(MPs_set):
        fig = plt.figure(task_idx + num)
        fig.suptitle('the raw data of ' + info)
        for demo_idx in range(MPs.num_demos):
            for joint_idx in range(joint_num):
                ax = fig.add_subplot(joint_num, 1, 1 + joint_idx)
                data = datasets_raw[task_idx][demo_idx][info][:, joint_idx]
                plt.plot(np.array(range(len(data)))/100.0, data)
                ax.set_xlabel('t(s)')
                ax.set_ylabel('y(m)')

# plot the norm data
def plot_norm_data(num=0):
    for task_idx, MPs in enumerate(MPs_set):
        fig = plt.figure(task_idx + num)
        fig.suptitle('the raw data of ' + info)
        for demo_idx in range(MPs.num_demos):
            for joint_idx in range(joint_num):
                ax = fig.add_subplot(joint_num, 1, 1 + joint_idx)
                data = datasets_norm[task_idx][demo_idx][info][:, joint_idx]
                plt.plot(np.array(range(len(data)))/100.0, data)
                ax.set_xlabel('t(s)')
                ax.set_ylabel('y(m)')

# plot the filtered data
def plot_filtered_data(num=0):
    for task_idx, MPs in enumerate(MPs_set):
        fig = plt.figure(task_idx + num)
        fig.suptitle('the filtered data of ' + info)
        for demo_idx in range(MPs.num_demos):
            for joint_idx in range(joint_num):
                ax = fig.add_subplot(joint_num, 1, 1 + joint_idx)
                data = datasets_filtered[task_idx][demo_idx][info][:, joint_idx]
                plt.plot(np.array(range(len(data)))/100.0, data, )
                ax.set_xlabel('t(s)')
                ax.set_ylabel('y(m)')
                plt.legend()

# plot the norm data
def plot_preproc_data(num=0):
    for task_idx, MPs in enumerate(MPs_set):
        fig = plt.figure(task_idx + num)
        fig.suptitle('the raw data of ' + info)
        for demo_idx in range(MPs.num_demos):
            for joint_idx in range(joint_num):
                ax = fig.add_subplot(joint_num, 1, 1 + joint_idx)
                data = datasets_norm_preproc[task_idx][demo_idx][info][:, joint_idx]
                plt.plot(np.array(range(len(data))) / 100.0, data)
                ax.set_xlabel('t(s)')
                ax.set_ylabel('y(m)')


# plot the prior distribution
def plot_prior(num=0):
    for task_idx, MPs in enumerate(MPs_set):
        fig = plt.figure(task_idx+num)
        fig.suptitle('the prior of ' + info + ' for ' + task_name[task_idx] + ' model')
        for joint_idx in range(joint_num):
            ax = fig.add_subplot(joint_num, 1, 1+joint_idx)
            MPs.promps[joint_idx + info_n_idx[info][0]].plot_prior(b_regression=True)


# plot alpha distribute
def plot_alpha(num=0):
    fig = plt.figure(num)
    for idx, ipromp in enumerate(MPs_set):
        # ax = fig.add_subplot(len(MPs_set), 1, 1+idx)
        plt.figure(idx)
        h = MPs_set[idx].alpha
        h.sort()
        h_mean = np.mean(h)
        h_std = np.std(h)
        pdf = stats.norm.pdf(h, h_mean, h_std)
        pdf = np.zeros_like(pdf)
        # pl.hist(h, normed=True, color='g')
        # plt.plot(h, pdf, marker='o', markersize=10, color='r')
        plt.scatter(h, pdf, s=100)
        xx = np.linspace(h_mean-3*h_std, h_mean+3*h_std, 100)
        yy = stats.norm.pdf(xx, h_mean, h_std)
        plt.plot(xx, yy, linewidth=2, color='r', markersize=10, alpha=0.8)
        plt.xlabel('Phase factor')
        plt.ylabel('Probability')
    # plt.figure(100)
    #
    # for i in range(10):
    #     plt.figure(i)
    #     h = MPs_set[0].alpha
    #     h_mean = np.mean(h)
    #     h_std = np.std(h)
    #     s = np.random.normal(h_mean, h_std, i+1)
    #     pdf = stats.norm.pdf(s, h_mean, h_std)
    #     plt.scatter(s, pdf, s=100, label='Phase factor candidate')
    #     xx = np.linspace(h_mean - 3 * h_std, h_mean + 3 * h_std, 100)
    #     yy = stats.norm.pdf(xx, h_mean, h_std)
    #     plt.plot(xx, yy, linewidth=2, color='r', markersize=10, alpha=0.8)
    #     plt.legend(loc=2)
    #     plt.xlabel('Phase factor candidate number')
    #     plt.ylabel('Probability')



        # candidate = ipromp.alpha_candidate()
        # candidate_x = [x['candidate'] for x in candidate]
        # prob = [x['prob'] for x in candidate]
        # plt.plot(candidate_x, prob, linewidth=0, color='g', marker='o', markersize=14)


# plot the post distribution
def plot_post(num=0):
    for task_idx, MPs in enumerate(MPs_set_post):
        fig = plt.figure(task_idx+num)
        for joint_idx in range(joint_num):
            ax = fig.add_subplot(joint_num, 1, 1 + joint_idx)
            MPs.promps[joint_idx + info_n_idx[info][0]].plot_nUpdated(color='r', via_show=True)


# plot the generated robot motion trajectory
def plot_robot_traj(num=0):
    fig = plt.figure(num)
    fig.suptitle('predict robot motion')
    for joint_idx in range(7):
        ax = fig.add_subplot(7, 1, 1 + joint_idx)
        plt.plot(np.linspace(0, 1.0, 101), robot_traj_online[:, joint_idx])


# plot the raw data index
def plot_raw_data_index(num=0):
    for task_idx, demo_list in enumerate(data_index):
        for demo_idx in demo_list:
            fig = plt.figure(num + task_idx)
            fig.suptitle('the raw data of ' + info)
            for joint_idx in range(joint_num):
                ax = fig.add_subplot(joint_num, 1, 1 + joint_idx)
                data = datasets_raw[task_idx][demo_idx][info][:, joint_idx]
                plt.plot(range(len(data)), data, label=str(demo_idx))
                plt.legend()


# plot the filter data index
def plot_filter_data_index(num=0):
    for task_idx, demo_list in enumerate(data_index):
        for demo_idx in demo_list:
            fig = plt.figure(num + task_idx)
            fig.suptitle('the raw data of ' + info)
            for joint_idx in range(joint_num):
                ax = fig.add_subplot(joint_num, 1, 1 + joint_idx)
                data = datasets_filtered[task_idx][demo_idx][info][:, joint_idx]
                plt.plot(range(len(data)), data, label=str(demo_idx))
                ax.legend()
    # for task_idx, demo_list in enumerate(data_index):
    #     fig = plt.figure(num)
    #     for demo_idx in demo_list:
    #         ax = fig.add_subplot(len(data_index), 1, task_idx)
    #         data = datasets_filtered[task_idx][demo_idx][info][:, 2]
    #         plt.plot(range(len(data)), data)
    #         plt.xlabel('t(s)')
    #         plt.ylabel('EMG(mA)')
    #         ax.legend()


# plot the 3d raw traj
def plot_3d_raw_traj(num=0):
    for task_idx, demo_list in enumerate(data_index):
        fig = plt.figure(task_idx + num)
        ax = fig.gca(projection='3d')
        for demo_idx in demo_list:
            for joint_idx in range(joint_num):
                data = datasets_raw[task_idx][demo_idx][info]
                ax.plot(data[:, 0], data[:, 1], data[:, 2], label=str(demo_idx))
                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')


# plot the 3d filtered traj
def plot_3d_filtered_h_traj(num=0):
    for task_idx, demo_list in enumerate(data_index):
        fig = plt.figure(task_idx+num,figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w')
        ax = fig.gca(projection='3d')
        for demo_idx in demo_list:
            data = datasets_filtered[task_idx][demo_idx]['left_hand']
            if demo_idx == 0:
                    ax.plot(data[:, 0], data[:, 1], data[:, 2], linewidth=1,color="grey",
                    label='Model sample', alpha=1)
            ax.plot(data[:, 0], data[:, 1], data[:, 2], linewidth=1,color="grey",
                    alpha=1)
        ax.set_xlabel('X (m)',fontsize=15)
        ax.set_ylabel('Y (m)',fontsize=15)
        ax.set_zlabel('Z (m)',fontsize=15)
        # ax.legend(fontsize=20)

# plot the 3d filtered robot traj
def plot_3d_filtered_r_traj(num=0):
    for task_idx, demo_list in enumerate(data_index):
        fig = plt.figure(task_idx+num,figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w')
        ax = fig.gca(projection='3d')
        for demo_idx in demo_list:
            data = datasets_filtered[task_idx][demo_idx]['left_joints']
            if demo_idx == 0:
                    ax.plot(data[:, 0], data[:, 1], data[:, 2],
                    linewidth=1, linestyle='-', alpha=1,label='Robot training dataset', color="grey")
            ax.plot(data[:, 0], data[:, 1], data[:, 2], linewidth=1, linestyle='-', alpha=1, color="grey")
        ax.legend(fontsize=20)


start_idx = 40
ratio = 5
# plot the offline obs
def plot_offline_3d_obs(num=0):
    for task_idx, demo_list in enumerate(data_index):
        fig = plt.figure(task_idx + num,figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w')
        ax = fig.gca(projection='3d')
        obs_data_dict = ground_truth
        data = obs_data_dict['left_hand']
        ax.plot(data[start_idx:num_obs:ratio, 0], data[start_idx:num_obs:ratio, 1], data[start_idx:num_obs:ratio, 2],
                'o', markersize=10, label='Human motion observations', alpha=1.0,
                markerfacecolor='none', markeredgewidth=1.0, markeredgecolor='r')
        ax.plot(data[:, 0], data[:, 1], data[:, 2],
                '-', linewidth=2, color='r', label='Ground truth', alpha = 1.0)
        data = ground_truth['left_joints']
        ax.plot(data[:, 0], data[:, 1], data[:, 2],
                linewidth=2, linestyle='-', color='r', alpha=1.0)
        # ax.legend(fontsize=20)

def plot_promp_movement(num=0):
    for task_idx, demo_list in enumerate(data_index):
        fig = plt.figure(task_idx + num,figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w')
        ax = fig.gca(projection='3d')
        robot_gt = ground_truth['left_joints']
        robot_pred = promp_robotTraj_offline[task_idx]
        ax.plot(robot_gt[:, 0], robot_gt[:, 1], robot_gt[:, 2],
                linewidth=3, linestyle='-', color='r', alpha=1.0, label='Robot movement groundtruth')
        ax.plot(robot_pred[:, 0], robot_pred[:, 1], robot_pred[:, 2],
                linewidth=3, linestyle='-', color='g', alpha=1.0, label='Robot movement prediction')  
        ax.plot(promp_viapoint[:, 0], promp_viapoint[:, 1], promp_viapoint[:, 2],
                'o', markersize=10, label='via_points', alpha=1.0,
                markerfacecolor='none', markeredgewidth=1.0, markeredgecolor='r')                  
        ax.legend(fontsize=20)
        
# plot the 3d generated robot traj
def plot_gen_3d_offline_r_traj(num=0,figsize=(8, 6), dpi=300, facecolor='w', edgecolor='w'):
    for task_idx, demo_list in enumerate(data_index):
        fig = plt.figure(task_idx+num)
        ax = fig.gca(projection='3d')
        data = robot_traj_offline[task_idx][1]
        ax.plot(data[:, 0], data[:, 1], data[:, 2], 'b',
                linewidth=2, linestyle='-', label='Predicted mean')
        data = robot_traj_offline[task_idx][0]
        ax.plot(data[:, 0], data[:, 1], data[:, 2], 'b',
                linewidth=2, linestyle='-')
        ax.legend(fontsize=20)

# plot offline test pair
def pairs_offline(num=0):
    plot_3d_filtered_h_traj(num)
    plot_3d_filtered_r_traj(num)
    plot_offline_3d_obs(num)
    plot_gen_3d_offline_r_traj(num)

def promp_offline(num=0):
    plot_3d_filtered_r_traj(num)
    plot_promp_movement(num=0)
# plot the 3d generated robot traj
def plot_gen_3d_online_r_traj(num=0):
    for task_idx, demo_list in enumerate(data_index):
        fig = plt.figure(task_idx+num)
        ax = fig.gca(projection='3d')
        data = robot_traj_online
        ax.plot(data[:, 0], data[:, 1], data[:, 2],
                linewidth=8, linestyle='-', label='generated online robot traj', alpha=0.2)
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.zlabel('Z (m)')

def plot_online_3d_obs(num):
    for task_idx, demo_list in enumerate(data_index):
        fig = plt.figure(task_idx + num)
        ax = fig.gca(projection='3d')
        data = obs_data_online
        ax.plot(data[0:num_obs, 0], data[0:num_obs, 1], data[0:num_obs, 2],
                'o', linewidth=3, label='obs points', alpha=0.2)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')






# plot online test pair
def pairs_online(num=0):
    plot_3d_filtered_h_traj(num)
    plot_3d_filtered_r_traj(num)
    plot_online_3d_obs(num)
    plot_gen_3d_online_r_traj(num)

# plot the filtered data
def plot_single_dim(num=0):
    for task_idx, MPs in enumerate(MPs_set):
        fig = plt.figure(task_idx + num)
        fig.suptitle('the filtered data of ' + info)
        for demo_idx in range(MPs.num_demos):
            ax = fig.add_subplot(1, 1, 1 )
            data = datasets_filtered[task_idx][demo_idx][info][:, 1]
            plt.plot(np.array(range(len(data)))/100.0, data, )
            ax.set_xlabel('t(s)')
            ax.set_ylabel('y(m)')
            plt.legend()

def plot_preproc_result(num):
    fig = plt.figure(num)
    for demo_idx in range(17):
        ax = fig.add_subplot(1, 1, 1)
        data = datasets_norm[0][demo_idx][info][:, 1]
        plt.plot(np.array(range(len(data))) / 100.0, data)
        ax.set_xlabel('t(s)')
        ax.set_ylabel('y(m)')
    fig = plt.figure(num+1)
    for demo_idx in range(17):
        ax = fig.add_subplot(1, 1, 1)
        data = datasets_norm_preproc[0][demo_idx][info][:, 1]
        plt.plot(np.array(range(len(data))) / 100.0, data)
        ax.set_xlabel('t(s)')
        ax.set_ylabel('y(m)')


def plot_norm_result(num):
    fig = plt.figure(num)
    for demo_idx in range(17):
        ax = fig.add_subplot(1, 1, 1)
        data = datasets_filtered[0][demo_idx][info][:, 1]
        plt.plot(np.array(range(len(data))) / 100.0, data)
        ax.set_xlabel('t(s)')
        ax.set_ylabel('y(m)')
    fig = plt.figure(num+1)
    for demo_idx in range(17):
        ax = fig.add_subplot(1, 1, 1)
        data = datasets_norm[0][demo_idx][info][:, 1]
        plt.plot(np.array(range(len(data))) / 100.0, data)
        ax.set_xlabel('t(s)')
        ax.set_ylabel('y(m)')


def main():
    # conf_zh("Droid Sans Fallback")
    # plt.close('all')
    # plot_raw_data(0)
    # plot_norm_data(0)
    # plot_preproc_data(10)
    # plot_filtered_data(10)
    # plot_single_dim(0)
    # plot_prior(0)
    # plot_post()
    # plot_alpha()
    # plot_robot_traj()
    # plot_raw_data_index()
    # plot_filter_data_index(20)
    # plot_preproc_result(10)
    # plot_norm_result(10)

    #3D
<<<<<<< HEAD:visulization/visualization_lib.py
    # promp_offline(0)
=======
>>>>>>> 0f21c22cc0372c6ef39468053f9e8eed45f6f39e:visulization/visualization.py
    # plot_3d_raw_traj(10)
    # plot_3d_gen_r_traj_online(10)
    pairs_offline(0)
    # pairs_online(10)

    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.savefig('fig3',format='eps')
    plt.show()


if __name__ == '__main__':
    main()
