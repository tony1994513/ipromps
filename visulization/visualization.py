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

# load datasets
ipromps_set = joblib.load(os.path.join(datasets_path, 'pkl/ipromps_set.pkl'))
# ipromps_set_post = joblib.load(os.path.join(datasets_path, 'pkl/ipromps_set_post.pkl'))
datasets_raw = joblib.load(os.path.join(datasets_path, 'pkl/datasets_raw.pkl'))
datasets_norm = joblib.load(os.path.join(datasets_path, 'pkl/datasets_norm.pkl'))
datasets_filtered = joblib.load(os.path.join(datasets_path, 'pkl/datasets_filtered.pkl'))
datasets_norm_preproc = joblib.load(os.path.join(datasets_path, 'pkl/datasets_norm_preproc.pkl'))
task_name = joblib.load(os.path.join(datasets_path, 'pkl/task_name_list.pkl'))
[robot_traj_offline, ground_truth, num_obs] = joblib.load(os.path.join(datasets_path, 'pkl/robot_traj_offline.pkl'))
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
info = cp_models.get('visualization', 'info')
joint_num = info_n_idx[info][1] - info_n_idx[info][0]


# # zh config
# def conf_zh(font_name):
#     from pylab import mpl
#     mpl.rcParams['font.sans-serif'] = [font_name]
#     mpl.rcParams['axes.unicode_minus'] = False


# plot the raw data
def plot_raw_data(num=0):
    for task_idx, ipromps_idx in enumerate(ipromps_set):
        fig = plt.figure(task_idx + num)
        fig.suptitle('the raw data of ' + info)
        for demo_idx in range(ipromps_idx.num_demos):
            for joint_idx in range(joint_num):
                ax = fig.add_subplot(joint_num, 1, 1 + joint_idx)
                data = datasets_raw[task_idx][demo_idx][info][:, joint_idx]
                plt.plot(np.array(range(len(data)))/100.0, data)
                ax.set_xlabel('t(s)')
                ax.set_ylabel('y(m)')

# plot the norm data
def plot_norm_data(num=0):
    for task_idx, ipromps_idx in enumerate(ipromps_set):
        fig = plt.figure(task_idx + num)
        fig.suptitle('the raw data of ' + info)
        for demo_idx in range(ipromps_idx.num_demos):
            for joint_idx in range(joint_num):
                ax = fig.add_subplot(joint_num, 1, 1 + joint_idx)
                data = datasets_norm[task_idx][demo_idx][info][:, joint_idx]
                plt.plot(np.array(range(len(data)))/100.0, data)
                ax.set_xlabel('t(s)')
                ax.set_ylabel('y(m)')

# plot the filtered data
def plot_filtered_data(num=0):
    for task_idx, ipromps_idx in enumerate(ipromps_set):
        fig = plt.figure(task_idx + num)
        fig.suptitle('the filtered data of ' + info)
        for demo_idx in range(ipromps_idx.num_demos):
            for joint_idx in range(joint_num):
                ax = fig.add_subplot(joint_num, 1, 1 + joint_idx)
                data = datasets_filtered[task_idx][demo_idx][info][:, joint_idx]
                plt.plot(np.array(range(len(data)))/100.0, data, )
                ax.set_xlabel('t(s)')
                ax.set_ylabel('y(m)')
                plt.legend()

# plot the norm data
def plot_preproc_data(num=0):
    for task_idx, ipromps_idx in enumerate(ipromps_set):
        fig = plt.figure(task_idx + num)
        fig.suptitle('the raw data of ' + info)
        for demo_idx in range(ipromps_idx.num_demos):
            for joint_idx in range(joint_num):
                ax = fig.add_subplot(joint_num, 1, 1 + joint_idx)
                data = datasets_norm_preproc[task_idx][demo_idx][info][:, joint_idx]
                plt.plot(np.array(range(len(data))) / 100.0, data)
                ax.set_xlabel('t(s)')
                ax.set_ylabel('y(m)')


# plot the prior distribution
def plot_prior(num=0):
    for task_idx, ipromps_idx in enumerate(ipromps_set):
        fig = plt.figure(task_idx+num)
        fig.suptitle('the prior of ' + info + ' for ' + task_name[task_idx] + ' model')
        for joint_idx in range(joint_num):
            ax = fig.add_subplot(joint_num, 1, 1+joint_idx)
            ipromps_idx.promps[joint_idx + info_n_idx[info][0]].plot_prior(b_regression=True)


# plot alpha distribute
def plot_alpha(num=0):
    fig = plt.figure(num)
    for idx, ipromp in enumerate(ipromps_set):
        # ax = fig.add_subplot(len(ipromps_set), 1, 1+idx)
        plt.figure(idx)
        h = ipromps_set[idx].alpha
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
    #     h = ipromps_set[0].alpha
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
    for task_idx, ipromps_idx in enumerate(ipromps_set_post):
        fig = plt.figure(task_idx+num)
        for joint_idx in range(joint_num):
            ax = fig.add_subplot(joint_num, 1, 1 + joint_idx)
            ipromps_idx.promps[joint_idx + info_n_idx[info][0]].plot_nUpdated(color='r', via_show=True)


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
                ax.legend()


# plot the 3d filtered traj
def plot_3d_filtered_h_traj(num=0):
    for task_idx, demo_list in enumerate(data_index):
        fig = plt.figure(task_idx+num)
        ax = fig.gca(projection='3d')
        for demo_idx in demo_list:
            data = datasets_filtered[task_idx][demo_idx]['left_hand']
            if demo_idx == 0:
                    ax.plot(data[:, 0], data[:, 1], data[:, 2], linewidth=1,color="grey",
                    label='Training dataset', alpha=1)
            ax.plot(data[:, 0], data[:, 1], data[:, 2], linewidth=1,color="grey",
                    alpha=1)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')

# plot the 3d filtered robot traj
def plot_3d_filtered_r_traj(num=0):
    for task_idx, demo_list in enumerate(data_index):
        fig = plt.figure(task_idx+num)
        ax = fig.gca(projection='3d')
        for demo_idx in demo_list:
            data = datasets_filtered[task_idx][demo_idx]['left_joints']
            # if demo_idx == 0:
            #         ax.plot(data[:, 0], data[:, 1], data[:, 2],
            #         linewidth=1, linestyle='-', alpha=1,label='Robot training dataset', color="grey")
            ax.plot(data[:, 0], data[:, 1], data[:, 2], linewidth=1, linestyle='-', alpha=1, color="grey")


# plot the offline obs
def plot_offline_3d_obs(num=0):
    for task_idx, demo_list in enumerate(data_index):
        fig = plt.figure(task_idx + num)
        ax = fig.gca(projection='3d')
        obs_data_dict = ground_truth
        data = obs_data_dict['left_hand']
        ax.plot(data[0:num_obs:15, 0], data[0:num_obs:15, 1], data[0:num_obs:15, 2],
                'o', markersize=10, label='Human motion observations', alpha=1.0,
                markerfacecolor='none', markeredgewidth=3.0, markeredgecolor='g')
        ax.plot(data[:, 0], data[:, 1], data[:, 2],
                '-', linewidth=5, color='blue', label='Human trajectory ground truth', alpha = 1.0)
        data = ground_truth['left_joints']
        # import ipdb;ipdb.set_trace()
        ax.plot(data[:, -7], data[:, -6], data[:, -5],
                linewidth=5, linestyle='-', color='r', label='Robot trajectory ground Truth', alpha=1.0)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        # ax.legend(loc=2)




# plot the 3d generated robot traj
def plot_gen_3d_offline_r_traj(num=0):
    for task_idx, demo_list in enumerate(data_index):
        fig = plt.figure(task_idx+num)
        ax = fig.gca(projection='3d')
        data = robot_traj_offline[task_idx][1]
        ax.plot(data[:, 0], data[:, 1], data[:, 2], 'r',
                linewidth=5, linestyle='--', label='Predicted robot trajectory')
        data = robot_traj_offline[task_idx][0]
        ax.plot(data[:, 0], data[:, 1], data[:, 2], 'b',
                linewidth=5, linestyle='--', label='Predicted human trajectory')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend(loc=0)


# plot the 3d generated robot traj
def plot_gen_3d_online_r_traj(num=0):
    for task_idx, demo_list in enumerate(data_index):
        fig = plt.figure(task_idx+num)
        ax = fig.gca(projection='3d')
        data = robot_traj_online
        ax.plot(data[:, 0], data[:, 1], data[:, 2],
                linewidth=8, linestyle='-', label='generated online robot traj', alpha=0.2)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.legend()


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
        ax.legend()


# plot offline test pair
def pairs_offline(num=0):
    plot_3d_filtered_h_traj(num)
    plot_3d_filtered_r_traj(num)
    plot_offline_3d_obs(num)
    plot_gen_3d_offline_r_traj(num)


# plot online test pair
def pairs_online(num=0):
    plot_3d_filtered_h_traj(num)
    plot_3d_filtered_r_traj(num)
    plot_online_3d_obs(num)
    plot_gen_3d_online_r_traj(num)

# plot the filtered data
def plot_single_dim(num=0):
    for task_idx, ipromps_idx in enumerate(ipromps_set):
        fig = plt.figure(task_idx + num)
        fig.suptitle('the filtered data of ' + info)
        for demo_idx in range(ipromps_idx.num_demos):
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
    # plot_3d_raw_traj(10)
    # plot_3d_gen_r_traj_online(10)
    pairs_offline(0)
    # pairs_online(10)
    plt.legend(prop = {'size': 20})
    plt.show()


if __name__ == '__main__':
    main()
