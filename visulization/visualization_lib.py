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
import ipdb

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


def plot_single_dim_prior_distribution_diff_task(num=0,dim_idx=0):
    for task_idx, MPs in enumerate(MPs_set):
        fig = plt.figure(task_idx+num)
        fig.suptitle('Prior_distribution_' + info + '_' + task_name[task_idx]+"_dim_"+str(dim_idx))
        if method == "ipromp":
            MPs.promps[dim_idx + info_n_idx[info][0]].plot_prior_distribution()
        elif method == "promp":
            MPs.promps[dim_idx].plot_prior_distribution()

def plot_single_dim_prior_distribution_same_task(num=0,task_idx=0):
    MPs = MPs_set[task_idx]
    for dim_idx in range(joint_num):
        fig = plt.figure(dim_idx+num)
        fig.suptitle('Prior_distribution_' + info + '_' + task_name[task_idx]+"_dim_"+str(dim_idx))
        if method == "ipromp":
            MPs.promps[dim_idx + info_n_idx[info][0]].plot_prior_distribution()
        elif method == "promp":
            MPs.promps[dim_idx].plot_prior_distribution()

def plot_single_dim_post_distribution_diff_task(num=0,dim_idx=0):
    for task_idx, MPs_post in enumerate(MPs_set_post):
        fig = plt.figure(task_idx+num)
        fig.suptitle('Prior_distribution_' + info + '_' + task_name[task_idx]+"_dim_"+str(dim_idx))
        if method == "ipromp":
            MPs_post.promps[dim_idx + info_n_idx[info][0]].plot_prior_distribution()
        elif method == "promp":
            MPs_post.promps[dim_idx].plot_nUpdated_distribution()

def plot_single_dim_post_distribution_same_task(num=0,task_idx=0):
    MPs_post = MPs_set_post[task_idx]
    for dim_idx in range(joint_num):
        fig = plt.figure(dim_idx+num)
        fig.suptitle('Prior_distribution_' + info + '_' + task_name[task_idx]+"_dim_"+str(dim_idx))
        if method == "ipromp":
            MPs_post.promps[dim_idx + info_n_idx[info][0]].plot_prior_distribution()
        elif method == "promp":
            MPs_post.promps[dim_idx].plot_nUpdated_distribution()


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

def plot_single_dim_traj_diff_task(num=0, data_type=None,dim_idx=0):
    for task_idx, MPs in enumerate(MPs_set):
        fig = plt.figure(task_idx + num)
        fig.suptitle(info+"_"+data_type+"_dimIndex_"+str(dim_idx))
        for demo_idx in range(MPs.num_demos):
            ax = fig.add_subplot(1, 1, 1 )
            if data_type == "raw":
                data = datasets_raw[task_idx][demo_idx][info][:, dim_idx]
            elif data_type == "filter":
                data = datasets_filtered[task_idx][demo_idx][info][:, dim_idx]
            elif data_type == "norm":
                data = datasets_norm[task_idx][demo_idx][info][:, dim_idx]
            plt.plot(np.array(range(len(data)))/100.0, data)
            ax.set_xlabel('t(s)')
            ax.set_ylabel('y(m)')
            plt.legend()

def plot_single_dim_traj_same_task(num=0, data_type=None,task_idx=0):
        for dim_idx in range(joint_num):
            fig = plt.figure(dim_idx + num)
            ax = fig.add_subplot(1, 1, 1 )
            fig.suptitle(info+"_"+data_type+"_taskIndex_"+task_name[task_idx]+"_dim_"+str(dim_idx))
            for demo_idx in range(MPs_set[task_idx].num_demos):      
                if data_type == "raw":
                    data = datasets_raw[task_idx][demo_idx][info][:, dim_idx]
                elif data_type == "filter":
                    data = datasets_filtered[task_idx][demo_idx][info][:, dim_idx]
                elif data_type == "norm":
                    data = datasets_norm[task_idx][demo_idx][info][:, dim_idx]
                plt.plot(np.array(range(len(data)))/100.0, data)
                ax.set_xlabel('t(s)')
                ax.set_ylabel('y(m)')
            plt.legend()

def plot_MPs_gen(num=0):
    for task_idx, demo_list in enumerate(data_index):
        fig = plt.figure(task_idx + num,figsize=(8, 6), dpi=80, facecolor='w', edgecolor='w')
        fig.suptitle(task_name[task_idx],fontsize=20)
        ax = fig.gca(projection='3d')
        if method == "promp":
            robot_gt = ground_truth['left_joints']
            robot_pred = MP_traj_offline[task_idx]
            ax.plot(robot_gt[:, 0], robot_gt[:, 1], robot_gt[:, 2],
                    linewidth=3, linestyle='-', color='r', alpha=1.0, label='Robot movement groundtruth')
            ax.plot(robot_pred[:, 0], robot_pred[:, 1], robot_pred[:, 2],
                    linewidth=3, linestyle='-', color='g', alpha=1.0, label='Robot movement prediction')   
            
        elif method == "ipromp":
            robot_gt = ground_truth['left_joints']
            robot_pred = MP_traj_offline[task_idx][1]
            human_gt = ground_truth['left_hand']
            human_pred = MP_traj_offline[task_idx][0]
            ax.plot(robot_gt[:, 0], robot_gt[:, 1], robot_gt[:, 2],
                    linewidth=4, linestyle='-', color='r', alpha=1.0, label='Robot movement groundtruth')
            ax.plot(robot_pred[:, 0], robot_pred[:, 1], robot_pred[:, 2],
                    linewidth=4, linestyle='-', color='g', alpha=1.0, label='Robot movement prediction')  
            ax.plot(human_gt[:, 0], human_gt[:, 1], human_gt[:, 2],
                    linewidth=4, linestyle='-', color='r', alpha=1.0, label='Human movement groundtruth')
            ax.plot(human_pred[:, 0], human_pred[:, 1], human_pred[:, 2],
                    linewidth=4, linestyle='-', color='g', alpha=1.0, label='Human movement prediction')  
       
        ax.plot(viapoint[:, 0], viapoint[:, 1], viapoint[:, 2],
                'o', markersize=15, label='Via_points', alpha=1.0,
                markerfacecolor='none', markeredgewidth=4.0,  markeredgecolor='r')  

        label_font = 20
        label_pad = 25
        tick_fontsize = 20
        tick_pad = 10
        legend_fontsize = 20

        ax.set_xlabel("X(m)",fontsize=label_font,labelpad=label_pad) # set axis label
        ax.set_ylabel("Y(m)",fontsize=label_font,labelpad=label_pad)               
        ax.set_zlabel("Z(m)",fontsize=label_font,labelpad=label_pad)

        ax.xaxis.set_tick_params(labelsize=tick_fontsize,pad=tick_pad)
        ax.yaxis.set_tick_params(labelsize=tick_fontsize,pad=tick_pad)
        ax.zaxis.set_tick_params(labelsize=tick_fontsize,pad=tick_pad)
        ax.legend(fontsize=legend_fontsize)


def plot_MPs_offline(num=0):
    if method=="promp":
        plot_3d_filtered_r_traj(num)
        plot_MPs_gen(num)
    elif method=="ipromp":
        plot_3d_filtered_r_traj(num)
        plot_MPs_gen(num)
        plot_3d_filtered_h_traj(num)


def main():
    # conf_zh("Droid Sans Fallback")
    # plt.close('all')
    # plot_raw_data(0)
    # plot_norm_data(0)
    # plot_preproc_data(10)
    # plot_filtered_data(10)
    # plot_single_dim_traj_diff_task(data_type="norm",dim_idx=1) # raw, filter,norm
    # plot_single_dim_traj_same_task(data_type="norm",task_idx=1) 
    # plot_single_dim_prior_distribution_diff_task(dim_idx=0)
    # plot_single_dim_prior_distribution_same_task(num=0,task_idx=0)
    # plot_single_dim_post_distribution_diff_task(dim_idx=0)
    # plot_single_dim_post_distribution_same_task(num=0,task_idx=0)
    # plot_alpha()
    # plot_raw_data_index()
    # plot_filter_data_index(20)

    # plot_3d_raw_traj(10)
    # plot_3d_gen_r_traj_online(10)
    plot_MPs_offline(0)
    # pairs_online(10)
    # plt.yticks(fontsize=20)
    # plt.xticks(fontsize=20)
    plt.legend()
    # plt.savefig('fig3',format='eps')
    plt.show()


if __name__ == '__main__':
    main()
