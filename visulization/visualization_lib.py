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
MPs_post_set = joblib.load(os.path.join(datasets_path, 'pkl/'+method+'_post_offline.pkl'))
datasets_raw = joblib.load(os.path.join(datasets_path, 'pkl/'+method+'_datasets_raw.pkl'))
datasets_norm = joblib.load(os.path.join(datasets_path, 'pkl/'+method+'_datasets_norm.pkl'))
datasets_filtered = joblib.load(os.path.join(datasets_path, 'pkl/'+method+'_datasets_filtered.pkl'))
datasets_norm_preproc = joblib.load(os.path.join(datasets_path, 'pkl/'+method+'_datasets_norm_preproc.pkl'))
task_name = joblib.load(os.path.join(datasets_path, 'pkl/task_name_list.pkl'))
[MP_traj_offline, ground_truth, viapoint,gt_time,viapoint_time] = joblib.load(os.path.join(datasets_path, 'pkl/'+method+'_traj_offline.pkl'))

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
    joint_num = cp_models.getint('promp_param', 'num_dim')
elif method == "ipromp":
    info = cp_models.get('visualization', 'info')
    joint_num = cp_models.getint('ipromp_param', 'num_dim')


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

def min_max_inverse_func(MPs,demo_1d):
    temp = np.zeros((demo_1d.shape[0],joint_num))
    temp[:,0] = demo_1d
    temp_inv = MPs.min_max_scaler.inverse_transform(temp)
    return temp_inv[:,0]

def plot_MPs_func(MPs, MPs_post, num=0,task_idx=None,dim_idx=None,prior_distribution_flag=True):
    color = "b";prior_alpha=0.3;post_alpha=0.7;mean_line_width=4;std_times=2
    viapoint_color="r"
    if prior_distribution_flag == True:
        fig = plt.figure(task_idx+num)
        fig.suptitle(method+'Prior_distribution_' + info + '_' + task_name[task_idx]+"_dim_"+str(dim_idx))
        prior_pred_time = np.linspace(0,MPs.mean_alpha, MPs.num_samples)
        mean = np.dot(MPs.Phi.T, MPs.promps[dim_idx].meanW)
        mean = min_max_inverse_func(MPs,mean)
        std = std_times * np.sqrt(np.diag(np.dot(MPs.Phi.T, np.dot(MPs.promps[dim_idx].sigmaW, MPs.Phi))))
        plt.fill_between(prior_pred_time, mean-std, mean+std, color=color, alpha=prior_alpha,label="Prior distribution")
        plt.legend()
    else:
        post_pred_time = np.linspace(0,MPs_post.alpha_fit, MPs_post.num_samples)
        mean_post = np.dot(MPs_post.Phi.T, MPs_post.promps[dim_idx].sigmaW_nUpdated)
        std_post = std_times * np.sqrt(np.diag(np.dot(MPs_post.Phi.T, np.dot(MPs_post.promps[dim_idx].sigmaW_nUpdated, MPs_post.Phi))))
        if method == "promp":
            fig = plt.figure(task_idx+num)
            fig.suptitle(method+'Post_distribution_' + info + '_' + task_name[task_idx]+"_dim_"+str(dim_idx))
            robot_pred = MP_traj_offline[task_idx][:,dim_idx]
            robot_gt = ground_truth['left_joints'][:,dim_idx]
            plt.plot(post_pred_time, robot_pred, label="Robot prediction",linestyle='-', color=color,  linewidth=mean_line_width)
            plt.plot(gt_time, robot_gt, label="Robot gorundtruth",linestyle='-', color="black",  linewidth=mean_line_width)
            plt.fill_between(post_pred_time, robot_pred-std_post, robot_pred+std_post, color=color, alpha=post_alpha,label="Initial robot distribution")
            plt.plot(viapoint_time, viapoint[:,dim_idx], marker="o", markersize=10, color=viapoint_color,
                      label='Via_points', alpha=1.0, markerfacecolor='none', markeredgewidth=4.0,  markeredgecolor='r')  
            plt.legend()

        elif method == "ipromp":
            if dim_idx>= human_index[0] and dim_idx<human_index[1]:
                fig = plt.figure(task_idx+num+dim_idx)
                fig.suptitle(method+'Post_distribution_' + info + '_' + task_name[task_idx]+"_dim_"+str(dim_idx))
                human_gt = ground_truth['left_hand'][:,dim_idx]
                human_pred = MP_traj_offline[task_idx][0][:,dim_idx]
                plt.plot(gt_time, human_gt, label="Human gorundtruth",linestyle='-', color="black",  linewidth=mean_line_width)
                plt.plot(post_pred_time, human_pred, label="Human prediction",linestyle='-', color=color,  linewidth=mean_line_width)
                plt.fill_between(post_pred_time, human_pred-std_post, human_pred+std_post, color=color, alpha=post_alpha,label="Post distribution")
                # ipdb.set_trace()
                plt.plot(viapoint_time, viapoint[:,dim_idx], marker="o", markersize=10, color=viapoint_color,
                        label='Via_points', alpha=1.0, markerfacecolor='none', markeredgewidth=4.0,  markeredgecolor='r')
                plt.legend()
            else:
                fig = plt.figure(task_idx+num+dim_idx)
                fig.suptitle(method+'Post_distribution_' + info + '_' + task_name[task_idx]+"_dim_"+str(dim_idx))
                robot_gt = ground_truth['left_joints'][:,dim_idx-human_index[1]]
                robot_pred = MP_traj_offline[task_idx][1][:,dim_idx-human_index[1]]
                plt.plot(post_pred_time, robot_pred, label="Robot prediction",linestyle='-', color=color,  linewidth=mean_line_width)
                plt.plot(gt_time, robot_gt, label="Robot gorundtruth",linestyle='-', color="black",  linewidth=mean_line_width)
                plt.fill_between(post_pred_time, robot_pred-std, robot_pred+std, color=color, alpha=post_alpha,label="Post distribution")
                plt.legend()

def plot_single_dim_distribution(num=0,dim_idx=0,task_idx=0,diffTask_flag=None,prior_distribution_flag=None):
    if diffTask_flag == True:
        for task_idx, MPs in enumerate(MPs_set):
            Mps_post = MPs_post_set[task_idx]
            plot_MPs_func(MPs, Mps_post, num=num,prior_distribution_flag=prior_distribution_flag,task_idx=task_idx,dim_idx=dim_idx)
    else:
        MPs = MPs_set[task_idx]
        Mps_post = MPs_post_set[task_idx]
        for dim_idx in range(joint_num):
            plot_MPs_func(MPs, Mps_post,num=num, prior_distribution_flag=prior_distribution_flag, task_idx=task_idx,dim_idx=dim_idx)
        

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
        plt.scatter(h, pdf, s=100)
        xx = np.linspace(h_mean-3*h_std, h_mean+3*h_std, 100)
        yy = stats.norm.pdf(xx, h_mean, h_std)
        plt.plot(xx, yy, linewidth=2, color='r', markersize=10, alpha=0.8)
        plt.xlabel('Phase factor')
        plt.ylabel('Probability')


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

def plot_single_dim_traj(num=0,dim_idx=0,task_idx=0, data_type=None, diffTask_flag=None):
    if diffTask_flag == True:
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
                time = np.linspace(0,MPs.alpha[demo_idx],len(data))
                if demo_idx == 0:
                    plt.plot(time, data, color=color,label="Sampled data")
                else:
                    plt.plot(time, data, color="grey")
                ax.set_xlabel('t(s)')
                ax.set_ylabel('y(m)')
                plt.legend()
    else:
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
                        time = np.linspace(0,MPs_set[task_idx].alpha[demo_idx],len(data))
                        if demo_idx == 0:
                            plt.plot(time, data, color="grey",label="Sampled data")
                        else:
                            plt.plot(time, data, color="grey")
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
    # plot_single_dim_traj(num=0,dim_idx=0, diffTask_flag=False,data_type="norm") # raw, filter,norm
    plot_single_dim_distribution(num=0,dim_idx=0,task_idx=0,diffTask_flag=True,prior_distribution_flag=True)
    plot_single_dim_distribution(num=0,dim_idx=0,task_idx=0,diffTask_flag=True,prior_distribution_flag=False)

    # plot_alpha()
    # plot_raw_data_index()
    # plot_filter_data_index(20)

    # plot_3d_raw_traj(10)
    # plot_3d_gen_r_traj_online(10)
    # plot_MPs_offline(0)
    # pairs_online(10)
    # plt.yticks(fontsize=20)
    # plt.xticks(fontsize=20)
    plt.legend()
    # plt.savefig('fig3',format='eps')
    plt.show()


if __name__ == '__main__':
    main()
