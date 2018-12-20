#!/usr/bin/python
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import os
import ConfigParser
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from numpy.linalg import inv
import matplotlib.pyplot as plt
from collections import deque


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
num_joints = cp_models.getint('datasets', 'num_dim')
num_obs_joints = cp_models.getint('datasets', 'num_obs_dim')
from scipy.interpolate import griddata

# read datasets cfg file
cp_datasets = ConfigParser.SafeConfigParser()
cp_datasets.read(os.path.join(datasets_path, './info/cfg/datasets.cfg'))
# read datasets params
data_index_sec = cp_datasets.items('index_17')
data_index = [map(int, task[1].split(',')) for task in data_index_sec]
import ipdb

def sigmoid(t1,t2,len_num=21):
    x = np.linspace(-5,5,201)
    grid = np.linspace(t1, t2, len_num)
    time_stamp = np.linspace(t1, t2,60)
    i = -3
    k = 5
    y1 = 1 / (1 + np.exp(-k*(x-i)))[10:70]
    y_increase = griddata(time_stamp, y1, grid, method='linear')
    y2 = 1 / (1 + np.exp(k*(x-i)))[10:70]
    y_decrease = griddata(time_stamp, y2, grid, method='linear')
    return y_increase, y_decrease



def traj_blending_fuc(mean0,std0,mean1,std1,phase0,phase1):
    if len(mean0) != len(mean1) != len(phase0):
        print "size is diff"
    sigma0_activated = std0 / phase1
    sigma1_activated = std1 / phase0
    sigma_norm = sigma0_activated * sigma1_activated
    sigma_denorm = sigma0_activated + sigma1_activated
    sigma_blending = sigma_norm / sigma_denorm 
    mean_divd_up = mean0*sigma1_activated + mean1*sigma0_activated
    mean_blending =  mean_divd_up / sigma_denorm
    return mean_blending,sigma_blending

def main():
    task_id = 2
    test_index = 4

    obs_ratio_time_1 = np.array([0.0,0.2,0.4,0.6,0.8])
    obs_ratio_time_2 = np.array([0.2,0.4,0.6,0.8,1.0])
    obs_ratio_time = np.column_stack((obs_ratio_time_1,obs_ratio_time_2))

    # read test data
    obs_data_dict = datasets_raw[task_id][test_index]

    left_hand = obs_data_dict['left_hand']
    left_joints = obs_data_dict['left_joints']

    obs_data = np.hstack([left_hand, left_joints])
    timestamp = obs_data_dict['stamp']

    # preprocessing for the data
    obs_data_post_arr = ipromps_set[0].min_max_scaler.transform(obs_data)
    # consider the unobserved info
    obs_data_post_arr[:, num_obs_joints:] = 0.0
    
    # store blended robot trajectory
    mean_blending_list = deque()
    std_blending_list = deque()
    phase_0 = deque()
    phase_1 = deque()
    phase_2 = deque()
    phase_3 = deque()

    # sparse human observation data
    for obs_idx,obs_ratio in enumerate(obs_ratio_time):
        obs_ratio_1 = int(obs_ratio_time[obs_idx][0] * len(obs_data_post_arr))
        obs_ratio_2 = int(obs_ratio_time[obs_idx][1] * len(obs_data_post_arr))
        # sample data
        tmp = obs_ratio_2 % 3
        if tmp == 0:
            obs_ratio_2 = obs_ratio_2 -tmp - 1
        obs_ratio_2 = obs_ratio_2 -tmp
        sampled_list = np.linspace(obs_ratio_1,obs_ratio_2,3).astype("int")
        obs_data_post_arr =obs_data_post_arr[sampled_list,:]
        timestamp = timestamp[sampled_list]

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
            ipromp.param_update(unit_update=True,update_time=obs_idx)
        print('Computing the likelihood for each model under observations...')

        prob_task = []
        for ipromp in ipromps_set:
            prob_task_temp = ipromp.prob_obs()
            prob_task.append(prob_task_temp)
        idx_max_prob = np.argmax(prob_task)
        print('The max fit model index is task %s' % task_name[idx_max_prob])

        #####################

        robot_promps = ipromps_set[idx_max_prob].promps[3]
       
        Phi = ipromps_set[idx_max_prob].Phi

        meanW_nUpdated = robot_promps.meanW_nUpdated   
        sigmaW_nUpdated = robot_promps.sigmaW_nUpdated
        num_of_point = (obs_ratio_time_2[obs_idx] - obs_ratio_time_1[obs_idx])*100 
        ipdb.set_trace()
        phase_increase ,phase_decrease = sigmoid(obs_ratio_time_1[obs_idx],obs_ratio_time_2[obs_idx],num_of_point)
        
        if obs_idx == 0:
            phase_0.append(phase_decrease)
            phase_1.append(phase_increase)
        if obs_idx == 1:
            phase_1.append(phase_decrease)
            phase_2.append(phase_increase)
        if obs_idx == 2:
            phase_2.append(phase_decrease)
            phase_3.append(phase_increase)


        index_0 = int(obs_ratio_time_1[obs_idx]*100)
        index_1 = int(obs_ratio_time_2[obs_idx]*100)
        

        if obs_idx == 0:
            
            meanW0 = robot_promps.meanW
            sigmaW0 = robot_promps.sigmaW
            # sigma3 = inv(inv(sigmaW0)+ inv(sigmaW_nUpdated))
            # mean3 = np.dot(np.dot(sigma3, inv(sigmaW0)),meanW0) + np.dot(np.dot(sigma3, inv(sigmaW_nUpdated)),meanW_nUpdated)
            # ipdb.set_trace()

            mean0 = np.dot(Phi.T,meanW0)
            std0 = 2 * np.sqrt(np.diag(np.dot(Phi.T, np.dot(sigmaW0, Phi))))

            mean1 =  np.dot(Phi.T,meanW_nUpdated)
            std1 = 2 * np.sqrt(np.diag(np.dot(Phi.T, np.dot(sigmaW_nUpdated, Phi))))

            num_of_point = (obs_ratio_time_2[obs_idx] - obs_ratio_time_1[obs_idx])*100
        
            phase_increase ,phase_decrease = sigmoid(obs_ratio_time_1[obs_idx],obs_ratio_time_2[obs_idx],num_of_point)

            mean0_ = mean0[index_0:index_1]
            std0_ = std0[index_0:index_1]

            mean1_ = mean1[index_0:index_1]
            std1_ = std1[index_0:index_1]

            mean_blending, sigma_blending = traj_blending_fuc(mean0_,std0_,mean1_,std1_,phase_increase ,phase_decrease)
            mean_blending_list.append(np.copy(mean_blending))
            std_blending_list.append(np.copy(sigma_blending))

        if obs_idx == 1:

            mean2 =  np.dot(Phi.T,meanW_nUpdated)
            std2 = 2 * np.sqrt(np.diag(np.dot(Phi.T, np.dot(sigmaW_nUpdated, Phi))))

            mean1_ = mean1[index_0:index_1]
            std1_ = std1[index_0:index_1]
            mean2_ = mean2[index_0:index_1]
            std2_ = std2[index_0:index_1]

            mean_blending, sigma_blending = traj_blending_fuc(mean1_,std1_,mean2_,std2_,phase_increase ,phase_decrease)
            mean_blending_list.append(np.copy(mean_blending))
            std_blending_list.append(np.copy(sigma_blending))

        if obs_idx == 2:

            mean3 =  np.dot(Phi.T,meanW_nUpdated)
            std3 = 2 * np.sqrt(np.diag(np.dot(Phi.T, np.dot(sigmaW_nUpdated, Phi))))

            mean2_ = mean2[index_0:index_1]
            std2_ = std2[index_0:index_1]
            mean3_ = mean2[index_0:index_1]
            std3_ = std2[index_0:index_1]
            
            mean_blending, sigma_blending = traj_blending_fuc(mean2_,std2_,mean3_,std3_,phase_increase ,phase_decrease)
            mean_blending_list.append(np.copy(mean_blending))
            std_blending_list.append(np.copy(sigma_blending))

            
    
    # ipdb.set_trace()
    t = np.linspace(0,0.9,90)
    mean_blending_proc = np.concatenate((mean_blending_list[0],mean_blending_list[1],mean_blending_list[2]))
    std_blending_proc = np.concatenate((std_blending_list[0],std_blending_list[1],std_blending_list[2]))


    phase_0_proc =  np.concatenate(phase_0)
    phase_1_proc =  np.concatenate((phase_1[0],phase_1[1]))
    phase_2_proc =  np.concatenate((phase_2[0],phase_2[1]))
    phase_3_proc =  np.concatenate(phase_3)

    fig = plt.figure(0)
    plt.subplot(2,1,1)
    plt.title("Robot trajectory blending")
    plt.ylabel("X[m]")
    # plt.fill_between(t,mean0-std0,mean0+std0,alpha=0.3,color="b")

    # plt.fill_between(t,mean1-std1,mean1+std1,alpha=0.3,color="r")
    plt.plot(t,mean0[0:90],label="robot trajectory 1",linestyle='--',color="r")
    plt.plot(t,mean1[0:90],label="robot trajectory 2",linestyle='--',color="g")
    plt.plot(t,mean2[0:90],label="robot trajectory 3",linestyle='--',color="b")
    plt.plot(t,mean3[0:90],label="robot trajectory 4",linestyle='--',color="black")
    plt.plot(t,mean_blending_proc,color="lime",label="robot blended trajectory",linewidth=2)

    # plt.fill_between(t, mean_blending_proc-std_blending_proc, mean_blending_proc+std_blending_proc,alpha=0.4,color="lime")

    plt.legend()
    plt.subplot(2,1,2)
    plt.title("Activation function")
    plt.plot(np.linspace(0,0.3,30),phase_0_proc,color="r")
    plt.plot(np.linspace(0,0.6,60),phase_1_proc,color="g")
    plt.plot(np.linspace(0.3,0.9,60),phase_2_proc,color="b")
    plt.plot(np.linspace(0.6,0.9,30),phase_3_proc,color="black")
    plt.show()
            
            
              
       
       
        











if __name__ == '__main__':
    main()