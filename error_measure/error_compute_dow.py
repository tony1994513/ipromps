#!/usr/bin/python
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import os
import ConfigParser
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from numpy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D
from sklearn.externals import joblib
import copy
import dtw
from collections import deque
import  random
from scipy.interpolate import griddata
import ipdb


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
datasets_norm_preproc = joblib.load(os.path.join(datasets_path, 'pkl/datasets_norm_preproc.pkl'))
datasets_norm = joblib.load(os.path.join(datasets_path, 'pkl/datasets_norm.pkl'))

num_joints = cp_models.getint('datasets', 'num_joints')
num_obs_joints = cp_models.getint('datasets', 'num_obs_joints')

# read datasets cfg file
cp_datasets = ConfigParser.SafeConfigParser()
cp_datasets.read(os.path.join(datasets_path, './info/cfg/datasets.cfg'))
# read datasets params
data_index_sec = cp_datasets.items('index_17')
data_index = [map(int, task[1].split(',')) for task in data_index_sec]






def error_dist(pred, ground_truth):
    "calculate square root distance error along a trajectory"
    dist = deque()
    for p1, p2 in zip(pred, ground_truth):
        dist.append(np.sqrt(np.sum((p1-p2)**2)))
    dist = np.asarray(dist)
    return np.array(dist).mean()


def dtw_fuc(pred):
    ground_truth = datasets_norm_preproc[task_id][test_index]['left_joints'][:,0:3]
    dist, cost, acc, path = dtw.dtw(pred, ground_truth, dist=lambda x, y: np.linalg.norm(x - y, ord=2))
    return dist


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




def main(ipromps_set, test_index,task_id, num_alpha_candidate):


    # obs_ratio_time_1 = np.array([0.0, 0.2,0.4,0.6,0.8])
    # obs_ratio_time_2 = np.array([0.2, 0.4,0.6,0.8,1.0])

    obs_ratio_time_1 = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    obs_ratio_time_2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # obs_ratio_time_1 = np.array([0.0])
    # obs_ratio_time_2 = np.array([1.0])
    # read test data
    obs_data_dict = datasets_raw[task_id][data_index[task_id][test_index[0]]]
    left_hand = obs_data_dict['left_hand']
    left_joints = obs_data_dict['left_joints']
    obs_data = np.hstack([left_hand, left_joints])
    timestamp = obs_data_dict['stamp']    

    # filter the data
    obs_data = gaussian_filter1d(obs_data.T, sigma=sigma).T


    # preprocessing for the data
    obs_data_post_arr = ipromps_set[0].min_max_scaler.transform(obs_data)
    
    # consider the unobserved info
    obs_data_post_arr[:, num_obs_joints:] = 0.0    


    obs_ratio_time = np.column_stack((obs_ratio_time_1,obs_ratio_time_2))

    mean_blending_full = deque() 
    std_blending_full = deque() 

    # sparse human observation data
    for obs_idx,obs_ratio in enumerate(obs_ratio_time):
        obs_ratio_1 = int(obs_ratio_time[obs_idx][0] * len(obs_data_post_arr))
        obs_ratio_2 = int(obs_ratio_time[obs_idx][1] * len(obs_data_post_arr))
        tmp = obs_ratio_2 % 5
        if tmp == 0:
            obs_ratio_2 = obs_ratio_2 -tmp - 1
        obs_ratio_2 = obs_ratio_2 -tmp

        sampled_list = np.linspace(obs_ratio_1,obs_ratio_2,5).astype("int")
        obs_data_post_arr_ = obs_data_post_arr[sampled_list, :]
        timestamp_ = timestamp[sampled_list]

        # ipdb.set_trace()
        # phase estimation
        if obs_idx == 0 :
            # print('Phase estimating...')
            alpha_max_list = []
            for ipromp in ipromps_set:
                alpha_temp = ipromp.alpha_candidate(num_alpha_candidate)
                idx_max = ipromp.estimate_alpha(alpha_temp, obs_data_post_arr_, timestamp_)
                alpha_max_list.append(alpha_temp[idx_max]['candidate'])
                ipromp.set_alpha(alpha_temp[idx_max]['candidate'])
                # print("alpha is %s" %alpha_temp[idx_max]['candidate'] )

        # task recognition
        # print('Adding via points in each trained model...')
        for task_idx, ipromp in enumerate(ipromps_set):
            for idx in range(len(timestamp_)):
                # ipromp.add_viapoint(timestamp_[idx] / alpha_max_list[task_idx], obs_data_post_arr_[idx, :])
                ipromp.add_viapoint(timestamp_[idx]/alpha_max_list[task_id], obs_data_post_arr_[idx, :])
            ipromp.param_update(unit_update=True)

        # print('Computing the likelihood for each model under observations...')
        
        prob_task = []
        for ipromp in ipromps_set:
            prob_task_temp = ipromp.prob_obs()
            prob_task.append(prob_task_temp)
        idx_max_prob = np.argmax(prob_task) 
        # print("prob of tasks are %s" %prob_task)
        # print('The max fit model index is task %s' % (idx_max_prob))

        robot_promps = [ipromps_set[idx_max_prob].promps[3],ipromps_set[idx_max_prob].promps[4],ipromps_set[idx_max_prob].promps[5]]
        Phi = ipromps_set[idx_max_prob].Phi

        

   
        num_of_point = int(obs_ratio_time_2[obs_idx]*10 - obs_ratio_time_1[obs_idx]*10)*10
        # print "num_of_point %s" %num_of_point
        phase_increase ,phase_decrease = sigmoid(obs_ratio_time_1[obs_idx],obs_ratio_time_2[obs_idx],num_of_point)

        index_0 = int(obs_ratio_time_1[obs_idx]*100)
        index_1 = int(obs_ratio_time_2[obs_idx]*100)


        if obs_idx == 0:
            # print("get into obs_idx %s," %obs_idx  )
            # init robot distribution 
            init_seed = random.randint(0,2)
            robot_promps_init = [ipromps_set[init_seed].promps[3],ipromps_set[init_seed].promps[4],ipromps_set[init_seed].promps[5]]
            mean0_list = deque(); std0_list = deque()
            mean1_list = deque();std1_list = deque() 

            for init_promp in robot_promps_init:
                mean0_list.append(np.dot(Phi.T,init_promp.meanW))
                std0_list.append(2 * np.sqrt(np.diag(np.dot(Phi.T, np.dot(init_promp.sigmaW, Phi)))))
            
            # idx_max_prob = 2
            # robot_promps = [ipromps_set[idx_max_prob].promps[3],ipromps_set[idx_max_prob].promps[4],ipromps_set[idx_max_prob].promps[5]]
            for robot_promp in robot_promps:
                mean1_list.append(np.dot(Phi.T,robot_promp.meanW_nUpdated))
                std1_list.append(2 * np.sqrt(np.diag(np.dot(Phi.T, np.dot(robot_promp.sigmaW_nUpdated, Phi)))))    

            for mean0,std0,mean1,std1 in zip(mean0_list,std0_list, mean1_list,std1_list):
                mean_blending, std_blending = traj_blending_fuc(mean0[index_0:index_1], std0[index_0:index_1], mean1[index_0:index_1], std1[index_0:index_1],phase_increase ,phase_decrease)
                mean_blending_full.append(mean_blending)
                std_blending_full.append(std_blending)

        if obs_idx == 1:
            # print("get into obs_idx %s," %obs_idx  )
            mean2_list = deque(); std2_list = deque()
            # updated robot distribution after getting new obervations

            for robot_promp in robot_promps:
                mean2_list.append(np.dot(Phi.T,robot_promp.meanW_nUpdated))
                std2_list.append(2 * np.sqrt(np.diag(np.dot(Phi.T, np.dot(robot_promp.sigmaW_nUpdated, Phi)))))

            for mean0,std0,mean1,std1 in zip(mean1_list,std1_list, mean2_list,std2_list):
                mean_blending, std_blending = traj_blending_fuc(mean0[index_0:index_1], std0[index_0:index_1], mean1[index_0:index_1], std1[index_0:index_1],phase_increase ,phase_decrease)
                mean_blending_full.append(mean_blending)
                std_blending_full.append(std_blending)    

        if obs_idx == 2:
            # print("get into obs_idx %s," %obs_idx  )
            mean3_list = deque(); std3_list = deque()
            # updated robot distribution after getting new obervations
            for robot_promp in robot_promps:
                mean3_list.append(np.dot(Phi.T,robot_promp.meanW_nUpdated))
                std3_list.append(2 * np.sqrt(np.diag(np.dot(Phi.T, np.dot(robot_promp.sigmaW_nUpdated, Phi)))))
            # ipdb.set_trace()
            for mean0,std0,mean1,std1 in zip(mean2_list,std2_list, mean3_list,std3_list):
                mean_blending, std_blending = traj_blending_fuc(mean0[index_0:index_1], std0[index_0:index_1], mean1[index_0:index_1], std1[index_0:index_1],phase_increase ,phase_decrease)
                mean_blending_full.append(mean_blending)
                std_blending_full.append(std_blending) 

            
        if obs_idx == 3:
            # print("get into obs_idx %s," %obs_idx  )
            mean4_list = deque(); std4_list = deque()
            # updated robot distribution after getting new obervations
            
            for robot_promp in robot_promps:
                mean4_list.append(np.dot(Phi.T,robot_promp.meanW_nUpdated))
                std4_list.append(2 * np.sqrt(np.diag(np.dot(Phi.T, np.dot(robot_promp.sigmaW_nUpdated, Phi)))))

            for mean0,std0,mean1,std1 in zip(mean3_list,std3_list, mean4_list,std4_list):
                mean_blending, std_blending = traj_blending_fuc(mean0[index_0:index_1], std0[index_0:index_1], mean1[index_0:index_1], std1[index_0:index_1],phase_increase ,phase_decrease)
                mean_blending_full.append(mean_blending)
                std_blending_full.append(std_blending) 

        if obs_idx == 4:
            # print("get into obs_idx %s," %obs_idx  )
            mean5_list = deque(); std5_list = deque()
            # updated robot distribution after getting new obervations
            for robot_promp in robot_promps:
                mean5_list.append(np.dot(Phi.T,robot_promp.meanW_nUpdated))
                std5_list.append(2 * np.sqrt(np.diag(np.dot(Phi.T, np.dot(robot_promp.sigmaW_nUpdated, Phi)))))

            for mean0,std0,mean1,std1 in zip(mean4_list,std4_list, mean5_list,std5_list):
                mean_blending, std_blending = traj_blending_fuc(mean0[index_0:index_1], std0[index_0:index_1], mean1[index_0:index_1], std1[index_0:index_1],phase_increase ,phase_decrease)
                mean_blending_full.append(mean_blending)
                std_blending_full.append(std_blending) 

        if obs_idx == 5:
            # ipdb.set_trace()
            # print("get into obs_idx %s," %obs_idx  )
            mean6_list = deque(); std6_list = deque()
            # updated robot distribution after getting new obervations
            for robot_promp in robot_promps:
                mean6_list.append(np.dot(Phi.T,robot_promp.meanW_nUpdated))
                std6_list.append(2 * np.sqrt(np.diag(np.dot(Phi.T, np.dot(robot_promp.sigmaW_nUpdated, Phi)))))

            for mean0,std0,mean1,std1 in zip(mean5_list,std5_list, mean6_list,std6_list):
                mean_blending, std_blending = traj_blending_fuc(mean0[index_0:index_1], std0[index_0:index_1], mean1[index_0:index_1], std1[index_0:index_1],phase_increase ,phase_decrease)
                mean_blending_full.append(mean_blending)
                std_blending_full.append(std_blending) 

        if obs_idx == 6:
            # print("get into obs_idx %s," %obs_idx  )
            mean7_list = deque(); std7_list = deque()
            # updated robot distribution after getting new obervations
            for robot_promp in robot_promps:
                mean7_list.append(np.dot(Phi.T,robot_promp.meanW_nUpdated))
                std7_list.append(2 * np.sqrt(np.diag(np.dot(Phi.T, np.dot(robot_promp.sigmaW_nUpdated, Phi)))))

            for mean0,std0,mean1,std1 in zip(mean6_list,std6_list, mean7_list,std7_list):
                mean_blending, std_blending = traj_blending_fuc(mean0[index_0:index_1], std0[index_0:index_1], mean1[index_0:index_1], std1[index_0:index_1],phase_increase ,phase_decrease)
                mean_blending_full.append(mean_blending)
                std_blending_full.append(std_blending) 

        if obs_idx == 7:
            # print("get into obs_idx %s," %obs_idx  )
            mean8_list = deque(); std8_list = deque()
            # updated robot distribution after getting new obervations
            for robot_promp in robot_promps:
                mean8_list.append(np.dot(Phi.T,robot_promp.meanW_nUpdated))
                std8_list.append(2 * np.sqrt(np.diag(np.dot(Phi.T, np.dot(robot_promp.sigmaW_nUpdated, Phi)))))

            for mean0,std0,mean1,std1 in zip(mean7_list,std7_list, mean8_list,std8_list):
                mean_blending, std_blending = traj_blending_fuc(mean0[index_0:index_1], std0[index_0:index_1], mean1[index_0:index_1], std1[index_0:index_1],phase_increase ,phase_decrease)
                mean_blending_full.append(mean_blending)
                std_blending_full.append(std_blending)       

        if obs_idx == 8:
            # print("get into obs_idx %s," %obs_idx  )
            mean9_list = deque(); std9_list = deque()
            # updated robot distribution after getting new obervations
            for robot_promp in robot_promps:
                mean9_list.append(np.dot(Phi.T,robot_promp.meanW_nUpdated))
                std9_list.append(2 * np.sqrt(np.diag(np.dot(Phi.T, np.dot(robot_promp.sigmaW_nUpdated, Phi)))))

            for mean0,std0,mean1,std1 in zip(mean8_list,std8_list, mean9_list,std9_list):
                mean_blending, std_blending = traj_blending_fuc(mean0[index_0:index_1], std0[index_0:index_1], mean1[index_0:index_1], std1[index_0:index_1],phase_increase ,phase_decrease)
                mean_blending_full.append(mean_blending)
                std_blending_full.append(std_blending)    

        if obs_idx == 9:
            # print("get into obs_idx %s," %obs_idx  )
            mean10_list = deque(); std10_list = deque()
            # updated robot distribution after getting new obervations
            for robot_promp in robot_promps:
                mean10_list.append(np.dot(Phi.T,robot_promp.meanW_nUpdated))
                std10_list.append(2 * np.sqrt(np.diag(np.dot(Phi.T, np.dot(robot_promp.sigmaW_nUpdated, Phi)))))

            for mean0,std0,mean1,std1 in zip(mean9_list,std9_list, mean10_list,std10_list):
                mean_blending, std_blending = traj_blending_fuc(mean0[index_0:index_1], std0[index_0:index_1], mean1[index_0:index_1], std1[index_0:index_1],phase_increase ,phase_decrease)
                mean_blending_full.append(mean_blending)
                std_blending_full.append(std_blending)    


    blended_mean_x = deque()
    blended_mean_y = deque()
    blended_mean_z = deque()
    
    for idx, blended_mean in enumerate(mean_blending_full):
        if idx % 3 == 0:
            blended_mean_x.append(blended_mean)
        if idx % 3 == 1:
            blended_mean_y.append(blended_mean)
        if idx % 3 == 2:
            blended_mean_z.append(blended_mean)
    # ipdb.set_trace()
    blended_mean_x_full = np.concatenate(blended_mean_x)
    blended_mean_y_full = np.concatenate(blended_mean_y)
    blended_mean_z_full = np.concatenate(blended_mean_z)
    tmp = np.zeros((100,10))
    tmp[:,3] = blended_mean_x_full
    tmp[:,4] = blended_mean_y_full
    tmp[:,5] = blended_mean_z_full

    pred = ipromps_set[0].min_max_scaler.inverse_transform(tmp)[:,3:6]



    # ground_truth = datasets_norm_preproc[task_id][0]['left_joints'][:,0:3]
    
    # ground_truth = left_joints[:,0:3]
    # ipdb.set_trace()
    ground_truth = datasets_norm[task_id][12]['left_joints'][:,0:3]
    # robot_positioning_error = np.sqrt(np.sum(np.square(pred[-1] - ground_truth[-1])))
    robot_positioning_error = error_dist(pred,ground_truth)
    phase_error = alpha_max_list[task_id] - datasets_norm_preproc[task_id][test_index[0]]['alpha']
    # ipdb.set_trace()
    return [ robot_positioning_error, idx_max_prob, np.abs(phase_error)]

    
    # print "robot_positioning_error %s" %robot_positioning_error

    # ipdb.set_trace()    
    # fig = plt.figure(0)
    # ax = fig.gca(projection='3d') 
    # ax.plot(mean0_list[0],mean0_list[1],mean0_list[2],linestyle='--',label="init robot trajectory")   
    # ax.plot(mean1_list[0],mean1_list[1],mean1_list[2],linestyle='--',label="robot trajectory " +str(1)) 
    # ax.plot(mean2_list[0],mean2_list[1],mean2_list[2],linestyle='--',label="robot trajectory " +str(2)) 
    # ax.plot(mean3_list[0],mean3_list[1],mean3_list[2],linestyle='--',label="robot trajectory " +str(3)) 
    # ax.plot(mean4_list[0],mean4_list[1],mean4_list[2],linestyle='--',label="robot trajectory " +str(4)) 
    # ax.plot(mean5_list[0],mean5_list[1],mean5_list[2],linestyle='--',label="robot trajectory " +str(5)) 
    # # # ax.plot(mean6_list[0],mean6_list[1],mean6_list[2],linestyle='--',label="robot trajectory " +str(6)) 
    # # # ax.plot(mean7_list[0],mean7_list[1],mean7_list[2],linestyle='--',label="robot trajectory " +str(7)) 
    # # # ax.plot(mean8_list[0],mean8_list[1],mean8_list[2],linestyle='--')     
    # ax.plot(pred[:,0],pred[:,1],pred[:,2],linewidth=2,label="blended robot trajectory",color="r")
    # ax.plot(ground_truth[:,0],ground_truth[:,1],ground_truth[:,2],linewidth=2,label="ground truth",color="b")
    # ax.set_xlabel('X Label (m)')
    # ax.set_ylabel('Y Label (m)')
    # ax.set_zlabel('Z Label (m)')
    # # ax.set_title("Robot Trajectory Blending")
    # ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # # fig = plt.figure(1)
    # # plt.plot(phase_increase)
    # # plt.plot(phase_decrease)
    # ax.legend(fontsize='xx-large')
    # plt.show()




if __name__ == '__main__':
    main()