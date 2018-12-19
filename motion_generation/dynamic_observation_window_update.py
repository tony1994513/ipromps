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
import random

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

# read datasets cfg file
cp_datasets = ConfigParser.SafeConfigParser()
cp_datasets.read(os.path.join(datasets_path, './info/cfg/datasets.cfg'))
# read datasets params
data_index_sec = cp_datasets.items('index_17')
data_index = [map(int, task[1].split(',')) for task in data_index_sec]




def sigmoid_increase(trans_x,k=2):
    y_list = []
    k = 3
    T = np.linspace(-5,5,100)
    i_indx = int(T.shape[0] * trans_x)
    i = T[i_indx]
    for t_ in T:
        y =  (1 / (1 + np.exp(-k*(t_-i))))
        y_list.append(y)
    return np.array(y_list)

def sigmoid_decrease(trans_x,k=2):
    y_list = []
    k = 3
    T = np.linspace(-5,5,100)
    i_indx = int(T.shape[0] * trans_x)
    i = T[i_indx]
    for t_ in T:
        y = (1 / (1 + np.exp(k*(t_-i))))
        y_list.append(y)
    return np.array(y_list)

def gau_kl(muP , sigmaP, muq, sigmaq):

    pv = sigmaP
    qv = sigmaq
    pm = muP
    qm = muq
    # Determinants of diagonal covariances pv, qv
    dpv = np.linalg.det(pv)
    dqv = np.linalg.det(qv)
    # Inverse of diagonal covariance qv
    iqv = np.linalg.inv(qv)
    # Difference between means pm, qm
    diff = qm - pm
    result = 0.5 *(np.log(dqv / dpv)            
             + np.trace(np.dot(iqv, pv))                 
             + np.dot(np.dot(diff.T,iqv),diff)      
             - len(pm))            
    return result     

def dtw_fuc(mu1,mu2):
    mu1 = mu1.reshape(-1,1)
    mu2 = mu2.reshape(-1,1)
    dist, cost, acc, path = dtw.dtw(mu1, mu2, dist=lambda x, y: np.linalg.norm(x - y, ord=1))
    return dist

def piecewise_fnc(obs_idx,obs_t):
    k = 10
    x = np.linspace(0,1,100)
    y1 = np.piecewise(x, [(x < obs_t), (x > obs_t+0.1), (x >obs_t)&(x < obs_t+0.1) ], [lambda x: 0.001,lambda x: 0.999,lambda x: k*(x-0.1*obs_idx)])
    y2 = np.piecewise(x, [(x < obs_t), (x > obs_t+0.1), (x >obs_t)&(x < obs_t+0.1) ], [lambda x: 0.999,lambda x: 0.001, lambda x: 1.0 - k*(x-0.1*obs_idx)])
    return np.array(y1),np.array(y2)

def plot_1d_traj(traj1,traj2):
    plt.plot(traj1)
    plt.plot(traj2)  
    plt.show() 

def main():
    task_id = 0
    test_index = 0

    # read test data
    obs_data_dict = datasets_raw[task_id][test_index]

    left_hand = obs_data_dict['left_hand']
    left_joints = obs_data_dict['left_joints']
    obs_data = np.hstack([left_hand, left_joints])
    timestamp = obs_data_dict['stamp']    

    # filter the data
    obs_data = gaussian_filter1d(obs_data.T, sigma=sigma).T
    human_obs = copy.deepcopy(obs_data)

    # preprocessing for the data
    obs_data_post_arr = ipromps_set[0].min_max_scaler.transform(obs_data)
    
    # consider the unobserved info
    obs_data_post_arr[:, num_obs_joints:] = 0.0    

    obs_ratio_time_1 = np.array([0.1,0.2,0.3,0.4,0.5,0.6])
    obs_ratio_time_2 = np.array([0.2,0.3,0.4,0.5,0.6,0.7])
    obs_ratio_time = np.column_stack((obs_ratio_time_1,obs_ratio_time_2))

    
    mean_merge_traj_full = [None,None,None]
    sigma_merge_traj_full = [None,None,None]

    task0_promp0_mean0=[]
    task0_promp0_mean1=[]
    task0_promp0_mean2=[]
    task0_promp1_mean0=[]
    task0_promp1_mean1=[]
    task0_promp1_mean2=[]
    task0_promp2_mean0=[]
    task0_promp2_mean1=[]
    task0_promp2_mean2=[]

    obs_ratio_1 = int(0.1 * obs_data.shape[0])
    for obs_idx,obs_ratio in enumerate(obs_ratio_time):
    
        obs_ratio_2 = int(obs_ratio[1] * obs_data.shape[0])

        seed = range(obs_ratio_1,obs_ratio_2)
        sampled_list = random.sample(seed,3)
        sampled_list = sorted(sampled_list)
        obs_data_post_arr_ =obs_data_post_arr[sampled_list,:]
        timestamp_ = timestamp[sampled_list]
        print('Phase estimating...')
        alpha_max_list = []
        for ipromp in ipromps_set:
            alpha_temp = ipromp.alpha_candidate(num_alpha_candidate)
            idx_max = ipromp.estimate_alpha(alpha_temp, obs_data_post_arr_, timestamp_)
            alpha_max_list.append(alpha_temp[idx_max]['candidate'])
            ipromp.set_alpha(alpha_temp[idx_max]['candidate'])
            print("alpha is %s" %alpha_temp[idx_max]['candidate'] )

        # task recognition
        print('Adding via points in each trained model...')
        for task_idx, ipromp in enumerate(ipromps_set):
            for idx in range(len(timestamp_)):
                ipromp.add_viapoint(timestamp_[idx] / alpha_max_list[task_idx], obs_data_post_arr_[idx, :])
            ipromp.param_update(unit_update=True)
        print('Computing the likelihood for each model under observations...')

        prob_task = []
        for ipromp in ipromps_set:
            prob_task_temp = ipromp.prob_obs()
            prob_task.append(prob_task_temp)
        idx_max_prob = np.argmax(prob_task)
        print("prob of tasks are %s" %prob_task)
        print('The max fit model index is task %s' % task_name[idx_max_prob])


        # phase_increase,phase_decrease = piecewise_fnc(obs_idx+1,obs_ratio[1])
        phase_list = obs_ratio_time_2
        phase_increase = sigmoid_increase(phase_list[obs_idx])
        phase_decrease = sigmoid_decrease(phase_list[obs_idx])
 
        phase_stack = np.column_stack((phase_decrease,phase_increase))

        ipromps_max = ipromps_set[idx_max_prob]
        Phi = ipromps_max.Phi
        robot_promps = [ipromps_max.promps[3],ipromps_max.promps[4],ipromps_max.promps[5]]
            
        for promp_idx,promp in enumerate(robot_promps):
                            
            sigma_merge_traj = []
            mean_merge_traj = []
            mean_traj = []
            sigma_traj = []
            mean_updated_traj = []
            sigma_updated_traj = []

            mean_updated = promp.meanW_nUpdated
            sigma_updated = promp.sigmaW_nUpdated

            for phase_idx, phase_ in enumerate (phase_stack):
                
                phase_de = phase_[0]
                phase_in = phase_[1]
                phi = Phi.T[phase_idx]

                if obs_ratio[0] == 0.1:
                    meanW0 = promp.meanW
                    sigmaW0 = promp.sigmaW
                    mean_point = np.dot(phi.T, meanW0)
                    mean_traj_dtw = np.dot(Phi.T, meanW0)
                    sigma_point = np.dot(phi.T, np.dot(sigmaW0, phi))
                    
                else:
                    mean_point = mean_merge_traj_full[promp_idx][phase_idx]
                    sigma_point = sigma_merge_traj_full[promp_idx][phase_idx]

                mean_point_updated = np.dot(phi.T, mean_updated)
                mean_traj_updated_dtw = np.dot(Phi.T, mean_updated)
                # plot_1d_traj(mean_traj_dtw,mean_traj_updated_dtw)
                sigma_point_updated = np.dot(phi.T, np.dot(sigma_updated, phi))

                sigma_point_activated = sigma_point/phase_de
                sigma_updated_activated = sigma_point_updated/phase_in

                sigma_divd_up = sigma_point_activated * sigma_updated_activated
                sigma_divdend = sigma_point_activated + sigma_updated_activated
                sigma_merge_point = sigma_divd_up / sigma_divdend

                mean_divd_up = mean_point*sigma_updated_activated + mean_point_updated*sigma_point_activated
                mean_merge_point =  mean_divd_up / sigma_divdend

                mean_traj.append(mean_point)
                sigma_traj.append(sigma_point)

                mean_updated_traj.append(mean_point_updated)
                sigma_updated_traj.append(sigma_point_updated)

                mean_merge_traj.append(mean_merge_point)
                sigma_merge_traj.append(sigma_merge_point)

            mean_traj = np.array(mean_traj)
            sigma_traj = np.array(sigma_traj)

            mean_updated_traj = np.array(mean_updated_traj)
            sigma_updated_traj = np.array(sigma_updated_traj)

            # dtw_score = dtw_fuc(mean_traj,mean_updated_traj)           

            mean_merge_traj = np.array(mean_merge_traj)
            sigma_merge_traj = np.array(sigma_merge_traj)
            dtw_score = dtw_fuc(mean_updated_traj,mean_merge_traj) 
            # print(dtw_score)
            # 
            # std_traj = 2*np.sqrt(sigma_traj).astype("float64")
            # std_updated_traj = 2*np.sqrt(sigma_updated_traj).astype("float64")
            # std_merge_traj = 2*np.sqrt(sigma_merge_traj).astype("float64")
            mean_merge_traj_full[promp_idx]= mean_merge_traj
            sigma_merge_traj_full[promp_idx]=sigma_merge_traj 



            if promp_idx == 0:

                task0_promp0_mean0.append(mean_traj)
                task0_promp0_mean1.append(mean_updated_traj)
                task0_promp0_mean2.append(mean_merge_traj)

            if promp_idx == 1:
                task0_promp1_mean0.append(mean_traj)
                task0_promp1_mean1.append(mean_updated_traj)
                task0_promp1_mean2.append(mean_merge_traj)

            if promp_idx == 2:
                task0_promp2_mean0.append(mean_traj)
                task0_promp2_mean1.append(mean_updated_traj)
                task0_promp2_mean2.append(mean_merge_traj) 

            

    task0 = np.array([[task0_promp0_mean0,task0_promp1_mean0,task0_promp2_mean0], [task0_promp0_mean1,task0_promp1_mean1,task0_promp2_mean1], [task0_promp0_mean2,task0_promp1_mean2,task0_promp2_mean2]])                     
 
    joblib.dump(task0,os.path.join(datasets_path, 'pkl/continous_obs_update.pkl'))
    joblib.dump(human_obs,os.path.join(datasets_path, 'pkl/continous_human_obs.pkl'))









if __name__ == '__main__':
    main()
