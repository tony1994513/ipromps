#!/usr/bin/python
# Filename: imu_pose_test.py

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import ipromps_lib
import scipy.linalg
from sklearn.externals import joblib

# init
plt.close('all')    # close all windows
len_normal = 101    # the len of normalized traj, don't change it

# model parameter
num_demos = 10         # number of trajectoreis for training
obs_ratio = 10
num_basis = 31
# measurement noise
imu_noise = 1.0
emg_noise = 2.0
pose_noise = 1.0
num_alpha_candidate = 10
states_refresh_rate = 50.0
nominal_duration = 1.0
# scaling factor for data
sf_imu = 1000.0
sf_emg = 100.0
sf_pose = 0.1

# plot options
b_plot_raw_dateset = False
b_plot_prior_distribution = True
b_plot_update_distribution = True
b_plot_phase_distribution = False



#################################
# load raw date sets
#################################
dataset_aluminum_hold = joblib.load('./pkl/dataset_aluminum_hold.pkl')
dataset_spanner_handover = joblib.load('./pkl/dataset_spanner_handover.pkl')
dataset_tape_hold = joblib.load('./pkl/dataset_tape_hold.pkl')

#################################
# load norm date sets
#################################
dataset_aluminum_hold_norm = joblib.load('./pkl/dataset_aluminum_hold_norm.pkl')
dataset_spanner_handover_norm = joblib.load('./pkl/dataset_spanner_handover_norm.pkl')
dataset_tape_hold_norm = joblib.load('./pkl/dataset_tape_hold_norm.pkl')


#################################
# Interaction ProMPs train
#################################
# create a 3 tasks iProMP
ipromp_aluminum_hold = ipromps_lib.NDProMP(num_joints=11, num_basis=num_basis, sigma_basis=0.05, num_samples=101)
ipromp_spanner_handover = ipromps_lib.NDProMP(num_joints=11, num_basis=num_basis, sigma_basis=0.05, num_samples=101)
ipromp_tape_hold = ipromps_lib.NDProMP(num_joints=11, num_basis=num_basis, sigma_basis=0.05, num_samples=101)

# add demostration
for idx in range(0, num_demos):
    # add demonstration of aluminum_hold
    demo_temp = np.hstack([dataset_aluminum_hold_norm[idx]["imu"]/sf_imu, dataset_aluminum_hold_norm[idx]["pose"]/sf_pose])
    ipromp_aluminum_hold.add_demonstration(demo_temp)
    # add demonstration of spanner_handover
    demo_temp = np.hstack([dataset_spanner_handover_norm[idx]["imu"]/sf_imu, dataset_spanner_handover_norm[idx]["pose"]/sf_pose])
    ipromp_spanner_handover.add_demonstration(demo_temp)
    # add demonstration of tape_hold
    demo_temp = np.hstack([dataset_tape_hold_norm[idx]["imu"]/sf_imu, dataset_tape_hold_norm[idx]["pose"]/sf_pose])
    ipromp_tape_hold.add_demonstration(demo_temp)


################################
# Interaction ProMPs test
################################
# select the testset
# selected_testset = dataset_aluminum_hold_norm[np.int(sys.argv[1])]
# selected_testset = dataset_spanner_handover_norm[np.int(sys.argv[1])]
selected_testset = dataset_tape_hold_norm[1]
# construct the test set
test_set = np.hstack((selected_testset["imu"]/sf_imu, selected_testset["pose"]/sf_pose))
robot_response = selected_testset["pose"]/sf_pose

# add via point as observation
imu_meansurement_noise_cov = np.eye((4)) * imu_noise
pose_meansurement_noise_cov = np.eye((7)) * pose_noise
meansurement_noise_cov_full = scipy.linalg.block_diag(imu_meansurement_noise_cov, pose_meansurement_noise_cov)

# add via points to update the distribution
for idx in range(obs_ratio):
    ipromp_aluminum_hold.add_viapoint(0.01*idx, test_set[idx, :], meansurement_noise_cov_full)
    ipromp_spanner_handover.add_viapoint(0.01*idx, test_set[idx,:], meansurement_noise_cov_full)
    ipromp_tape_hold.add_viapoint(0.01*idx, test_set[idx,:], meansurement_noise_cov_full)


#################################
# plot raw data
#################################
if b_plot_raw_dateset == True:
    ## plot the aluminum hold task raw data
    plt.figure(0)
    for ch_ex in range(4):
       plt.subplot(411+ch_ex)
       for idx in range(num_demos):
           plt.plot(range(len(dataset_aluminum_hold[idx]["imu"][:, ch_ex])), dataset_aluminum_hold[idx]["imu"][:, ch_ex]); plt.axis('off')
    plt.figure(2)
    for ch_ex in range(7):
       plt.subplot(711+ch_ex)
       for idx in range(num_demos):
           plt.plot(range(len(dataset_aluminum_hold[idx]["pose"][:, ch_ex])), dataset_aluminum_hold[idx]["pose"][:, ch_ex]); plt.axis('off')
    ## plot the spanner handover task raw data
    plt.figure(10)
    for ch_ex in range(4):
       plt.subplot(411 + ch_ex)
       for idx in range(num_demos):
           plt.plot(range(len(dataset_spanner_handover[idx]["imu"][:, ch_ex])), dataset_spanner_handover[idx]["imu"][:, ch_ex]); plt.axis('off')
    plt.figure(12)
    for ch_ex in range(7):
       plt.subplot(711 + ch_ex)
       for idx in range(num_demos):
           plt.plot(range(len(dataset_spanner_handover[idx]["pose"][:, ch_ex])), dataset_spanner_handover[idx]["pose"][:, ch_ex]); plt.axis('off')
    ## plot the tape hold task raw data
    plt.figure(20)
    for ch_ex in range(4):
       plt.subplot(411 + ch_ex)
       for idx in range(num_demos):
           plt.plot(range(len(dataset_tape_hold[idx]["imu"][:, ch_ex])), dataset_tape_hold[idx]["imu"][:, ch_ex]); plt.axis('off')
    plt.figure(22)
    for ch_ex in range(7):
       plt.subplot(711 + ch_ex)
       for idx in range(num_demos):
           plt.plot(range(len(dataset_tape_hold[idx]["pose"][:, ch_ex])), dataset_tape_hold[idx]["pose"][:, ch_ex]); plt.axis('off')


#################################
# plot the prior distributioin
#################################
if b_plot_prior_distribution == True:
    # plot ipromp_aluminum_hold
    plt.figure(50)
    for i in range(4):
        plt.subplot(411+i)
        ipromp_aluminum_hold.promps[i].plot_prior(color='b', legend='alumnium hold model, imu'); plt.axis('off')
    plt.figure(52)
    for i in range(7):
        plt.subplot(711+i)
        ipromp_aluminum_hold.promps[4+i].plot_prior(color='b', legend='alumnium hold model, pose'); plt.axis('off')
    # plot ipromp_spanner_handover
    plt.figure(60)
    for i in range(4):
        plt.subplot(411+i)
        ipromp_spanner_handover.promps[i].plot_prior(color='b', legend='spanner handover model, imu'); plt.axis('off')
    plt.figure(62)
    for i in range(7):
        plt.subplot(711+i)
        ipromp_spanner_handover.promps[4+i].plot_prior(color='b', legend='spanner handover model, pose'); plt.axis('off')
    # plot ipromp_tape_hold
    plt.figure(70)
    for i in range(4):
        plt.subplot(411+i)
        ipromp_tape_hold.promps[i].plot_prior(color='b', legend='tape hold model, imu'); plt.axis('off')
    plt.figure(72)
    for i in range(7):
        plt.subplot(711+i)
        ipromp_tape_hold.promps[4+i].plot_prior(color='b', legend='tape hold model, pose'); plt.axis('off')


#################################
# plot the updated distributioin
#################################
if b_plot_update_distribution == True:
    # plot ipromp_aluminum_hold
    plt.figure(50)
    for i in range(4):
        plt.subplot(411+i)
        plt.plot(ipromp_aluminum_hold.x, test_set[:, i], color='r', linewidth=3, label='ground truth'); plt.legend();
        ipromp_aluminum_hold.promps[i].plot_updated(color='r', legend='updated distribution', via_show=True); plt.legend();
    plt.figure(52)
    for i in range(7):
        plt.subplot(711+i)
        plt.plot(ipromp_aluminum_hold.x, robot_response[:, i], color='r', linewidth=3, label='ground truth'); plt.legend();
        ipromp_aluminum_hold.promps[4+i].plot_updated(color='r', legend='updated distribution', via_show=False); plt.legend();
    # plot ipromp_spanner_handover
    plt.figure(60)
    for i in range(4):
        plt.subplot(411+i)
        plt.plot(ipromp_aluminum_hold.x, test_set[:, i], color='r', linewidth=3, label='ground truth'); plt.legend()
        ipromp_spanner_handover.promps[i].plot_updated(color='r', legend='updated distribution', via_show=True); plt.legend();
    plt.figure(62)
    for i in range(7):
        plt.subplot(711+i)
        plt.plot(ipromp_aluminum_hold.x, robot_response[:, i], color='r', linewidth=3, label='ground truth'); plt.legend()
        ipromp_spanner_handover.promps[4+i].plot_updated(color='r', legend='updated distribution', via_show=False); plt.legend();
    # plot ipromp_tape_hold
    plt.figure(70)
    for i in range(4):
        plt.subplot(411+i)
        plt.plot(ipromp_aluminum_hold.x, test_set[:, i], color='r', linewidth=3, label='ground truth'); plt.legend()
        ipromp_tape_hold.promps[i].plot_updated(color='r', legend='updated distribution', via_show=True); plt.legend();
    plt.figure(72)
    for i in range(7):
        plt.subplot(711+i)
        plt.plot(ipromp_aluminum_hold.x, robot_response[:, i], color='r', linewidth=3, label='ground truth'); plt.legend()
        ipromp_tape_hold.promps[4+i].plot_updated(color='r', legend='updated distribution', via_show=False); plt.legend();

plt.show()