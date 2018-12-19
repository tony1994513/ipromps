#!/usr/bin/python
import loocv_ee_error
from sklearn.externals import joblib
import numpy as np
import os
import ConfigParser
import matplotlib.pyplot as plt


# read conf file
file_path = os.path.dirname(__file__)
cp_models = ConfigParser.SafeConfigParser()
cp_models.read(os.path.join(file_path, '../cfg/models.cfg'))
# the datasets path
datasets_path = os.path.join(file_path, cp_models.get('datasets', 'path'))
# load datasets
task_name = joblib.load(os.path.join(datasets_path, 'pkl/task_name_list.pkl'))


def ee_error_n_recog_accuracy():
    num_alpha_candidate = 10
    obs_ratio_list = [0.3,0.9]
    result_full_full = []
    for obs_ratio in obs_ratio_list:
        print '---------------------------'
        result_full = []
        for task_id, name in enumerate(task_name):
            print 'testing the '+ name + ' th obs_ratio %f' % obs_ratio
            [error_full, acc, phase_error_full] = loocv_ee_error.main(obs_ratio, task_id, num_alpha_candidate)
            result_full.append([error_full, acc, phase_error_full])
        result_full_full.append(result_full)

        for id, name in enumerate(task_name):
            print 'the accuracy for ' + name + ' is %f' % result_full[id][1]
        for id, name in enumerate(task_name):
            print '---------------------------'
            arr = np.array(result_full[id][0])
            var = np.var(arr)
            mean = np.mean(arr)
            print 'the mean for ' + name + ' is %f' % mean
            print 'the variance for ' + name + ' is %f' % var
            print '---------------------------'
    joblib.dump(result_full_full, os.path.join(datasets_path, 'pkl/ee_error_n_recog_accuracy_with_emg.pkl'))


def phase_error():
    num_alpha_candidate_list = list(np.linspace(1,10,10))
    obs_ratio = 0.3

    print '---------------------------'
    result_full_full = []
    for num_alpha_candidate in num_alpha_candidate_list:
        result_full = []
        for id, name in enumerate(task_name):
            print 'testing the '+ name + ' with num phase: %f' % num_alpha_candidate
            [error_full, acc, phase_error_full] = loocv_ee_error.main(obs_ratio, id, int(num_alpha_candidate))
            result_full.append([error_full, acc, phase_error_full])
        result_full_full.append(result_full)

    joblib.dump(result_full_full, os.path.join(datasets_path, 'pkl/phase_estimate_full_full_with_emg.pkl'))

    for task_id, name in enumerate(task_name):
        plt.figure(task_id)
        for phase_id, data in enumerate(result_full_full):
            phase_data = result_full_full[phase_id][task_id][2]
            mean_phase_data = np.mean(phase_data)
            std_phase_data = np.std(phase_data)
            plt.errorbar(phase_id+1, mean_phase_data, std_phase_data, fmt='o', markersize=5, capsize=8,
                         elinewidth=5, markerfacecolor='none', markeredgewidth=1.5, markeredgecolor='w')
            plt.xlim([0, 11])
            plt.grid(True)
            plt.xlabel('Observation ratio')
            plt.ylabel('phase ratio error (s)')
    plt.show()


def draw_phase_error():
    result_full_full = joblib.load(os.path.join(datasets_path, 'pkl/ee_error_n_recog_accuracy_with_emg.pkl'))
    for task_id, name in enumerate(task_name):
        plt.figure(task_id)
        for phase_id, data in enumerate(result_full_full):
            phase_data = result_full_full[phase_id][task_id][2]
            mean_phase_data = np.mean(phase_data)
            std_phase_data = np.std(phase_data)
            plt.errorbar(phase_id+1, mean_phase_data, std_phase_data, fmt='o', markersize=5, capsize=8,
                         elinewidth=5, markerfacecolor='none', markeredgewidth=1.5, markeredgecolor='w', color='black')
            plt.xlim([0, 11])
            plt.grid(True)
            plt.xlabel('Phase factor candidate number')
            plt.ylabel('Phase factor error (s)')
    plt.show()


def draw_all_phase_error():
    result_full_full = joblib.load(os.path.join(datasets_path, 'pkl/phase_estimate_full_full.pkl'))
    plt.figure(0)
    for phase_id, data in enumerate(result_full_full):
        phase_data_full = []
        for task_id, name in enumerate(task_name):
            phase_data = result_full_full[phase_id][task_id][2]
            phase_data_full.append(phase_data)
        mean_phase_data = np.mean(phase_data_full)
        std_phase_data = np.std(phase_data_full)
        plt.errorbar(phase_id+1, mean_phase_data, std_phase_data, fmt='o', markersize=8, capsize=6,
                     elinewidth=2, markerfacecolor='none', markeredgewidth=2, markeredgecolor='black', color='black')
        plt.xlim([0, 11])
        plt.grid(True)
        plt.xlabel('Phase factor candidate number')
        plt.ylabel('Phase factor error (s)')
    plt.show()


def draw_positioning_error():
    obs_ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    result_full_full = joblib.load(os.path.join(datasets_path, 'pkl/ee_error_n_recog_accuracy_with_emg.pkl'))
    for task_id, name in enumerate(task_name):
        plt.figure(task_id)
        for phase_id, data in enumerate(result_full_full):
            ee_error = result_full_full[phase_id][task_id][0]
            mean_phase_data = np.mean(ee_error)
            std_phase_data = np.std(ee_error)
            plt.errorbar(obs_ratio_list[phase_id], mean_phase_data, std_phase_data, fmt='o', markersize=5, capsize=8,
                         elinewidth=5, markerfacecolor='none', markeredgewidth=1.5, markeredgecolor='w', color='black')
            plt.xlim([0, 0.55])
            plt.grid(True)
            plt.xlabel('Observation ratio')
            plt.ylabel('Positioning error(m)')
    plt.show()


def draw_all_positioning_error():
    obs_ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    result_full_full = joblib.load(os.path.join(datasets_path, 'pkl/ee_error_n_recog_accuracy_with_emg.pkl'))
    plt.figure(0)
    for phase_id, data in enumerate(result_full_full):
        ee_error_full = []
        for task_id, name in enumerate(task_name):
            ee_error = result_full_full[phase_id][task_id][0]
            ee_error_full.append(ee_error)
        mean_phase_data = np.mean(ee_error_full)
        std_phase_data = np.std(ee_error_full)
        plt.errorbar(obs_ratio_list[phase_id], mean_phase_data, std_phase_data, fmt='o', markersize=8, capsize=6,
                     elinewidth=2, markerfacecolor='none', markeredgewidth=2, markeredgecolor='black', color='black')
        plt.xlim([0, 0.55])
        plt.grid(True)
        plt.xlabel('Observation ratio')
        plt.ylabel('Positioning error (m)')
    plt.show()


def main():
    ee_error_n_recog_accuracy()
    # phase_error()
    # draw_phase_error()
    # draw_positioning_error()
    # draw_all_positioning_error()
    # draw_all_phase_error()

if __name__ == '__main__':
    main()
