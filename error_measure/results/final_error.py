#!/usr/bin/python
# use removing the low variance feature to do the feature selection
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import ipdb


def main():
    t0 = [[0.0711, 0.0013], [0.0534, 0.00098], [0.0509, 0.00156], [0.03559, 0.000607], [0.032418, 0.000236]]
    t1 = [[0.1044, 0.0016], [0.0811, 0.00293], [0.0620, 0.00134], [0.05492, 0.000955], [0.052038, 0.000949]]
    t2 = [[0.1038, 0.0020], [0.0824, 0.00080], [0.08092, 0.00178], [0.08916, 0.002404], [0.103882, 0.001983]]
    t3 = [[0.1237, 0.0048], [0.0788, 0.0006], [0.069092, 0.000986], [0.057308, 0.000344], [0.046839, 0.0003111]]
    task_mean_var = [t0, t1, t2, t3]
    mean = np.array(data)[:,0]
    for task_id, data in enumerate(task_mean_var):
        plt.figure(task_id)
        plt.errorbar(np.linspace(0.1,0.5,5), np.array(data)[:,0], np.array(data)[:,1],
                     fmt='o', markersize=5, capsize=8, elinewidth=5,
                     markerfacecolor='none', markeredgewidth=1.5, markeredgecolor='w', color='black')
        plt.xlim([0, 0.55])
        plt.grid(True)
        plt.xlabel('Observation ratio')
        plt.ylabel('End-effector positioning error (m)')
    plt.show()

if __name__ == '__main__':
    main()
