#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt


def accuracy(num):
    return num/17.0


def main():

    task_marker = ['o', '*', '^', 'd']
    task_label = ['gummed_paper', 'screw_driver', 'pencil_box', 'measurement_tape']

    t0 = [accuracy(15), accuracy(15), accuracy(16), accuracy(16), accuracy(17)]
    t1 = [accuracy(17), accuracy(17), accuracy(17), accuracy(17), accuracy(17)]
    t2 = [accuracy(16), accuracy(16), accuracy(17), accuracy(17), accuracy(17)]
    t3 = [accuracy(15), accuracy(15), accuracy(17), accuracy(17), accuracy(17)]
    task = [t0, t1, t2, t3]

    ########################################with EMG
    plt.figure(0)
    for task_id, task_data in enumerate(task):
        plt.plot(np.linspace(0.1,0.5,5), task_data, task_marker[task_id], markersize=15, alpha=0.7, label=task_label[task_id])
        plt.ylim([0,1.1])
        plt.xlim([0, 0.55])
        plt.xlabel('Observation ratio')
        plt.ylabel('Task recognition accuracy')
        plt.grid(True)
        plt.legend(loc=4)

    ########################################without EMG
    t0 = [accuracy(11), accuracy(13), accuracy(15), accuracy(15), accuracy(15)]
    t1 = [accuracy(8), accuracy(8), accuracy(11), accuracy(11), accuracy(11)]
    t2 = [accuracy(12), accuracy(12), accuracy(12), accuracy(11), accuracy(10)]
    t3 = [accuracy(13), accuracy(15), accuracy(12), accuracy(10), accuracy(9)]
    task = [t0, t1, t2, t3]

    plt.figure(1)
    for task_id, task_data in enumerate(task):
        plt.plot(np.linspace(0.1, 0.5, 5), task_data, task_marker[task_id], markersize=15, alpha=0.7,
                 label=task_label[task_id])
        plt.ylim([0, 1.1])
        plt.xlim([0, 0.55])
        plt.xlabel('Observation ratio')
        plt.ylabel('Task recognition accuracy')
        plt.grid(True)
        plt.legend(loc=4)
    plt.show()


if __name__ == '__main__':
    main()
