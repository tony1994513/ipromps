import ipdb
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import griddata

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


obs_ratio_time_1 = np.array([0.0,0.3,0.6])
obs_ratio_time_2 = np.array([0.3,0.6,0.9])
obs_ratio_time = np.column_stack((obs_ratio_time_1,obs_ratio_time_2))
 

for obs_idx,obs_ratio in enumerate(obs_ratio_time):
    ipdb.set_trace()
    num_of_point = (obs_ratio_time_2[obs_idx] - obs_ratio_time_1[obs_idx])*100
    phase_increase ,phase_decrease = sigmoid(obs_ratio_time_1[obs_idx],obs_ratio_time_2[obs_idx],num_of_point)