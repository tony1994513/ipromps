import os,sys
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
import ipdb
import glob
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import griddata
from sklearn.externals import joblib

dir_of_this_script = os.path.dirname(os.path.realpath(__file__))
demonstration_dir = os.path.join(dir_of_this_script, '..', 'datasets', 'pick_20190228')
demo_path_list = glob.glob(os.path.join(demonstration_dir,'*.npy'))
demo_path_list = sorted(demo_path_list)
sigma = 5

len_norm=101
datasets_raw = []
datasets_filtered = []
datasets_norm = []
fig = plt.figure(0)
ax = fig.gca(projection='3d')
for demo_path in demo_path_list:
    raw_demo = np.load(demo_path, 'r')
    filtered_demo = gaussian_filter1d(raw_demo.T, sigma=sigma).T
    grid = np.linspace(0, 2, len_norm)
    time_stamp = np.linspace(0, 2, len(raw_demo))
    norm_demo = griddata(time_stamp, filtered_demo, grid, method='linear')
    datasets_raw.append(raw_demo)
    datasets_filtered.append(datasets_raw)
    datasets_norm.append(datasets_norm)
#     ax.plot(norm_demo[:,0],norm_demo[:,1],norm_demo[:,2])
# plt.show()
pkl_dir = os.path.join(demonstration_dir,"pkl")
print('Saving the datasets as pkl ...')
joblib.dump(datasets_raw, os.path.join(pkl_dir, 'datasets_raw.pkl'))
joblib.dump(datasets_filtered, os.path.join(pkl_dir, 'datasets_filtered.pkl'))
joblib.dump(datasets_norm, os.path.join(pkl_dir, 'datasets_norm.pkl'))
print('Loaded, filtered, normalized, preprocessed and saved the datasets successfully!!!')


    


