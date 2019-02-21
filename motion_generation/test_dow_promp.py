#!/usr/bin/python
import numpy as np
from ipromps import ipromps_lib
from sklearn.externals import joblib
import os
import ConfigParser
import matplotlib.pyplot as plt
from numpy.linalg import inv
import ipdb
from scipy.interpolate import griddata
# the current file path
file_path = os.path.dirname(__file__)

# read models cfg file
cp_models = ConfigParser.SafeConfigParser()
cp_models.read(os.path.join(file_path, '../cfg/models.cfg'))
# read models params
datasets_path = os.path.join(file_path, cp_models.get('datasets', 'path'))
len_norm = cp_models.getint('datasets', 'len_norm')
num_basis = cp_models.getint('basisFunc', 'num_basisFunc')
sigma_basis = cp_models.getfloat('basisFunc', 'sigma_basisFunc')
datasets_norm_preproc = joblib.load(os.path.join(datasets_path, 'pkl/datasets_norm_preproc.pkl'))

num_demo = 15
test_idx = 10

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


fig = plt.figure(0)
y_increase, y_decrease = sigmoid(0,1,101)
plt.plot(y_increase)
plt.plot(y_decrease)
ipdb.set_trace()
plt.show()
promp = ipromps_lib.ProMP()
for idx_demo in range(num_demo):
    promp.add_demonstration(datasets_norm_preproc[0][idx_demo]['left_joints'][:,1])
# promp.plot_prior(b_regression=False, linewidth_mean=5, b_dataset=False)

promp.add_viapoint(0.1, datasets_norm_preproc[0][test_idx]['left_joints'][10,1])
# promp.add_viapoint(0.2, datasets_norm_preproc[0][test_idx]['left_joints'][20,0])
promp.add_viapoint(0.3, datasets_norm_preproc[0][test_idx]['left_joints'][30,1])
# promp.add_viapoint(0.4, datasets_norm_preproc[0][test_idx]['left_joints'][40,0])
promp.add_viapoint(0.5, datasets_norm_preproc[0][test_idx]['left_joints'][50,1])
# promp.add_viapoint(1.0, datasets_norm_preproc[0][test_idx]['left_joints'][100,0])
promp.param_updata()

meanW0 = promp.meanW
meanW1 = promp.meanW_uUpdated
sigmaW0 = promp.sigmaW
sigmaW1 = promp.sigmaW_uUpdated

sigmaW2 = inv(inv(sigmaW0)+ inv(sigmaW1))
meanW2 = np.dot(np.dot(sigmaW2, inv(sigmaW0)),meanW0) + np.dot(np.dot(sigmaW2, inv(sigmaW1)),meanW1)
meanw2 = np.dot(np.dot(sigmaW1, inv(sigmaW0+sigmaW1)),meanW0) + np.dot(np.dot(sigmaW0, inv(sigmaW0+sigmaW1)),meanW1)


