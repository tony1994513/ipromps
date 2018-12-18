#!/usr/bin/python
import numpy as np
import ipromps_lib
from sklearn.externals import joblib
import os
import ConfigParser
import matplotlib.pyplot as plt

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

fig = plt.figure(0)

promp = ipromps_lib.ProMP()
for idx_demo in range(num_demo):
    promp.add_demonstration(datasets_norm_preproc[0][idx_demo]['left_joints'][:,1])
promp.plot_prior(b_regression=False, linewidth_mean=5, b_dataset=False)

promp.add_viapoint(0.1, datasets_norm_preproc[0][test_idx]['left_joints'][10,0])
# promp.add_viapoint(0.2, datasets_norm_preproc[0][test_idx]['left_joints'][20,0])
promp.add_viapoint(0.3, datasets_norm_preproc[0][test_idx]['left_joints'][30,0])
# promp.add_viapoint(0.4, datasets_norm_preproc[0][test_idx]['left_joints'][40,0])
promp.add_viapoint(0.5, datasets_norm_preproc[0][test_idx]['left_joints'][50,0])
# promp.add_viapoint(1.0, datasets_norm_preproc[0][test_idx]['left_joints'][100,0])
promp.param_updata()
# promp.plot_uUpdated(legend='Inferred trajectory')
# promp.plot_uViapoints()

# plt.plot(promp.x, datasets_norm_preproc[0][test_idx]['left_joints'][:,0], color='g', linewidth=5, label='ground truth')

for demo_idx in range(num_demo):
    data = datasets_norm_preproc[0][demo_idx]['left_joints'][:,1]
    plt.plot(np.array(range(len(data)))/100.0, data, color='grey', linewidth=2)

plt.xlabel('t(s)')
plt.ylabel('y(m)')

plt.legend(loc=2)
plt.show()

