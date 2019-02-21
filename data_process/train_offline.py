#!/usr/bin/python
import load_data
import train_models
import noise_cov_cal

promp = True
ipromp = False
emg_ipromp = False
# the current file path
file_path = os.path.dirname(__file__)

# read models cfg file
cp_models = ConfigParser.SafeConfigParser()
cp_models.read(os.path.join(file_path, '../cfg/models.cfg'))
# read models params
datasets_path = os.path.join(file_path, cp_models.get('datasets', 'path'))
if promp:
    num_dim = cp_models.getint('promp_param', 'num_dim')
    num_obs_dim = cp_models.getint('promp_param', 'num_obs_dim')
elif ipromp:
    num_dim = cp_models.getint('ipromp_param', 'num_dim')
    num_obs_dim = cp_models.getint('ipromp_param', 'num_obs_dim')
elif emg_ipromp:
    num_dim = cp_models.getint('emg_ipromp_param', 'num_dim')
    num_obs_dim = cp_models.getint('ipromp_param', 'num_obs_dim')

len_norm = cp_models.getint('datasets', 'len_norm')
num_basis = cp_models.getint('basisFunc', 'num_basisFunc')
sigma_basis = cp_models.getfloat('basisFunc', 'sigma_basisFunc')
num_alpha_candidate = cp_models.getint('phase', 'num_phaseCandidate')

# the pkl data
datasets_pkl_path = os.path.join(datasets_path, 'pkl')
task_name_path = os.path.join(datasets_pkl_path, 'task_name_list.pkl')
datasets_norm_preproc_path = os.path.join(datasets_pkl_path, 'datasets_norm_preproc.pkl')
min_max_scaler_path = os.path.join(datasets_pkl_path, 'min_max_scaler.pkl')
noise_cov_path = os.path.join(datasets_pkl_path, 'noise_cov.pkl')

# datasets-related info
task_path_list = glob.glob(os.path.join(datasets_path, 'raw/*'))
task_path_list =sorted(task_path_list)
task_name_list = [task_path.split('/')[-1] for task_path in task_path_list]

def main():
    print("-----------")
    print('## Running the %s' % load_data.__name__)
    load_data.main()
    print("-----------")
    print('## Running the %s' % noise_cov_cal.__name__)
    noise_cov_cal.main()
    print("-----------")
    print('## Running the %s' % train_models.__name__)
    train_models.main()

if __name__ == '__main__':
    main()
