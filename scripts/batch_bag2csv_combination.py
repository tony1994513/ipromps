#!/usr/bin/env python
import glob
import os
import subprocess
import ConfigParser

# read conf file
file_path = os.path.dirname(__file__)
cp = ConfigParser.SafeConfigParser()
cp.read(os.path.join(file_path, '../cfg/models.cfg'))
# load the raw data set
datasets_raw_path = os.path.join(file_path, cp.get('datasets', 'path'), 'raw')


def main():
    # run sh script for each rosbag file in datasets path
    task_path_list = glob.glob(os.path.join(datasets_raw_path, "*"))
    task_path_list = sorted(task_path_list)
    for task_path in task_path_list:
        tmp = glob.glob(os.path.join(task_path, "*"))
        for tmp_path in tmp: 
            demo_path_list = glob.glob(os.path.join(tmp_path, "*.bag"))
            for demo_path in demo_path_list:
                subprocess.Popen([os.path.join(file_path, 'bag_to_csv.sh') + ' ' +
                              demo_path + ' ' +
                              os.path.join(tmp_path, 'csv')], shell=True)  # non block function
                              # os.path.join(task_path, 'csv')], shell = True).wait()   # the block function

if __name__ == '__main__':
    main()
