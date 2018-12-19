#!/usr/bin/python
import rospy
from states_manager.msg import multiModal
import numpy as np
import threading
from scipy.ndimage.filters import gaussian_filter1d
import time
import os
import ConfigParser
from sklearn.externals import joblib
import sys
from geometry_msgs.msg import Pose
import moveit_commander
import util
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose
import baxter_interface
from baxter_interface import CHECK_VERSION
import ipdb
# read conf file
file_path = os.path.dirname(__file__)
cp = ConfigParser.SafeConfigParser()
cp.read(os.path.join(file_path, '../cfg/models.cfg'))

# load param
datasets_path = os.path.join(file_path, cp.get('datasets', 'path'))
num_alpha_candidate = cp.getint('phase', 'num_phaseCandidate')
timer_interval = cp.getfloat('online', 'timer_interval')
task_name_path = os.path.join(datasets_path, 'pkl/task_name_list.pkl')
task_name = joblib.load(task_name_path)
sigma = cp.getint('filter', 'sigma')
GROUP_NAME_ARM = 'left_arm'

# tape: [ 1.1169808  -0.12402304 0.23784241]
# screw: [ 1.09847439 -0.11998706 0.25383563]
# box: [ 1.10284653 -0.13208417 0.2812103 ]
#measure: [ 1.12147662 -0.08184267 0.26975943]
specify_position = np.array([1.12549807, -0.1559526 ,  0.2495953 ])

threshold = 0.01
# threshold = 0.12785109291


def move_robot(traj, traj_time):
    moveit_commander.roscpp_initialize(sys.argv)
    moveit_commander.RobotCommander()
    moveit_commander.PlanningSceneInterface()
    left = moveit_commander.MoveGroupCommander(GROUP_NAME_ARM)
    # marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size=1000)
    reference_frame = 'base'
    left.set_goal_position_tolerance(0.01)
    left.set_goal_orientation_tolerance(0.05)
    left.allow_replanning(True)
    left.set_planning_time(5)
    left.set_pose_reference_frame(reference_frame)
    # left.set_end_effector_link('left_hand')
    end_effector_link = left.get_end_effector_link()
    # print end_effector_link
    # # end_effector_link = 'left_hand'
    #
    # current_pose = left.get_current_pose(end_effector_link).pose
    # print current_pose
    global left_gripper
    left_gripper.open()
    list_of_poses = []
    for idx in range(len(traj)):
        traj_pose = Pose()
        traj_pose.position.x = traj[idx, 0]
        traj_pose.position.y = traj[idx, 1]
        traj_pose.position.z = traj[idx, 2]-0.218
        traj_pose.orientation.x = 0.000016822
        traj_pose.orientation.y = 0.9993395783
        traj_pose.orientation.z = - 0.0362892576
        traj_pose.orientation.w = 0.0018699794

        list_of_poses.append(traj_pose)
    max_tries = 50
    attempts = 0
    fraction = 0.0
    while fraction < 1.0 and attempts < max_tries:
        (plan, fraction) = left.compute_cartesian_path(
            list_of_poses,  # waypoints to follow
            0.01,  # eef_step
            0.0,  # jump_threshold
            True)  # avoid_collisions

        # Increment the number of attempts
        attempts += 1

        # Print out a progress message
        if attempts % 10 == 0:
            rospy.loginfo("Still trying after " + str(attempts) + " attempts...")

        # If we have a complete plan, execute the trajectory
        if fraction == 1.0:
            rospy.loginfo("Path computed successfully. Moving the arm.")

            left.execute(plan)

            rospy.loginfo("Path execution complete.")
        else:
            rospy.loginfo(
                "Path planning failed with only " + str(fraction) + " success after " + str(max_tries) + " attempts.")
    rospy.sleep(1)
    # global left_gripper
    # left_gripper.close()
    print 'closed the gripper'
    draw_line_rviz(traj)


def draw_line_rviz(traj):
    marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size=1000)
    rospy.sleep(0.5)
    for idx in range(traj.shape[0]):
        traj_pose = Pose()
        traj_pose.position.x = traj[idx, 0]
        traj_pose.position.y = traj[idx, 1]
        # traj_pose.position.z = traj[idx, 2]-0.15
        traj_pose.position.z = traj[idx, 2] - 0.21
        traj_pose.orientation.x = 0.000016822
        traj_pose.orientation.y = 0.9993395783
        traj_pose.orientation.z = - 0.0362892576
        traj_pose.orientation.w = 0.0018699794
        # visualize the pose in rviz using marker
        alpha = float(idx) / traj.shape[0] * 0.5 + 0.5
        rgba_tuple = ((0.5 * idx), 0.5, (0.5 * idx), alpha)
        util.send_traj_point_marker(marker_pub=marker_pub, pose=traj_pose, id=idx, rgba_tuple=rgba_tuple)


def fun_timer():
    """
    the timer callback function
    :return:
    """
    rospy.loginfo('Time out!!!')
    global flag_record
    flag_record = False  # stop record the msg
    global ipromps_set, obs_data_list
    rospy.loginfo('The len of observed data is %d', len(obs_data_list))
    global obs_data
    obs_data = np.array([]).reshape([0, ipromps_set[0].num_joints])
    timestamp = []
    for obs_data_list_idx in obs_data_list:
        emg = obs_data_list_idx['emg']
        left_hand = obs_data_list_idx['left_hand']
        # ipdb.set_trace()
        left_joints = obs_data_list_idx['left_joints']
        full_data = np.hstack([left_hand, emg, left_joints])
        # full_data = np.hstack([left_hand, left_joints])
        obs_data = np.vstack([obs_data, full_data])
        timestamp.append(obs_data_list_idx['stamp'])

    # filter the data
    obs_data = gaussian_filter1d(obs_data.T, sigma=sigma).T
    # preprocessing for the data
    obs_data_post_arr = ipromps_set[0].min_max_scaler.transform(obs_data)
    # consider the unobserved info
    obs_data_post_arr[:, 11:18] = 0.0
    # ipdb.set_trace()  

    # phase estimation
    rospy.loginfo('Phase estimating...')
    alpha_max_list = []
    for ipromp in ipromps_set:
        alpha_temp = ipromp.alpha_candidate()
        idx_max = ipromp.estimate_alpha(alpha_temp, obs_data_post_arr, timestamp)
        alpha_max_list.append(alpha_temp[idx_max]['candidate'])
        ipromp.set_alpha(alpha_temp[idx_max]['candidate'])

    # task recognition
    rospy.loginfo('Adding via points in each trained model...')
    for task_idx, ipromp in enumerate(ipromps_set):
        for idx in range(len(obs_data_list)):
            ipromp.add_viapoint(obs_data_list[idx]['stamp']/alpha_max_list[task_idx], obs_data_post_arr[idx, :])
        ipromp.param_update(unit_update=True)
    rospy.loginfo('Computing the likelihood for each model under observations...')

    prob_task = []
    for ipromp in ipromps_set:
        prob_task_temp = ipromp.prob_obs()
        prob_task.append(prob_task_temp)
    idx_max_prob = np.argmax(prob_task)
    # idx_max_prob = 3    # a trick for testing
    rospy.loginfo('The max fit model index is task %s', task_name[idx_max_prob])

    # robot motion generation
    [traj_time, traj] = ipromps_set[idx_max_prob].gen_real_traj(alpha_max_list[idx_max_prob])
    traj = ipromps_set[idx_max_prob].min_max_scaler.inverse_transform(traj)
    robot_traj = traj[:, 11:17]
    # robot_traj = traj[:, 3:6]
    robot_traj = filter_static_points(robot_traj)

    ######
    # move the robot
    # move_robot(robot_traj, traj_time)

    # save the conditional result
    rospy.loginfo('Saving the post IProMPs...')
    joblib.dump(ipromps_set, os.path.join(datasets_path, 'pkl/ipromps_set_post.pkl'))
    # save the robot traj
    rospy.loginfo('Saving the robot traj...')
    joblib.dump(robot_traj, os.path.join(datasets_path, 'pkl/robot_traj_online.pkl'))
    # save the obs
    rospy.loginfo('Saving the obs...')
    joblib.dump(obs_data, os.path.join(datasets_path, 'pkl/obs_data_online.pkl'))
    # finished
    rospy.loginfo('All finished!!!')

    rospy.loginfo('The max fit model index is task %s', task_name[idx_max_prob])


def callback(data):
    global flag_record
    # if not flag_record:
    #     return

    if not flag_record:
        # left_hand
        left_hand = np.array([data.tf_of_interest.transforms[5].transform.translation.x,
                              data.tf_of_interest.transforms[5].transform.translation.y,
                              data.tf_of_interest.transforms[5].transform.translation.z])
        # trigger
        print left_hand
        # print (np.linalg.norm(specify_position - left_hand))
        if np.linalg.norm(specify_position - left_hand) < threshold:
            flag_record = True
            print "I'm ready!!!"
            global timer
            # rospy.sleep(3)
            # print "go"
            # rospy.sleep(1)
            timer.start()   # start the timer
        else:
            return

    global init_time
    if init_time is None:
        init_time = data.header.stamp

    # # emg
    emg_data = np.array([data.emgStates.ch0, data.emgStates.ch1, data.emgStates.ch2,
                         data.emgStates.ch3, data.emgStates.ch4, data.emgStates.ch5,
                         data.emgStates.ch6, data.emgStates.ch7]).reshape([1, 8])

    # left_hand
    left_hand = np.array([data.tf_of_interest.transforms[5].transform.translation.x,
                          data.tf_of_interest.transforms[5].transform.translation.y,
                          data.tf_of_interest.transforms[5].transform.translation.z]).reshape([1, 3])

    # left_joints: left_hand actually
    left_joints = np.zeros_like(np.array([0, 0, 0,0, 0, 0, 0]).reshape([1, 7]))

    global obs_data_list
    time_stamp = (data.header.stamp - init_time).secs + (data.header.stamp - init_time).nsecs*1e-9
    obs_data_list.append({
                          'emg': emg_data,
                          'left_hand': left_hand,
                          'left_joints': left_joints,
                          'stamp': time_stamp})
    rospy.loginfo(obs_data_list[-1])


def filter_static_points(mat):
    last = mat[0] # for refercence
    new_mat = [last]
    for idx in range(mat.shape[0]):
        if np.linalg.norm(mat[idx]-last)<0.01:
            pass
        else:
            new_mat.append(mat[idx])
            last = mat[idx]
    return np.array(new_mat)


def main():
    # init node
    rospy.init_node('online_ipromps_node', anonymous=True)
    rospy.loginfo('Created the ROS node!')

    # load datasets
    rospy.loginfo('Loading the datasets...')
    global ipromps_set
    ipromps_set = joblib.load(os.path.join(datasets_path, 'pkl/ipromps_set.pkl'))

    # gripper
    # rs = baxter_interface.RobotEnable(CHECK_VERSION)
    # init_state = rs.state().enabled
    # global left_gripper
    # left_gripper = baxter_interface.Gripper('left', CHECK_VERSION)

    # the flag var of starting info record
    global flag_record, obs_data_list
    flag_record = False
    # to save the online data
    obs_data_list = []
    # create a timer
    global timer
    timer = threading.Timer(timer_interval, fun_timer)

    # the init time
    global init_time
    init_time = None

    # subscribe the /multiModal_states topic
    rospy.Subscriber("/multiModal_states", multiModal, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    main()
