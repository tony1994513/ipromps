#!/usr/bin/python
import rospy
from states_manager.msg import multiModal
import numpy as np

specify_pose = np.array([1.1132629934, -0.2387471985, 0.271010064])
threshold = 0.13856406460551018


def callback(data):
    # left_hand
    left_hand = np.array([data.tf_of_interest.transforms[5].transform.translation.x,
                          data.tf_of_interest.transforms[5].transform.translation.y,
                          data.tf_of_interest.transforms[5].transform.translation.z])

    # print it when human reach some specify pose
    if np.linalg.norm(specify_pose - left_hand) < threshold:
        print "I'm ready!!!"
    else:
        print "Not ready!"


def main():
    # init node
    rospy.init_node('action_trigger_node', anonymous=True)
    rospy.loginfo('Created the ROS node!')

    # suscribe the interes topic
    rospy.Subscriber("/multiModal_states", multiModal, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()



if __name__ == '__main__':
    main()