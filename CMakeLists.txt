cmake_minimum_required(VERSION 2.8.3)
project(ipromps)


find_package(catkin REQUIRED COMPONENTS
  rospy
)

catkin_python_setup()

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES ipromps
 CATKIN_DEPENDS rospy
#  DEPENDS system_lib
)


include_directories(
  ${catkin_INCLUDE_DIRS}
)

