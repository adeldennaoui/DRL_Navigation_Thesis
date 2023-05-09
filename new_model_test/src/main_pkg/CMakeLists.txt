cmake_minimum_required(VERSION 2.8.3)
project(main_pkg)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  roscpp
  rospy
  std_msgs
  std_srvs
  xacro
  move_base_msgs
  actionlib
  gazebo_msgs
  xacro
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)


