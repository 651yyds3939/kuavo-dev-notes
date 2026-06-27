cmake_minimum_required(VERSION 3.0.2)
project(kuavo_visual_follow)
find_package(catkin REQUIRED COMPONENTS rospy std_msgs vision_msgs kuavo_msgs)
catkin_package()
catkin_install_python(PROGRAMS
  scripts/head_track.py
  scripts/trim_test.py
  scripts/body_track.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
