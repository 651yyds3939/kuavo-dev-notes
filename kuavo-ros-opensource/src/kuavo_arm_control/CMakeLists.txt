cmake_minimum_required(VERSION 3.0.2)
project(kuavo_arm_control)

# 编译成 C++ 11 或更高版本（Pinocchio 的要求）
add_compile_options(-std=c++11)

# 寻找 ROS 核心依赖
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  kuavo_msgs   # <--- 【新增这一行】告诉编译器去找官方的自定义消息
)

# 寻找我们刚刚下载好的 Pinocchio 动力学库
find_package(pinocchio REQUIRED)

catkin_package()

# 包含头文件路径
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PINOCCHIO_INCLUDE_DIRS}
)

# 声明我们要编译的目标可执行文件
add_executable(gravity_comp_node src/gravity_comp_node.cpp)

# 链接对应的库
target_link_libraries(gravity_comp_node
  ${catkin_LIBRARIES}
  pinocchio::pinocchio
)