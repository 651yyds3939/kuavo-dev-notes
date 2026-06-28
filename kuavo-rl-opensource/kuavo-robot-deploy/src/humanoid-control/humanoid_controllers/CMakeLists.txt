cmake_minimum_required(VERSION 3.10)
project(humanoid_controllers)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

## Find catkin macros and libraries
find_package(catkin REQUIRED
        COMPONENTS
        roscpp
        controller_interface
        humanoid_interface
        humanoid_wbc
        humanoid_estimation
        # ocs2_biped_robot_ros
        ocs2_core
        angles
        kuavo_msgs
        humanoid_interface_drake
        kuavo_solver
        
        )
find_package(drake REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(OpenVINO REQUIRED)
pkg_check_modules(pinocchio REQUIRED pinocchio)

find_package(kuavo_estimation)
if(kuavo_estimation_FOUND)
    add_definitions(-DKUAVO_CONTROL_LIB_FOUND)
    message(STATUS "Found kuavo_estimation")
else()
    message(WARNING "Could not find kuavo_estimation, disabling kuavo_control library, if you need it, just update submodules")
endif()


catkin_package(
        INCLUDE_DIRS
        include
        LIBRARIES
        ${PROJECT_NAME}
        CATKIN_DEPENDS
        roscpp
        controller_interface
        humanoid_interface
        # biped_s3
        biped_s40
        biped_s42
        # mujoco_sim
        humanoid_wbc
        humanoid_estimation
        # kuavo_estimation
        # ocs2_biped_robot_ros
        angles
)

###########
## Build ##
###########

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${pinocchio_INCLUDE_DIRS}
        ${kuavo_solver_INCLUDE_DIRS}
)
if (kuavo_estimation_FOUND)
message(STATUS "add kuavo_estimation include")
include_directories(${kuavo_estimation_INCLUDE_DIRS})
endif()

link_directories(
        ${pinocchio_LIBRARY_DIRS}
)

## Declare a cpp library
add_library(${PROJECT_NAME}
        src/humanoidController.cpp
        # src/ankle_solver.cpp
        )

target_link_libraries(${PROJECT_NAME}
        ${catkin_LIBRARIES}
        ${pinocchio_LIBRARIES}
        drake::drake 
        openvino::runtime
        ${kuavo_solver_LIBRARIES}
        )
if (kuavo_estimation_FOUND)
message(STATUS "add kuavo_estimation link")
target_link_libraries(${PROJECT_NAME}
        ${kuavo_estimation_LIBRARIES}
        )
endif()

target_compile_options(${PROJECT_NAME} PUBLIC ${OCS2_CXX_FLAGS})

# Cheat Controller Node
add_executable(cheat_controller_node
        src/CheatControllerNode.cpp
        )
add_dependencies(cheat_controller_node
        ${PROJECT_NAME}
        ${catkin_EXPORTED_TARGETS}
        )
target_link_libraries(cheat_controller_node
        ${PROJECT_NAME}
        ${catkin_LIBRARIES}
        )
target_compile_options(cheat_controller_node PRIVATE ${OCS2_CXX_FLAGS})

# Controller Nodelet

add_library(nodelet_controller
  src/controllerNodelet.cpp
)
set_target_properties(nodelet_controller PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/lib)

add_dependencies(nodelet_controller ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(nodelet_controller
        ${PROJECT_NAME}
        ${catkin_LIBRARIES}
)

install(TARGETS nodelet_controller
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)



# Normal Controller Node
add_executable(normal_controller_node
        src/NormalControllerNode.cpp
        )
add_dependencies(normal_controller_node
        ${PROJECT_NAME}
        ${catkin_EXPORTED_TARGETS}
        )
target_link_libraries(normal_controller_node
        ${PROJECT_NAME}
        ${catkin_LIBRARIES}
        )
target_compile_options(normal_controller_node PRIVATE ${OCS2_CXX_FLAGS})

#########################
###   CLANG TOOLING   ###
#########################
find_package(cmake_clang_tools QUIET)
if (cmake_clang_tools_FOUND)
    message(STATUS "Run clang tooling for target " ${PROJECT_NAME})
    add_clang_tooling(
            TARGETS ${PROJECT_NAME} humanoid_target_trajectories_publisher
            SOURCE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include
            CT_HEADER_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
            CF_WERROR
    )
endif (cmake_clang_tools_FOUND)

#############
## Install ##
#############

# Mark executables and/or libraries for installation
install(TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )

# Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
        FILES_MATCHING PATTERN "*.h"
        )

# Install plugins
install(FILES humanoid_controllers_plugins.xml
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
