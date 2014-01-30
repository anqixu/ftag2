set(FTAG2_LIB_NAME "ftag2" CACHE STRING "FTag v2 API library name")

cmake_minimum_required(VERSION 2.8.3)
project(ftag2)

set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS} -g -Wall -O2")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  dynamic_reconfigure
  image_transport
  std_msgs
  cv_bridge
  geometry_msgs
  sensor_msgs
  tf
)

find_package(OMPL REQUIRED)

find_package(Boost REQUIRED COMPONENTS date_time-mt system-mt thread-mt serialization-mt program_options-mt)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui calib3d)

#######################################
## Declare ROS messages and services ##
#######################################

add_message_files(
  DIRECTORY msg
  FILES FreqTBMarkerInfo.msg FreqTBPhaseStats.msg
)

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  geometry_msgs sensor_msgs std_msgs
)

generate_dynamic_reconfigure_options(
  cfg/CamTestbench.cfg
  cfg/FreqTestbench.cfg
  )


###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${FTAG2_LIB_NAME}
  CATKIN_DEPENDS roscpp rospy dynamic_reconfigure image_transport cv_bridge std_msgs geometry_msgs sensor_msgs tf
#  DEPENDS system_lib
)

###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)
link_directories(
  lib
  ${OpenCV_LIBRARY_DIRS}
)

FILE ( GLOB COMMON_HEADERS   "include/common/*.h*" )
FILE ( GLOB DETECTOR_HEADERS "include/detector/*.h*" )
FILE ( GLOB DECODER_HEADERS  "include/decoder/*.h*" )
FILE ( GLOB ENCODER_HEADERS  "include/encoder/*.h*" )
FILE ( GLOB TRACKER_HEADERS  "include/tracker/*.h*" )

FILE ( GLOB COMMON_SOURCES   "src/common/*.c*" )
FILE ( GLOB DETECTOR_SOURCES "src/detector/*.c*" )
FILE ( GLOB DECODER_SOURCES  "src/decoder/*.c*" )
FILE ( GLOB ENCODER_SOURCES  "src/encoder/*.c*" )
FILE ( GLOB TRACKER_SOURCES  "src/tracker/*.c*" )

add_library(${FTAG2_LIB_NAME}
  ${COMMON_SOURCES}
  ${DETECTOR_SOURCES}
  ${DECODER_SOURCES}
  ${ENCODER_SOURCES}
  ${TRACKER_SOURCES}
)

target_link_libraries( ${FTAG2_LIB_NAME}
  ${OMPL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
)

add_executable(cam_testbench src/cam_testbench.cpp)
target_link_libraries(cam_testbench
  ${FTAG2_LIB_NAME}
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
  yaml-cpp
)
add_dependencies(cam_testbench ${PROJECT_NAME}_gencfg)

add_executable(freq_testbench src/freq_testbench.cpp)
target_link_libraries(freq_testbench
  ${FTAG2_LIB_NAME}
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
)
add_dependencies(freq_testbench ${PROJECT_NAME}_gencfg ftag2_generate_messages_cpp)


#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
install(TARGETS ${FTAG2_LIB_NAME} cam_testbench
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_ftag2.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
