set(FTAG2_LIB_NAME "ftag2" CACHE STRING "FTag v2 API library name")
option(USE_OPENMP "Enable parallelism with OpenMP" ON) 
set(FTAG2_READER_NODELET_NAME "ftag2_reader_nodelet")
set(FTAG2_TRACKER_NODELET_NAME "ftag2_tracker_nodelet")

cmake_minimum_required(VERSION 2.8.3)
project(ftag2)

set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS} -g -Wall -O2")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  message_generation
  nodelet
  dynamic_reconfigure
  image_transport
  std_msgs
  cv_bridge
  geometry_msgs
  sensor_msgs
  tf
  image_geometry
  image_transport
)

find_package(Boost REQUIRED COMPONENTS date_time system thread serialization program_options)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui calib3d)

if(USE_OPENMP)
  FIND_PACKAGE(OpenMP REQUIRED)
  if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  else()
    message("Could not find OpenMP libraries")
  endif()
endif()

#######################################
## Declare ROS messages and services ##
#######################################

add_message_files(
  DIRECTORY msg
  FILES FreqTBMarkerInfo.msg FreqTBPhaseStats.msg TagDetection.msg TagDetections.msg ARMarkerFT.msg ARMarkersFT.msg
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
  cfg/FTag2Reader.cfg
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
  LIBRARIES ${FTAG2_LIB_NAME} ${FTAG2_READER_NODELET_NAME} ${FTAG2_TRACKER_NODELET_NAME}
  CATKIN_DEPENDS roscpp rospy message_runtime nodelet dynamic_reconfigure image_transport cv_bridge std_msgs geometry_msgs sensor_msgs tf
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
FILE ( GLOB KALMAN_HEADERS	 "include/kalman/*.h*" )

FILE ( GLOB COMMON_SOURCES   "src/common/*.c*" )
FILE ( GLOB DETECTOR_SOURCES "src/detector/*.c*" )
FILE ( GLOB DECODER_SOURCES  "src/decoder/*.c*" )
FILE ( GLOB ENCODER_SOURCES  "src/encoder/*.c*" )
FILE ( GLOB TRACKER_SOURCES  "src/tracker/*.c*" )
FILE ( GLOB KALMAN_SOURCES	 "src/kalman/*.c*" )

add_library(${FTAG2_LIB_NAME}
  ${COMMON_SOURCES}
  ${DETECTOR_SOURCES}
  ${DECODER_SOURCES}
  ${ENCODER_SOURCES}
  ${TRACKER_SOURCES}
  ${KALMAN_SOURCES}
)
target_link_libraries( ${FTAG2_LIB_NAME}
  ${OpenCV_LIBRARIES}
)

add_library(${FTAG2_READER_NODELET_NAME} src/ftag2_reader_nodelet.cpp)
target_link_libraries(${FTAG2_READER_NODELET_NAME}
  ${FTAG2_LIB_NAME}
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
)
add_dependencies(${FTAG2_READER_NODELET_NAME} ${PROJECT_NAME}_gencfg)

add_library(${FTAG2_TRACKER_NODELET_NAME} src/ftag2_tracker_nodelet.cpp)
target_link_libraries(${FTAG2_TRACKER_NODELET_NAME}
  ${FTAG2_LIB_NAME}
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
)
add_dependencies(${FTAG2_TRACKER_NODELET_NAME} ${PROJECT_NAME}_gencfg)

#add_executable(ros_cam_testbench src/ros_cam_testbench.cpp)
#target_link_libraries(ros_cam_testbench
#  ${FTAG2_LIB_NAME}
#  ${OpenCV_LIBRARIES}
#  ${catkin_LIBRARIES}
#  yaml-cpp
# )
#add_dependencies(ros_cam_testbench ${PROJECT_NAME}_gencfg)

#add_executable(freq_testbench src/freq_testbench.cpp)
#target_link_libraries(freq_testbench
#  ${FTAG2_LIB_NAME}
#  ${OpenCV_LIBRARIES}
#  ${catkin_LIBRARIES}
#)
#add_dependencies(freq_testbench ${PROJECT_NAME}_gencfg ${PROJECT_NAME}_generate_messages_cpp)


#############
## Install ##
#############

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
install(TARGETS ${FTAG2_LIB_NAME} ${FTAG2_READER_NODELET_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )

## Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
  )

install(FILES nodelet_plugins.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  )

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  )

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
