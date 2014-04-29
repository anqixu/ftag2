/*
 * ObjectHypothesis.h
 *
 *  Created on: 2013-10-23
 *      Author: dacocp
 */

//#include <ompl/base/spaces/SO3StateSpace.h>
//#include <ompl/base/State.h>
//#include <ompl/base/ScopedState.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <random>
#include <iostream>
#include <chrono>
#include <ctime>
#include <limits>
#include <math.h>

#include "common/FTag2.hpp"
#include "common/VectorAndCircularMath.hpp"

#ifndef OBJECTHYPOTHESIS_H_
#define OBJECTHYPOTHESIS_H_

#define MS_PER_FRAME 33.0

#define sigma_init_pos 0.1
#define sigma_init_rot vc_math::pi/16

using namespace std;

class ObjectHypothesis {

private:
	FTag2Pose pose;
	FTag2Pose pose_prev;
	double log_weight;
	double vel_x;
	double vel_y;
	double vel_z;
	double vel_prev_x;
	double vel_prev_y;
	double vel_prev_z;
	double accel_x;
	double accel_y;
	double accel_z;

public:
	ObjectHypothesis();
//	ObjectHypothesis(float x, float y, float sx, float sy){centroid[0]=x,centroid[1]=y,size[0]=sx,size[1]=sy;};
	ObjectHypothesis(FTag2Pose pose, bool addNoise = true);
	virtual ~ObjectHypothesis();
	void motionUpdate(double position_noise_std, double orientation_noise_std, double velocity_noise_std, double acceleration_noise_std, double current_time_step_ms);
	double measurementUpdate(std::vector<FTag2Pose> detections, double position_std, double orientation_std);
	FTag2Pose getPose(){return pose;}
	double getLogWeight(){return log_weight;}
	void setLogWeight(double log_weight){this->log_weight = log_weight;}
};

#endif /* OBJECTHYPOTHESIS_H_ */
