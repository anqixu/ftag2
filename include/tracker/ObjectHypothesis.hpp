/*
 * ObjectHypothesis.h
 *
 *  Created on: 2013-10-23
 *      Author: dacocp
 */

#include <ompl/base/spaces/SO3StateSpace.h>
#include <ompl/base/State.h>
#include <ompl/base/ScopedState.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <random>
#include <iostream>
#include <chrono>
#include <ctime>
#include <limits>
#include <math.h>

#include "common/FTag2Marker.hpp"
#include "tracker/utils.hpp"

#ifndef OBJECTHYPOTHESIS_H_
#define OBJECTHYPOTHESIS_H_

#define PI 3.141592653589793238462643383279502884

#define sigma_init_pos 0.1
#define sigma_init_rot PI/16

using namespace std;

class ObjectHypothesis {

private:
	FTag2Marker pose;
	double log_weight;
	double position_std;
	double orientation_std;
	double position_noise_std;
	double orientation_noise_std;

public:
	ObjectHypothesis();
//	ObjectHypothesis(float x, float y, float sx, float sy){centroid[0]=x,centroid[1]=y,size[0]=sx,size[1]=sy;};
	ObjectHypothesis(FTag2Marker pose, double position_std, double orientation_std, double position_noise_std, double orientation_noise_std, bool addNoise = true);
	virtual ~ObjectHypothesis();
	void motionUpdate();
	double measurementUpdate(std::vector<FTag2Marker> detections);
	FTag2Marker getPose(){return pose;}
	double getLogWeight(){return log_weight;}
	void setLogWeight(double log_weight){this->log_weight = log_weight;}
};

#endif /* OBJECTHYPOTHESIS_H_ */
