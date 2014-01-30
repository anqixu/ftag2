/*
 * ParticleFilter.h
 *
 *  Created on: 2013-10-23
 *      Author: dacocp
 */

#include "ObjectHypothesis.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <ros/ros.h>
#include "tf/LinearMath/Transform.h"
#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_broadcaster.h>


#ifndef PARTICLEFILTER_H_
#define PARTICLEFILTER_H_

#define PI 3.141592653589793238462643383279502884

using namespace std;

class ParticleFilter {
private:
	unsigned int number_of_particles;
	std::vector< ObjectHypothesis > particles;
	std::vector<double> weights;
	double log_max_weight;
	double log_sum_of_weights;
	static double sampling_percent;
	double tagSize;
	bool disable_resampling;
	double position_std;
	double orientation_std;
	double position_noise_std;
	double orientation_noise_std;

public:
	ParticleFilter(){ number_of_particles = 100; disable_resampling = false; };
	ParticleFilter(int numP, double tagSize, std::vector<FTag2Marker> detections, double position_std, double orientation_std, double position_noise_std, double orientation_noise_std);
	void setParameters(int numP, double tagSize, double position_std, double orientation_std, double position_noise_std, double orientation_noise_std);
	virtual ~ParticleFilter();
	void motionUpdate();
	void normalizeWeights();
	void resample();
	void measurementUpdate(std::vector<FTag2Marker> detections);
	void displayParticles();
	FTag2Marker computeMeanPose();
};

#endif /* PARTICLEFILTER_H_ */
