/*
 * ParticleFilter.h
 *
 *  Created on: 2013-10-23
 *      Author: dacocp
 */

#include "tracker/ObjectHypothesis.hpp"
#include "common/FTag2Pose.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <ros/ros.h>
#include "tf/LinearMath/Transform.h"
#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_broadcaster.h>

#include <chrono>

#ifndef PARTICLEFILTER_H_
#define PARTICLEFILTER_H_

#define PI 3.141592653589793238462643383279502884

using namespace std;

class ParticleFilter {
public:
  typedef std::chrono::system_clock clock;
  typedef std::chrono::system_clock::time_point time_point;

private:
	unsigned int number_of_particles;
	std::vector< ObjectHypothesis > particles;
	std::vector<double> weights;
	double log_max_weight;
	double log_sum_of_weights;
	static double sampling_percent;
	bool disable_resampling;
	double position_std;
	double orientation_std;
	double position_noise_std;
	double orientation_noise_std;
	double velocity_noise_std;
	double acceleration_noise_std;
	time_point starting_time;
	time_point current_time;
	time_point last_observation_time;
	double current_time_step_ms;
	FTag2Pose estimated_pose;
	vector<FTag2Pose> new_observations;

public:
	ParticleFilter(){ number_of_particles = 100; disable_resampling = false; };
	ParticleFilter(int numP, std::vector<FTag2Pose> observations, double position_std_, double orientation_std_,
			double position_noise_std, double orientation_noise_std, double velocity_noise_std, double acceleration_noise_std_,
			time_point starting_time_);
	ParticleFilter(int numP, FTag2Pose observation, ParticleFilter::time_point starting_time_);
	void setParameters(int numP, double position_std_, double orientation_std_,
			double position_noise_std_, double orientation_noise_std_, double velocity_noise_std_, double acceleration_noise_std);
	virtual ~ParticleFilter();
	void step(FTag2Pose observation);
	void step();
	void motionUpdate(time_point new_time);
	void normalizeWeights();
	void resample();
	void measurementUpdate(std::vector<FTag2Pose> observations);
	void displayParticles();
	FTag2Pose computeMeanPose();
	FTag2Pose computeModePose();
};

#endif /* PARTICLEFILTER_H_ */
