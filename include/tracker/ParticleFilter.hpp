/*
 * ParticleFilter.h
 *
 *  Created on: 2013-10-23
 *      Author: dacocp
 */

#include "tracker/ObjectHypothesis.hpp"
#include "common/FTag2.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <ros/ros.h>
#include "tf/LinearMath/Transform.h"
#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_broadcaster.h>

#include <mutex>

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
//    std::mutex paramsMutex;
	std::vector< ObjectHypothesis > particles;
	std::vector<double> weights;
	static double sampling_percent;
	bool disable_resampling;
	static unsigned int number_of_particles;
	static double position_std;
	static double orientation_std;
	static double position_noise_std;
	static double orientation_noise_std;
	static double velocity_noise_std;
	static double acceleration_noise_std;
	double log_sum_of_weights;
	double log_max_weight;
	time_point starting_time;
	time_point current_time;
	time_point last_observation_time;
	double current_time_step_ms;
	FTag2Pose estimated_pose;
	vector<FTag2Pose> new_observations;

public:
	ParticleFilter();
	ParticleFilter(std::vector<FTag2Pose> observations);
	ParticleFilter(std::vector<FTag2Pose> observations, double position_std_, double orientation_std_,
			double position_noise_std, double orientation_noise_std, double velocity_noise_std, double acceleration_noise_std_);
	ParticleFilter(FTag2Pose observation);
	void updateParameters(int numP, double position_std_, double orientation_std_,
			double position_noise_std_, double orientation_noise_std_, double velocity_noise_std_, double acceleration_noise_std);
	virtual ~ParticleFilter();
	void step(FTag2Pose observation);
	void step();
	void motionUpdate( );
	void normalizeWeights();
	void resample();
	void measurementUpdate(std::vector<FTag2Pose> observations);
	void displayParticles(int frame_id);
	void publishTrackedPose(int marker_id);
	FTag2Pose computeMeanPose();
	FTag2Pose computeModePose();
	FTag2Pose getEstimatedPose() { return estimated_pose; };
};

#endif /* PARTICLEFILTER_H_ */
