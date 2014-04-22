/*
 * Ftag2Tracker.hpp
 *
 *  Created on: Mar 4, 2014
 *      Author: dacocp
 */

#ifndef FTAG2TRACKER_HPP_
#define FTAG2TRACKER_HPP_

#include "tracker/MarkerFilter.hpp"
#include "common/FTag2.hpp"
#include <cmath>

using namespace std;

class FTag2Tracker {

private:
	std::vector<MarkerFilter> filters_with_match;
	std::vector<MarkerFilter> not_matched;
	std::vector<MarkerFilter> ready_to_be_killed;
	std::vector<FTag2Marker> detection_matches;
	std::vector<FTag2Marker> to_be_spawned;
	constexpr static int MAX_FRAMES_NO_DETECTION = 50;

public:
	std::vector<MarkerFilter> filters;
	FTag2Tracker() {};
	virtual ~FTag2Tracker() {};
	void step(std::vector<FTag2Marker> detectedTags, double quadSizeM, cv::Mat cameraIntrinsic, cv::Mat cameraDistortion );
	void correspondence(std::vector<FTag2Marker> detectedTags);
	void spawnFilters();
	void updateFilters();
	void killFilters();
	void updateParameters(int numberOfParticles_, double position_std_, double orientation_std_, double position_noise_std_, double orientation_noise_std_, double velocity_noise_std_, double acceleration_noise_std_);
};

#endif /* FTAG2TRACKER_HPP_ */
