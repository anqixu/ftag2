/*
 * Ftag2Tracker.hpp
 *
 *  Created on: Mar 4, 2014
 *      Author: dacocp
 */

#ifndef FTAG2TRACKER_HPP_
#define FTAG2TRACKER_HPP_

#include "tracker/MarkerFilter.hpp"
#include "common/FTag2Marker.hpp"
#include <math.h>

using namespace std;

class FTag2Tracker {

private:
	std::vector<MarkerFilter> filters;
	std::vector<MarkerFilter> filters_with_match;
	std::vector<MarkerFilter> not_matched;
	std::vector<MarkerFilter> ready_to_be_killed;
	std::vector<FTag2Marker> detection_matches;
	std::vector<FTag2Marker> to_be_spawned;
	int MAX_FRAMES_NO_DETECTION = 10;

public:
	FTag2Tracker();
	virtual ~FTag2Tracker();
	void director(std::vector<FTag2Marker> detectedTags);
	void correspondence(std::vector<FTag2Marker> detectedTags);
	void spawnFilters();
	void updateFilters();
	void killFilters();
};

#endif /* FTAG2TRACKER_HPP_ */
