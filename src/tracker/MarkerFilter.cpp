/*
 * MarkerFilter.cpp
 *
 *  Created on: Mar 5, 2014
 *      Author: dacocp
 */

#include "tracker/MarkerFilter.hpp"

MarkerFilter::MarkerFilter( FTag2Marker detection ) {
	std::vector<FTag2Pose> observations;
	observations.push_back(detection.pose);
	PF = ParticleFilter(100, observations, ParticleFilter::clock::now() );
	IF = PayloadFilter();
	frames_without_detection = 0;
};

void MarkerFilter::step( FTag2Marker detection ) {
	PF.step(detection.pose);
	IF.step(detection.payload);
	hypothesis.corners = detection.corners;
	hypothesis.pose = PF.computeModePose();
	hypothesis.payload = IF.getFilteredPayload();
	frames_without_detection = 0;
};

void MarkerFilter::step() {
	PF.step();
	IF.step();
	hypothesis.pose = PF.computeModePose();
	hypothesis.payload = IF.getFilteredPayload();
	frames_without_detection++;
};
