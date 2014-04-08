/*
 * MarkerFilter.cpp
 *
 *  Created on: Mar 5, 2014
 *      Author: dacocp
 */

#include "tracker/MarkerFilter.hpp"

int MarkerFilter::num_Markers = 0;

MarkerFilter::MarkerFilter( FTag2Marker detection ) {
	num_Markers++;
	marker_id = num_Markers;
	std::vector<FTag2Pose> observations;
	observations.push_back(detection.pose);
	PF = ParticleFilter(100, observations);
	IF = PayloadFilter();
	frames_without_detection = 0;
};

void MarkerFilter::step( FTag2Marker detection ) {
	//std::cout << "MarkerFilter: stepping with detection" << std::endl;
	PF.step(detection.pose);
	hypothesis.corners = detection.corners;
	hypothesis.pose = PF.computeModePose();
//	PF.displayParticles(marker_id);

	IF.step(detection.payload);
	IF.getFilteredPayload();
	hypothesis.payload = IF.getFilteredPayload();

	frames_without_detection = 0;
};

void MarkerFilter::step() {
	//std::cout << "MarkerFilter: stepping without detection" << std::endl;
	PF.step();
	hypothesis.pose = PF.computeModePose();
//	PF.displayParticles(marker_id);

	IF.step();
	hypothesis.payload = IF.getFilteredPayload();

	frames_without_detection++;
};

void MarkerFilter::updateParameters(int numberOfParticles_, double position_std_, double orientation_std_, double position_noise_std_, double orientation_noise_std_, double velocity_noise_std_, double acceleration_noise_std_) {
	PF.updateParameters(numberOfParticles_, position_std_, orientation_std_, position_noise_std_, orientation_noise_std_, velocity_noise_std_, acceleration_noise_std_);
}
