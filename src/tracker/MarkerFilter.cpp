/*
 * MarkerFilter.cpp
 *
 *  Created on: Mar 5, 2014
 *      Author: dacocp
 */

#include "tracker/MarkerFilter.hpp"

int MarkerFilter::num_Markers = 0;
using namespace Kalman;

MarkerFilter::MarkerFilter( FTag2Marker detection ) {
	got_detection_in_current_frame = true;
	num_Markers++;
	marker_id = num_Markers;
//	std::vector<FTag2Pose> observations;
//	observations.push_back(detection.pose);
	KF = KalmanTrack(detection.pose);

//	PF = ParticleFilter(detection.pose);
	IF = PayloadFilter();
	frames_without_detection = 0;
};

void MarkerFilter::step( FTag2Marker detection, double quadSizeM, cv::Mat cameraIntrinsic, cv::Mat cameraDistortion ) {
	got_detection_in_current_frame = true;
//	PF.step(detection.pose);
	hypothesis.corners = detection.corners;
	hypothesis.pose = detection.pose;
//	hypothesis.pose = PF.getEstimatedPose();

	KF.step_( detection.pose );
	hypothesis.pose = KF.getEstimatedPose();

//	std::cout << "KF Pose: "
//			<< hypothesis.pose.position_x << ", "
//			<< hypothesis.pose.position_y << ", "
//			<< hypothesis.pose.position_z << ", "
//			<< hypothesis.pose.orientation_w << ", "
//			<< hypothesis.pose.orientation_x << ", "
//			<< hypothesis.pose.orientation_y << ", "
//			<< hypothesis.pose.orientation_z << ", " << std::endl;
//	hypothesis.back_proj_corners = backProjectQuad( hypothesis.pose.position_x,
//			hypothesis.pose.position_y, hypothesis.pose.position_z,
//			hypothesis.pose.orientation_w, hypothesis.pose.orientation_x,
//			hypothesis.pose.orientation_y, hypothesis.pose.orientation_z,
//			quadSizeM, cameraIntrinsic, cameraDistortion );

//	PF.publishTrackedPose(marker_id);
//	PF.displayParticles(marker_id);

	IF.step(detection.payload);
	IF.getFilteredPayload();
	hypothesis.payload = IF.getFilteredPayload();

	frames_without_detection = 0;
};

void MarkerFilter::step( double quadSizeM, cv::Mat cameraIntrinsic, cv::Mat cameraDistortion  ) {
	got_detection_in_current_frame = false;
//	std::cout << "MarkerFilter: stepping without detection" << std::endl;
//	PF.step();
	KF.step_( );
	hypothesis.pose = KF.getEstimatedPose();
//	hypothesis.back_proj_corners = backProjectQuad( hypothesis.pose.position_x,
//				hypothesis.pose.position_y, hypothesis.pose.position_z,
//				hypothesis.pose.orientation_w, hypothesis.pose.orientation_x,
//				hypothesis.pose.orientation_y, hypothesis.pose.orientation_z,
//				quadSizeM, cameraIntrinsic, cameraDistortion );
//	PF.publishTrackedPose(marker_id);
//	PF.displayParticles(marker_id);

	IF.step();
	hypothesis.payload = IF.getFilteredPayload();

	frames_without_detection++;
};

void MarkerFilter::updateParameters(int numberOfParticles_, double position_std_, double orientation_std_, double position_noise_std_, double orientation_noise_std_, double velocity_noise_std_, double acceleration_noise_std_) {
	PF.updateParameters(numberOfParticles_, position_std_, orientation_std_, position_noise_std_, orientation_noise_std_, velocity_noise_std_, acceleration_noise_std_);
}
