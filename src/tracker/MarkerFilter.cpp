/*
 * MarkerFilter.cpp
 *
 *  Created on: Mar 5, 2014
 *      Author: dacocp
 */

#include "tracker/MarkerFilter.hpp"

int MarkerFilter::num_Markers = 0;

MarkerFilter::MarkerFilter( FTag2Marker detection,  double quadSizeM, cv::Mat cameraIntrinsic, cv::Mat cameraDistortion ):
    detectedTag(detection.payload.type),
    IF(detection.payload.type),
    hypothesis(detection.payload.type) {
	got_detection_in_current_frame = true;
	num_Markers++;
	marker_id = num_Markers;

	KF = Kalman(detection.pose);
	hypothesis.pose = KF.getEstimatedPose();
	hypothesis.back_proj_corners = backProjectQuad( hypothesis.pose.position_x,
				hypothesis.pose.position_y, hypothesis.pose.position_z,
				hypothesis.pose.orientation_w, hypothesis.pose.orientation_x,
				hypothesis.pose.orientation_y, hypothesis.pose.orientation_z,
				quadSizeM, cameraIntrinsic, cameraDistortion );

	IF = PayloadFilter(detection.payload.type);
	IF.step(detection.payload);
	frames_without_detection = 0;
	active = false;
};

void MarkerFilter::step( FTag2Marker detection, double quadSizeM, cv::Mat cameraIntrinsic, cv::Mat cameraDistortion ) {
	got_detection_in_current_frame = true;
	hypothesis.tagCorners = detection.tagCorners;
	KF.step( detection.pose );
	hypothesis.pose = KF.getEstimatedPose();
	hypothesis.back_proj_corners = backProjectQuad( hypothesis.pose.position_x,
			hypothesis.pose.position_y, hypothesis.pose.position_z,
			hypothesis.pose.orientation_w, hypothesis.pose.orientation_x,
			hypothesis.pose.orientation_y, hypothesis.pose.orientation_z,
			quadSizeM, cameraIntrinsic, cameraDistortion );

	IF.step(detection.payload);
	IF.getFilteredPayload();
	hypothesis.payload = IF.getFilteredPayload();

	frames_without_detection = 0;
	active = true;
};

void MarkerFilter::step( double quadSizeM, cv::Mat cameraIntrinsic, cv::Mat cameraDistortion  ) {
	got_detection_in_current_frame = false;
	KF.step();
	hypothesis.pose = KF.getEstimatedPose();
	hypothesis.back_proj_corners = backProjectQuad( hypothesis.pose.position_x,
				hypothesis.pose.position_y, hypothesis.pose.position_z,
				hypothesis.pose.orientation_w, hypothesis.pose.orientation_x,
				hypothesis.pose.orientation_y, hypothesis.pose.orientation_z,
				quadSizeM, cameraIntrinsic, cameraDistortion );

	IF.step();
	hypothesis.payload = IF.getFilteredPayload();

	frames_without_detection++;
};

void MarkerFilter::updateParameters() {
}
