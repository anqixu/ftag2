/*
 * MarkerFilter.hpp
 *
 *  Created on: Mar 4, 2014
 *      Author: dacocp
 */

#ifndef MARKERFILTER_HPP_
#define MARKERFILTER_HPP_

#include "common/FTag2.hpp"
#include "tracker/PayloadFilter.hpp"
#include "detector/FTag2Detector.hpp"
#include "tracker/KalmanTrack.hpp"

using namespace std;

class MarkerFilter {

private:
	FTag2Marker detectedTag;
	int frames_without_detection;
	PayloadFilter IF;
	KalmanTrack KF;
	static int num_Markers;
	int marker_id;

public:
	bool got_detection_in_current_frame;
	FTag2Marker hypothesis;
	MarkerFilter(){ frames_without_detection = 0; got_detection_in_current_frame = false; };
	MarkerFilter(FTag2Marker detection);
	virtual ~MarkerFilter() {};
	FTag2Marker getHypothesis() { return hypothesis; }
	int get_frames_without_detection() { return frames_without_detection; }
	void step( FTag2Marker detection, double quadSizeM, cv::Mat cameraIntrinsic, cv::Mat cameraDistortion );
	void step( double quadSizeM, cv::Mat cameraIntrinsic, cv::Mat cameraDistortion );
	void updateParameters();
};

#endif /* MARKERFILTER_HPP_ */
