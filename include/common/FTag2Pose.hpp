/*
 * FTag2Pose.hpp
 *
 *  Created on: Mar 6, 2014
 *      Author: dacocp
 */

#include "common/BaseCV.hpp"

#ifndef FTAG2POSE_HPP_
#define FTAG2POSE_HPP_

struct FTag2Pose{
	double position_x;
	double position_y;
	double position_z;

	double orientation_x;
	double orientation_y;
	double orientation_z;
	double orientation_w;

	FTag2Pose() : position_x(0), position_y(0), position_z(0),
			orientation_x(0), orientation_y(0), orientation_z(0), orientation_w(0) {

	};

	virtual ~FTag2Pose() {};
	// Returns angle between marker's normal vector and camera's ray vector,
	// in radians. Also known as angle for out-of-plane rotation.
	//
	// WARNING: behavior is undefined if pose has not been set.
	inline double getAngleFromCamera() const {
		cv::Mat rotMat = vc_math::quat2RotMat(orientation_w, orientation_x, orientation_y, orientation_z);
		// tagVec = rotMat * [0; 0; 1]; // obtain +z ray of tag's pose, in camera frame
	    // a	ngle = acos([0; 0; 1] (dot) tagVec) // obtain angle using dot product rule
	    return acos(rotMat.at<double>(2, 2));
	}
};



#endif /* FTAG2POSE_HPP_ */
