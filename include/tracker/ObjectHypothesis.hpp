/*
 * ObjectHypothesis.h
 *
 *  Created on: 2013-10-23
 *      Author: dacocp
 */

#include <opencv2/imgproc/imgproc.hpp>
#include <random>
#include <iostream>
#include <chrono>
#include <ctime>
#include <limits>


#define sigma_pos 10
#define sigma_scale 2

#ifndef OBJECTHYPOTHESIS_H_
#define OBJECTHYPOTHESIS_H_

#define PI 3.141592653589793238462643383279502884

using namespace std;

class ObjectHypothesis {

private:
//	cv::Vec2f centroid;
//	cv::Vec2f size;
	std::vector< cv::Vec2i >corners;

public:
	ObjectHypothesis();
//	ObjectHypothesis(float x, float y, float sx, float sy){centroid[0]=x,centroid[1]=y,size[0]=sx,size[1]=sy;};
	ObjectHypothesis(std::vector< cv::Vec2i >corners){this->corners = corners;};
	ObjectHypothesis(int SX, int SY);
	virtual ~ObjectHypothesis();
	void motionUpdate();
	double measurementUpdate(std::vector<ObjectHypothesis> detections);
	std::vector<cv::Vec2i> getCorners(){return corners;}
};

#endif /* OBJECTHYPOTHESIS_H_ */
