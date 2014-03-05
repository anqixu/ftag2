/*
 * MarkerFilter.hpp
 *
 *  Created on: Mar 4, 2014
 *      Author: dacocp
 */

#ifndef MARKERFILTER_HPP_
#define MARKERFILTER_HPP_

#include "common/FTag2Marker.hpp"

using namespace std;

class MarkerFilter {

private:
	FTag2Marker hypothesis;
	FTag2Marker detectedTag;
	int frames_without_detection;

public:
	MarkerFilter(){};
	virtual ~MarkerFilter() {};
	double getSumOfStds() { return hypothesis.sumOfStds; }
	FTag2Marker getHypothesis() { return hypothesis; }
};

#endif /* MARKERFILTER_HPP_ */
