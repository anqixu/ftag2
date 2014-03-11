/*
 * MarkerFilter.hpp
 *
 *  Created on: Mar 4, 2014
 *      Author: dacocp
 */

#ifndef MARKERFILTER_HPP_
#define MARKERFILTER_HPP_

#include "common/FTag2Marker.hpp"
#include "tracker/ParticleFilter.hpp"
#include "tracker/PayloadFilter.hpp"

using namespace std;

class MarkerFilter {

private:
	FTag2Marker hypothesis;
	FTag2Marker detectedTag;
	int frames_without_detection;
	ParticleFilter PF;
	PayloadFilter IF;

public:
	MarkerFilter(){ frames_without_detection = 0;};
	MarkerFilter(FTag2Marker detection);
	virtual ~MarkerFilter() {};
	FTag2Marker getHypothesis() { return hypothesis; }
	int get_frames_without_detection() { return frames_without_detection; }
	void step( FTag2Marker detection );
	void step( );
	double sumOfStds();
};

#endif /* MARKERFILTER_HPP_ */
