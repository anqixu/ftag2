/*
 * MarkerFilter.cpp
 *
 *  Created on: Mar 5, 2014
 *      Author: dacocp
 */

#include "tracker/MarkerFilter.hpp"

MarkerFilter::MarkerFilter( FTag2Marker detection ) {
	PF = ParticleFilter(100, detection.pose, ParticleFilter::clock::now() );
	IF = PayloadFilter();
}

void MarkerFilter::step( FTag2Marker detection ) {
	PF.step(detection.pose);
	//IF.step(detection.payload);
};

void MarkerFilter::step() {
	PF.step();
	//IF.step();
};

