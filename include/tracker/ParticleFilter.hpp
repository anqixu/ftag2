/*
 * ParticleFilter.h
 *
 *  Created on: 2013-10-23
 *      Author: dacocp
 */

#include "ObjectHypothesis.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#ifndef PARTICLEFILTER_H_
#define PARTICLEFILTER_H_

#define PI 3.141592653589793238462643383279502884

using namespace std;

class ParticleFilter {
private:
	unsigned int number_of_particles;
	std::vector< ObjectHypothesis > particles;
	std::vector<double> weights;
	static float sampling_percent;
	int SX;
	int SY;

public:
	ParticleFilter(){ number_of_particles = 100; };
	ParticleFilter(int numP, int SX, int SY);
	virtual ~ParticleFilter();
	void motionUpdate();
	void resample();
	void measurementUpdate(std::vector<ObjectHypothesis> detections);
	void drawParticles(cv::Mat img);
};

#endif /* PARTICLEFILTER_H_ */
