/*
 * ParticleFilter.cpp
 *
 *  Created on: 2013-10-23
 *      Author: dacocp
 */

#include "tracker/ParticleFilter.hpp"

float ParticleFilter::sampling_percent = 0.9;

ParticleFilter::~ParticleFilter() {
	// TODO Auto-generated destructor stub
}

ParticleFilter::ParticleFilter(int numP, int SX, int SY){
	cout << "Creating PF" << endl;
	this->SX = SX;
	this->SY = SY;
	number_of_particles = numP;

	weights.resize(number_of_particles);
	particles.resize(number_of_particles);
	srand(time(NULL));
	for ( unsigned int i=0; i < number_of_particles; i++ )
	{
		weights[i] = 1.0/number_of_particles;
		particles[i] = ObjectHypothesis(SX,SY);
		std::vector<cv::Vec2i> corners = particles[i].getCorners();
		cout << "Corners of " << i << ": { (" << corners[0][0] << ", " << corners[0][1] << "), (" << corners[1][0] << ", "
						<< corners[1][1] << "), (" << corners[2][0] << ", " << corners[2][1] << "), (" << corners[3][0] << ", "
						<< corners[3][1] << ") }" << endl;
		//http://stackoverflow.com/questions/2259476/rotating-a-point-about-another-point-2d
	}
	cout << "Cloud created" << endl;
}

void ParticleFilter::motionUpdate() {
	for( unsigned int i=0; i < number_of_particles; i++ )
		particles[i].motionUpdate();
}

void ParticleFilter::measurementUpdate(std::vector<ObjectHypothesis> detections) {
}

void ParticleFilter::drawParticles(cv::Mat img){
	cout << "Drawing the filter" << endl;

	for ( unsigned int i=0; i<number_of_particles; i++ )
	{
		std::vector<cv::Vec2i> corners = particles[i].getCorners();

		cout << "Corners of " << i << ": { (" << corners[0][0] << ", " << corners[0][1] << "), (" << corners[1][0] << ", "
								<< corners[1][1] << "), (" << corners[2][0] << ", " << corners[2][1] << "), (" << corners[3][0] << ", "
								<< corners[3][1] << ") }" << endl;

		cv::line( img, cv::Point((int)corners[0][0], (int)corners[0][1]), cv::Point((int)corners[1][0], (int)corners[1][1]), cv::Scalar(255,0,0), 1, 8 );
		cv::line( img, cv::Point((int)corners[1][0], (int)corners[1][1]), cv::Point((int)corners[2][0], (int)corners[2][1]), cv::Scalar(255,0,0), 1, 8 );
		cv::line( img, cv::Point((int)corners[2][0], (int)corners[2][1]), cv::Point((int)corners[3][0], (int)corners[3][1]), cv::Scalar(255,0,0), 1, 8 );
		cv::line( img, cv::Point((int)corners[3][0], (int)corners[3][1]), cv::Point((int)corners[0][0], (int)corners[0][1]), cv::Scalar(255,0,0), 1, 8 );
		//PC.drawObject(img);
	}
	cout << "Finished creating image" << endl;
	cv::imshow("Particles", img);
	cout << "Finished drawing image" << endl;
}
