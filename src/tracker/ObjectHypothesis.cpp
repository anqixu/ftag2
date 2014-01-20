/*
 * ObjectHypothesis.cpp
 *
 *  Created on: 2013-10-23
 *      Author: dacocp
 */

#include "tracker/ObjectHypothesis.hpp"

ObjectHypothesis::ObjectHypothesis() {
	// TODO Auto-generated constructor stub
	corners = std::vector<cv::Vec2i>(4);
}

ObjectHypothesis::~ObjectHypothesis() {
	// TODO Auto-generated destructor stub
}

void ObjectHypothesis::motionUpdate() {
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::normal_distribution<double> distribution_pos(0,sigma_pos);
	std::normal_distribution<double> distribution_scale(0,sigma_scale);

	for ( unsigned int i = 0; i<4 ; i++ )
	{
		corners[i][0] += distribution_pos(generator);
		corners[i][1] += distribution_pos(generator);
	}
//	centroid[0] += distribution_pos(generator);
//	centroid[1] += distribution_pos(generator);
//	size[0] += distribution_scale(generator);
//	size[1] += distribution_scale(generator);
}

ObjectHypothesis::ObjectHypothesis(int SX, int SY){
	corners = std::vector<cv::Vec2i>(4);
	cout << "CREATING HYPOTHESIS" << endl;

	int cx = rand()%(SX+1);
	int cy = rand()%(SY+1);
	int sx = rand()%(SX/4);
	int sy = rand()%(SY/4);
	double theta = (rand()%360)*PI/180.0;

	float px0,px1,px2,px3,py0,py1,py2,py3;
	px0 = cx - sx/2;
	py0 = cy - sy/2;
	px1 = cx - sx/2;
	py1 = cy + sy/2;
	px2 = cx + sx/2;
	py2 = cy + sy/2;
	px3 = cx + sx/2;
	py3 = cy - sy/2;

	px0 = cos(theta) * (px0-cx) - sin(theta) * (py0-cy) + cx;
	py0 = sin(theta) * (px0-cx) + cos(theta) * (py0-cy) + cy;
	px1  = cos(theta) * (px1-cx) - sin(theta) * (py1-cy) + cx;
	py1 = sin(theta) * (px1-cx) + cos(theta) * (py1-cy) + cy;
	px2 = cos(theta) * (px2-cx) - sin(theta) * (py2-cy) + cx;
	py2 = sin(theta) * (px2-cx) + cos(theta) * (py2-cy) + cy;
	px3 = cos(theta) * (px3-cx) - sin(theta) * (py3-cy) + cx;
	py3 = sin(theta) * (px3-cx) + cos(theta) * (py3-cy) + cy;

	if (px0 < 0)
		corners[0][0] = 0;
	else if ( px0 > SX )
		corners[0][0] = SX;
	else
		corners[0][0] = px0;

	if (py0 < 0)
		corners[0][1] = 0;
	else if ( py0 > SY )
		corners[0][1] = SY;
	else
		corners[0][1] = py0;

	if (px1 < 0)
		corners[1][0] = 0;
	else if ( px1 > SX )
		corners[1][0] = SX;
	else
		corners[1][0] = px1;

	if (py1 < 0)
		corners[1][1] = 0;
	else if ( py1 > SY )
		corners[1][1] = SY;
	else
		corners[1][1] = py1;

	if (px2 < 0)
		corners[2][0] = 0;
	else if ( px2 > SX )
		corners[2][0] = SX;
	else
		corners[2][0] = px2;

	if (py2 < 0)
		corners[2][1] = 0;
	else if ( py2 > SY )
		corners[2][1] = SY;
	else
		corners[2][1] = py2;

	if (px3 < 0)
		corners[3][0] = 0;
	else if ( px3 > SX )
		corners[3][0] = SX;
	else
		corners[3][0] = px3;

	if (py3 < 0)
		corners[3][1] = 0;
	else if ( py3 > SY )
		corners[3][1] = SY;
	else
		corners[3][1] = py3;
/*		corners[0][0] = (px0>=0 && px0<SX) ? px0 : 0;
	corners[0][1] = (py0>=0 && py0<SY) ? py0 : 0;
	corners[1][0] = (px1>=0 && px1<SX) ? px1 : 0;
	corners[1][1] = (py1>=0 && py1<SY) ? py1 : 0;
	corners[2][0] = (px2>=0 && px2<SX) ? px2 : 0;
	corners[2][1] = (py2>=0 && py2<SY) ? py2 : 0;
	corners[3][0] = (px3>=0 && px3<SX) ? px3 : 0;
	corners[3][1] = (py3>=0 && py3<SY) ? py3 : 0;
*/
			//http://stackoverflow.com/questions/2259476/rotating-a-point-about-another-point-2d
	cout << "Corners: " << ": { (" << corners[0][0] << ", " << corners[0][1] << "), (" << corners[1][0] << ", "
							<< corners[1][1] << "), (" << corners[2][0] << ", " << corners[2][1] << "), (" << corners[3][0] << ", "
							<< corners[3][1] << ") }" << endl;
}

double ObjectHypothesis::measurementUpdate(std::vector<ObjectHypothesis> detections) {

	return 0.0;
}
