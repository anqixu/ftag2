/*
 * ObjectHypothesis.cpp
 *
 *  Created on: 2013-10-23
 *      Author: dacocp
 */

#include "tracker/ObjectHypothesis.hpp"

double phi(double x)
{
    // constants
    double a1 =  0.254829592;
    double a2 = -0.284496736;
    double a3 =  1.421413741;
    double a4 = -1.453152027;
    double a5 =  1.061405429;
    double p  =  0.3275911;

    // Save the sign of x
    int sign = 1;
    if (x < 0)
        sign = -1;
    x = fabs(x)/sqrt(2.0);

    // A&S formula 7.1.26
    double t = 1.0/(1.0 + p*x);
    double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);

    return 0.5*(1.0 + sign*y);
}

ObjectHypothesis::ObjectHypothesis() {
	// TODO Auto-generated constructor stub
//	corners = std::vector<cv::Vec2i>(4);
}

ObjectHypothesis::ObjectHypothesis(FTag2Marker marker, bool addNoise ) {
	this->pose = pose;
	if ( addNoise == true )
	{
		ompl::base::StateSpacePtr space(new ompl::base::SO3StateSpace());
		ompl::base::ScopedState<ompl::base::SO3StateSpace> stateMean(space);
		ompl::base::ScopedState<ompl::base::SO3StateSpace> stateNew(space);

		stateMean->as<ompl::base::SO3StateSpace::StateType>()->x = marker.orientation_x;
		stateMean->as<ompl::base::SO3StateSpace::StateType>()->y = marker.orientation_y;
		stateMean->as<ompl::base::SO3StateSpace::StateType>()->z = marker.orientation_z;
		stateMean->as<ompl::base::SO3StateSpace::StateType>()->w = marker.orientation_w;

		ompl::base::SO3StateSampler SO3ss(space->as<ompl::base::SO3StateSpace>());

		SO3ss.sampleGaussian(stateNew->as<ompl::base::SO3StateSpace::StateType>(),
				stateMean->as<ompl::base::SO3StateSpace::StateType>(), sigma_init_rot);

		pose.orientation_x = stateNew->as<ompl::base::SO3StateSpace::StateType>()->x;
		pose.orientation_y = stateNew->as<ompl::base::SO3StateSpace::StateType>()->y;
		pose.orientation_z = stateNew->as<ompl::base::SO3StateSpace::StateType>()->z;
		pose.orientation_w = stateNew->as<ompl::base::SO3StateSpace::StateType>()->w;

		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::default_random_engine generator(seed);
		std::normal_distribution<double> distribution_pos(0,sigma_init_pos);

		pose.position_x = marker.position_x + distribution_pos(generator);
		pose.position_y = marker.position_y + distribution_pos(generator);
		pose.position_z = marker.position_z + distribution_pos(generator);
	}
}

ObjectHypothesis::~ObjectHypothesis() {
	// TODO Auto-generated destructor stub
}

void ObjectHypothesis::motionUpdate(double position_noise_std, double orientation_noise_std) {

    ompl::base::StateSpacePtr space(new ompl::base::SO3StateSpace());
    ompl::base::ScopedState<ompl::base::SO3StateSpace> stateMean(space);
    ompl::base::ScopedState<ompl::base::SO3StateSpace> stateNew(space);

    stateMean->as<ompl::base::SO3StateSpace::StateType>()->x = pose.orientation_x;
	stateMean->as<ompl::base::SO3StateSpace::StateType>()->y = pose.orientation_y;
    stateMean->as<ompl::base::SO3StateSpace::StateType>()->z = pose.orientation_z;
    stateMean->as<ompl::base::SO3StateSpace::StateType>()->w = pose.orientation_w;

    ompl::base::SO3StateSampler SO3ss(space->as<ompl::base::SO3StateSpace>());

    SO3ss.sampleGaussian(stateNew->as<ompl::base::SO3StateSpace::StateType>(),
    		stateMean->as<ompl::base::SO3StateSpace::StateType>(), orientation_noise_std);

    pose.orientation_x = stateNew->as<ompl::base::SO3StateSpace::StateType>()->x;
    pose.orientation_y = stateNew->as<ompl::base::SO3StateSpace::StateType>()->y;
    pose.orientation_z = stateNew->as<ompl::base::SO3StateSpace::StateType>()->z;
    pose.orientation_w = stateNew->as<ompl::base::SO3StateSpace::StateType>()->w;

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::normal_distribution<double> distribution_pos(0, position_noise_std);

	pose.position_x += distribution_pos(generator);
	pose.position_y += distribution_pos(generator);
	pose.position_z += distribution_pos(generator);
}

/*
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
		//http://stackoverflow.com/questions/2259476/rotating-a-point-about-another-point-2d
	cout << "Corners: " << ": { (" << corners[0][0] << ", " << corners[0][1] << "), (" << corners[1][0] << ", "
							<< corners[1][1] << "), (" << corners[2][0] << ", " << corners[2][1] << "), (" << corners[3][0] << ", "
							<< corners[3][1] << ") }" << endl;
}
*/


double ObjectHypothesis::measurementUpdate(std::vector<FTag2Marker> detections, double position_std, double orientation_std) {

	if ( detections.size() == 0 )
		return log_weight;

	double maxP = -std::numeric_limits<double>::infinity();
	int max_log_prob_index = 0;

	ompl::base::StateSpacePtr space(new ompl::base::SO3StateSpace());
	ompl::base::ScopedState<ompl::base::SO3StateSpace> stateParticle(space);
	ompl::base::ScopedState<ompl::base::SO3StateSpace> stateDetection(space);
	stateParticle->as<ompl::base::SO3StateSpace::StateType>()->x = pose.orientation_x;
	stateParticle->as<ompl::base::SO3StateSpace::StateType>()->y = pose.orientation_y;
	stateParticle->as<ompl::base::SO3StateSpace::StateType>()->z = pose.orientation_z;
	stateParticle->as<ompl::base::SO3StateSpace::StateType>()->w = pose.orientation_w;
	for ( unsigned int i = 0; i < detections.size(); i++)
	{
		stateDetection->as<ompl::base::SO3StateSpace::StateType>()->x = detections[i].orientation_x;
		stateDetection->as<ompl::base::SO3StateSpace::StateType>()->y = detections[i].orientation_y;
		stateDetection->as<ompl::base::SO3StateSpace::StateType>()->z = detections[i].orientation_z;
		stateDetection->as<ompl::base::SO3StateSpace::StateType>()->w = detections[i].orientation_w;

		double rotation_dist = space->as<ompl::base::SO3StateSpace>()->distance(stateDetection->as<ompl::base::SO3StateSpace::StateType>(),stateParticle->as<ompl::base::SO3StateSpace::StateType>());
		double rotation_log_prob = log_normal_pdf(rotation_dist, 0, orientation_std);
		//double rotation_prob = 2*phi( (-1.0*rotation_dist) / rotation_std );

		double position_dist = sqrt( (pose.position_x - detections[i].position_x)*(pose.position_x - detections[i].position_x) + (pose.position_y - detections[i].position_y)*(pose.position_y - detections[i].position_y) + (pose.position_z - detections[i].position_z)*(pose.position_z - detections[i].position_z));
		double position_log_prob = log_normal_pdf(position_dist, 0, position_std);
		//double position_prob = 2*phi( (-1.0*position_dist) / position_std );

		double log_prob = position_log_prob+rotation_log_prob;
		if ( log_prob > maxP )
		{
			max_log_prob_index = i;
			maxP = log_prob;
		}
	}
	log_weight = maxP;
	return log_weight;
}
