/*
 * ObjectHypothesis.h
 *
 *  Created on: 2013-10-23
 *      Author: dacocp
 */

#ifndef Kalman_H_
#define Kalman_H_

#include "common/FTag2.hpp"
#include "common/VectorAndCircularMath.hpp"

#include <random>
#include <iostream>
#include <chrono>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <cmath>

#include <tf/transform_datatypes.h>

#define MS_PER_FRAME 33.0

using namespace std;
using namespace cv;
using std::chrono::milliseconds;
using std::chrono::duration_cast;
using std::chrono::system_clock;

class Kalman {

private:

protected:
    FTag2Pose current_observation;
    bool new_filter;

public:
    static double process_noise_pos;
    static double process_noise_rot;
    static double observation_noise_pos;
    static double observation_noise_rot;
    static unsigned int number_of_state_dimensions;
    static unsigned int number_of_observation_dimensions;
    static unsigned int number_of_process_noise_dimensions;
    static unsigned int number_of_observation_noise_dimensions;

    Mat miut;
    Mat miut_1;
    Mat Sigt;
    Mat current_pose_estimate;
    Mat current_covariance_estimate;

    std::chrono::time_point<std::chrono::system_clock> current_time;
    std::chrono::time_point<std::chrono::system_clock> previous_time;
    FTag2Pose previous_observation;

    Kalman(){

    }

    Kalman(FTag2Pose observation) {
    	tf::Quaternion q(observation.orientation_x, observation.orientation_y, observation.orientation_z, observation.orientation_w);
    	tf::Matrix3x3 m(q);
    	double roll, pitch, yaw;
    	m.getRPY(pitch, yaw, roll);

    	new_filter = true;
    	previous_time = current_time = system_clock::now();
    	miut = Mat(number_of_state_dimensions,1,CV_64F);
    	miut.at<double>(0,0) = observation.position_x; 		// x
    	miut.at<double>(1,0) = observation.position_y;		// y
    	miut.at<double>(2,0) = observation.position_z; 		// z
    	miut.at<double>(3,0) = roll;	// roll
    	miut.at<double>(4,0) = pitch;  	// pitch
    	miut.at<double>(5,0) = yaw;  	// yaw
    	current_pose_estimate = miut.clone();
    	Mat miut_1 = miut.clone();

    	initSigt();
    	previous_observation = observation;
    }

    virtual ~Kalman() {};

    void step(){
    }


    void step(FTag2Pose observation){
    	previous_time = current_time;
    	current_time = system_clock::now();
    	std::chrono::milliseconds elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - previous_time);
    	double delta_t = elapsed.count()/1000.0;
    	Mat miut_1 = miut.clone();
    	Mat Sigt_1 = Sigt.clone();
    	std::vector<double> curr_vel = computeVelocity(observation, previous_observation, delta_t);
    	if ( new_filter == true ) {
    		miut = measurementUpdate( observation );
    		current_pose_estimate = miut.clone();
    		new_filter = false;
    		return;
    	}

    	Mat miut_ = motionUpdate(miut_1, curr_vel, delta_t );
    	Mat At = Mat::eye(number_of_state_dimensions, number_of_state_dimensions, CV_64F);
    	Mat Rt = makeRt( delta_t );
    	Mat Sigt_ = propagateCovariance( At, Sigt_1, Rt );

    	Mat zt = measurementUpdate( observation );
    	Mat Ct = makeCt( );
    	Mat Qt = makeQt( );
    	Mat Kt = computeKalmanGain( Ct, Sigt_, Qt );

    	updatePoseEstimate( miut_, Sigt_, Kt, zt, Ct );
    	current_pose_estimate = miut.clone();

    	previous_observation = observation;
    }

    Mat motionUpdate( Mat miut_1, std::vector<double> curr_vel, double delta_t ){
    	Mat miut_ = Mat(number_of_state_dimensions,1,CV_64F);
    	miut_.at<double>(0,0) = miut_1.at<double>(0,0) + curr_vel[0]*delta_t; 	// x
    	miut_.at<double>(1,0) = miut_1.at<double>(1,0) + curr_vel[1]*delta_t; 	// y
    	miut_.at<double>(2,0) = miut_1.at<double>(2,0) + curr_vel[2]*delta_t; 	// z
    	miut_.at<double>(3,0) = miut_1.at<double>(3,0);  						// roll
    	miut_.at<double>(4,0) = miut_1.at<double>(4,0);  						// pitch
    	miut_.at<double>(5,0) = miut_1.at<double>(5,0);  						// yaw
    	return miut_;
    }

    Mat makeRt( double delta_t ){
    	Mat Rt = Mat::eye(number_of_state_dimensions, number_of_state_dimensions, CV_64F);
    	Rt.at<double>(0,0) = delta_t * process_noise_pos;		// x
    	Rt.at<double>(1,1) = delta_t * process_noise_pos;		// y
    	Rt.at<double>(2,2) = delta_t * process_noise_pos;		// z
    	Rt.at<double>(3,3) = delta_t * process_noise_rot;  		// r
    	Rt.at<double>(4,4) = delta_t * process_noise_rot;  		// p
    	Rt.at<double>(5,5) = delta_t * process_noise_rot;  		// y
    	return Rt;
    }

    void initSigt(){
    	Sigt = Mat::eye(number_of_state_dimensions, number_of_state_dimensions, CV_64F);
    	Sigt.at<double>(0,0) = observation_noise_pos;		// x
    	Sigt.at<double>(1,1) = observation_noise_pos;  		// y
    	Sigt.at<double>(2,2) = observation_noise_pos;  		// z
    	Sigt.at<double>(3,3) = observation_noise_rot;		// r
    	Sigt.at<double>(4,4) = observation_noise_rot;		// p
    	Sigt.at<double>(5,5) = observation_noise_rot;		// y
    }

    Mat propagateCovariance( Mat At, Mat Sigt_1, Mat Rt ){
    	Mat At_trans = Mat(number_of_state_dimensions, number_of_state_dimensions, CV_64F);
    	cv::transpose(At,At_trans);
		Mat tmp = At * Sigt_1;
		Mat Sigt_ = tmp*At_trans + Rt;
		return Sigt_;
    }

    Mat measurementUpdate(FTag2Pose observation ){

    	tf::Quaternion q(observation.orientation_x, observation.orientation_y, observation.orientation_z, observation.orientation_w);
    	tf::Matrix3x3 m(q);
    	double roll, pitch, yaw;
    	m.getRPY(pitch, yaw, roll);

    	if ( abs(roll - miut.at<double>(3,0)) > vc_math::pi )
    	{
    		if ( roll < miut.at<double>(3,0) )
    			roll += 2*vc_math::pi;
    		else
    			miut.at<double>(3,0) += 2*vc_math::pi;
    	}

    	Mat zt = Mat(number_of_observation_dimensions, 1, CV_64F);
    	zt.at<double>(0,0) = observation.position_x;
    	zt.at<double>(1,0) = observation.position_y;
    	zt.at<double>(2,0) = observation.position_z;
    	zt.at<double>(3,0) = roll;
       	zt.at<double>(4,0) = pitch;
       	zt.at<double>(5,0) = yaw;

       	return zt;
    }

    std::vector<double> computeVelocity(FTag2Pose observation, FTag2Pose previous_observation, double delta_t){
    	std::vector<double> curr_vel(3);
    	curr_vel[0] = (observation.position_x - previous_observation.position_x)/delta_t;
    	curr_vel[1] = (observation.position_y - previous_observation.position_y)/delta_t;
    	curr_vel[2] = (observation.position_z - previous_observation.position_z)/delta_t;
    	return curr_vel;
    }

    Mat makeCt( ){
    	Mat Ct = Mat::eye(number_of_observation_dimensions, number_of_state_dimensions, CV_64F);
    	return Ct;
    }

    Mat makeQt( ){
    	Mat Qt = Mat::eye(number_of_observation_dimensions,number_of_observation_dimensions, CV_64F);
    	Qt.at<double>(0,0) = observation_noise_pos;
    	Qt.at<double>(1,1) = observation_noise_pos;
    	Qt.at<double>(2,2) = observation_noise_pos;
		Qt.at<double>(3,3) = observation_noise_rot;
		Qt.at<double>(4,4) = observation_noise_rot;
		Qt.at<double>(5,5) = observation_noise_rot;
		return Qt;
    }

    Mat computeKalmanGain( Mat Ct, Mat Sigt_, Mat Qt ){
    	Mat Ct_trans = Mat(Ct.size(), CV_64F);
    	transpose(Ct, Ct_trans);
    	Mat tmp = Ct*Sigt_*Ct_trans + Qt;
    	Mat tmp_inv = tmp.inv();
    	Mat Kt = Sigt_*Ct_trans*tmp_inv;
    	return Kt;
    }

    void updatePoseEstimate(Mat miut_, Mat Sigt_, Mat Kt, Mat zt, Mat Ct){
    	miut = miut_ + Kt * ( zt - miut_ ); // or miut - zt?
    	miut.at<double>(3,0) = miut.at<double>(3,0) - (2*vc_math::pi) * floor( miut.at<double>(3,0) / (2.0*vc_math::pi) );
    	Mat tmp = Mat::eye(number_of_state_dimensions,number_of_state_dimensions, CV_64F);
      	Sigt = tmp*(Kt*Ct)*Sigt_;
    }

	FTag2Pose getEstimatedPose() {
		FTag2Pose pose;
		tf::Quaternion q;
		q = tf::createQuaternionFromRPY(current_pose_estimate.at<double>(4,0),
				current_pose_estimate.at<double>(5,0), current_pose_estimate.at<double>(3,0));

		pose.position_x = current_pose_estimate.at<double>(0,0);
		pose.position_y = current_pose_estimate.at<double>(1,0);
		pose.position_z = current_pose_estimate.at<double>(2,0);
		pose.orientation_w = q.w();
		pose.orientation_x = q.x();
		pose.orientation_y = q.y();
		pose.orientation_z = q.z();
		return pose;
	}

};

#endif /* Kalman_H_ */
