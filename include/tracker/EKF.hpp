/*
 * ObjectHypothesis.h
 *
 *  Created on: 2013-10-23
 *      Author: dacocp
 */

#ifndef EKF_H_
#define EKF_H_

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

class EKF {

private:

protected:
    FTag2Pose current_observation;
    bool new_filter;

public:
    static double process_noise_pos;
    static double process_noise_vel;
    static double process_noise_rot;
    static double observation_noise_pos;
    static double observation_noise_vel;
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

    EKF(){

    }

    EKF(FTag2Pose observation) {
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
    	miut.at<double>(3,0) = 0;  							// \dot{x}
    	miut.at<double>(4,0) = 0;							// \dot{y}
    	miut.at<double>(5,0) = 0;  							// \dot{z}
    	miut.at<double>(6,0) = roll;	// rot_w
    	miut.at<double>(7,0) = pitch;  	// rot_x
    	miut.at<double>(8,0) = yaw;  	// rot_y
    	current_pose_estimate = miut.clone();
    	Mat miut_1 = miut.clone();

//    	new_filter = true;
//    	previous_time = current_time = system_clock::now();
//    	miut = Mat(number_of_state_dimensions,1,CV_64F);
//    	miut.at<double>(0,0) = observation.position_x; 		// x
//    	miut.at<double>(1,0) = observation.position_y;		// y
//    	miut.at<double>(2,0) = observation.position_z; 		// z
//    	miut.at<double>(3,0) = 0;  							// \dot{x}
//    	miut.at<double>(4,0) = 0;							// \dot{y}
//    	miut.at<double>(5,0) = 0;  							// \dot{z}
//    	miut.at<double>(6,0) = observation.orientation_w;	// rot_w
//    	miut.at<double>(7,0) = observation.orientation_x;  	// rot_x
//    	miut.at<double>(8,0) = observation.orientation_y;  	// rot_y
//    	miut.at<double>(9,0) = observation.orientation_z;	// rot_z
//    	current_pose_estimate = miut.clone();

//    	cout << "Constructor: " << current_pose_estimate << endl;

    	initSigt();
//    	Kt = Mat(number_of_state_dimensions, number_of_observation_dimensions, CV_64F);

    	previous_observation = observation;
    }

    virtual ~EKF() {};

    void step(){
    	if ( new_filter == true )
    		return;
    	std::chrono::duration<double> elapsed = system_clock::now() - previous_time;
    	double delta_t = elapsed.count();

    	current_pose_estimate = motionUpdate(miut.clone(),delta_t);
    }

    void step(FTag2Pose observation){
//    	cout << "******************** 111" << endl;
    	previous_time = current_time;
    	current_time = system_clock::now();
    	std::chrono::duration<double> elapsed = current_time - previous_time;
    	double delta_t = elapsed.count();
//    	cout << "*******************DELTA T = " << delta_t << endl;
    	Mat miut_1 = miut.clone();
//    	cout << "Previous pose miut_1: " << miut_1 << endl;
    	Mat Sigt_1 = Sigt.clone();
//    	cout << "******************** 222" << endl;

    	std::vector<double> curr_vel = computeVelocity(observation, previous_observation, delta_t);
    	if ( new_filter == true ) {
//    		cout << "New filter before: " << current_pose_estimate << endl;
    		miut = measurementUpdate( observation, curr_vel, delta_t );
//    		cout << "******************** 222.111" << endl;
    		current_pose_estimate = miut.clone();
//        	cout << "New filter after: " << current_pose_estimate << endl;
    		new_filter = false;
//    		cout << "******************** 222.222" << endl;
    		return;
    	}

//    	cout << "******************** 333" << endl;
//    	cout << "miut_1 = " << miut_1 << endl;
//    	cout << "Sigt_1 = " << Sigt_1 << endl;
    	Mat miut_ = motionUpdate(miut_1, delta_t );
//    	cout << "******************** 444" << endl;
    	Mat Gt = makeGt( delta_t );
    	Mat Rt = makeRt( delta_t );
    	Mat Sigt_ = propagateCovariance( Gt, Sigt_1, Rt );
//    	cout << "******************** 555" << endl;
//    	cout << "miut_ = " << miut_ << endl;
//    	cout << "Sigt_ = " << Sigt_ << endl;
//    	cout << "333" << endl;

    	Mat zt = measurementUpdate( observation, curr_vel, delta_t );
//    	cout << "zt = " << zt << endl;
    	Mat Ht = makeHt( delta_t);
    	Mat Qt = makeQt( delta_t );
    	Mat Kt = computeKalmanGain( Ht, Sigt_, Qt );
    	updatePoseEstimate( miut_, Sigt_, Kt, zt, Ht );
//    	cout << "miut = " << miut << endl;
//    	cout << "Sigt = " << Sigt << endl;
//    	cout << "******************** 999" << endl;
    	current_pose_estimate = miut.clone();

    	previous_observation = observation;
    }

    std::vector<double> computeVelocity(FTag2Pose observation, FTag2Pose previous_observation, double delta_t){
    	std::vector<double> curr_vel(3);
    	curr_vel[0] = (observation.position_x - previous_observation.position_x)/delta_t;
    	curr_vel[1] = (observation.position_y - previous_observation.position_y)/delta_t;
    	curr_vel[2] = (observation.position_z - previous_observation.position_z)/delta_t;
    	return curr_vel;
    }

    void firstObservation( FTag2Pose observation, std::vector<double> curr_vel, Mat miut_1, double delta_t  )
    {
    	tf::Quaternion q(observation.orientation_x, observation.orientation_y, observation.orientation_z, observation.orientation_w);
    	tf::Matrix3x3 m(q);
    	double roll, pitch, yaw;
    	m.getRPY(pitch, yaw, roll);

    	if ( abs(roll - miut_1.at<double>(6,0)) > vc_math::pi )
    	{
    		if ( roll < miut_1.at<double>(6,0) )
    			roll += 2*vc_math::pi;
    		else
    			miut_1.at<double>(6,0) += 2*vc_math::pi;
    	}

    	miut.at<double>(0,0) = observation.position_x;
    	miut.at<double>(1,0) = observation.position_y;
    	miut.at<double>(2,0) = observation.position_z;
    	miut.at<double>(3,0) = curr_vel[0];   // (observation.position_x - miut_1.at<double>(0,0))/delta_t;
    	miut.at<double>(4,0) = curr_vel[1];   // (observation.position_y - miut_1.at<double>(1,0))/delta_t;
    	miut.at<double>(5,0) = curr_vel[2];   // (observation.position_z - miut_1.at<double>(2,0))/delta_t;
    	miut.at<double>(6,0) = roll;
    	miut.at<double>(7,0) = pitch;
    	miut.at<double>(8,0) = yaw;

//    	miut.at<double>(0,0) = observation.position_x;
//    	miut.at<double>(1,0) = observation.position_y;
//    	miut.at<double>(2,0) = observation.position_z;
//    	miut.at<double>(3,0) = (observation.position_x - miut_1.at<double>(0,0))/delta_t;
//    	miut.at<double>(4,0) = (observation.position_y - miut_1.at<double>(1,0))/delta_t;
//    	miut.at<double>(5,0) = (observation.position_z - miut_1.at<double>(2,0))/delta_t;
//    	miut.at<double>(6,0) = observation.orientation_w;
//    	miut.at<double>(7,0) = observation.orientation_x;
//    	miut.at<double>(8,0) = observation.orientation_y;
//    	miut.at<double>(9,0) = observation.orientation_z;
    }

    Mat motionUpdate( Mat miut_1, double delta_t ){
    	Mat miut_ = Mat(number_of_state_dimensions,1,CV_64F);
    	miut_.at<double>(0,0) = miut_1.at<double>(0,0) + miut_1.at<double>(3,0)*delta_t; 	// x
    	miut_.at<double>(1,0) = miut_1.at<double>(1,0) + miut_1.at<double>(4,0)*delta_t; 	// y
    	miut_.at<double>(2,0) = miut_1.at<double>(2,0) + miut_1.at<double>(5,0)*delta_t; 	// z
    	miut_.at<double>(3,0) = miut_1.at<double>(3,0);  							// \dot{x}
    	miut_.at<double>(4,0) = miut_1.at<double>(4,0);							 	// \dot{y}
    	miut_.at<double>(5,0) = miut_1.at<double>(5,0);  							// \dot{z}
    	miut_.at<double>(6,0) = miut_1.at<double>(6,0);  							// rot_w
    	miut_.at<double>(7,0) = miut_1.at<double>(7,0);  							// rot_x
    	miut_.at<double>(8,0) = miut_1.at<double>(8,0);  							// rot_y
//    	miut_.at<double>(9,0) = miut_1.at<double>(9,0);  							// rot_z
//    	cout << "Finished process" << endl;
    	return miut_;
    }

    Mat makeGt(double delta_t){
    	Mat Gt = Mat::eye(number_of_state_dimensions, number_of_state_dimensions, CV_64F);
    	Gt.at<double>(0,3) = delta_t; // derivative of x wrt \dot{x}
    	Gt.at<double>(1,4) = delta_t; // derivative of y wrt \dot{y}
    	Gt.at<double>(2,5) = delta_t; // derivative of z wrt \dot{z}
    	return Gt;
    }

    Mat makeRt( double delta_t ){
    	Mat Rt = Mat::eye(number_of_state_dimensions, number_of_state_dimensions, CV_64F);
    	Rt.at<double>(0,0) = delta_t * process_noise_pos;		// x
    	Rt.at<double>(1,1) = delta_t * process_noise_pos;		// y
    	Rt.at<double>(2,2) = delta_t * process_noise_pos;		// z
    	Rt.at<double>(3,3) = delta_t * process_noise_vel;  		// x'
    	Rt.at<double>(4,4) = delta_t * process_noise_vel;  		// y'
    	Rt.at<double>(5,5) = delta_t * process_noise_vel;  		// z'
    	Rt.at<double>(6,6) = delta_t * process_noise_rot;  		// rw
    	Rt.at<double>(7,7) = delta_t * process_noise_rot;  		// rx
    	Rt.at<double>(8,8) = delta_t * process_noise_rot;  		// ry
//    	Rt.at<double>(9,9) = delta_t * process_noise_rot;  		// rz
    	return Rt;
    }

    void initSigt(){
    	Sigt = Mat::eye(number_of_state_dimensions, number_of_state_dimensions, CV_64F);
    	Sigt.at<double>(0,0) = 4.0;  //  process_noise_pos;		// x
    	Sigt.at<double>(1,1) = 4.0;  //  process_noise_pos;		// y
    	Sigt.at<double>(2,2) = 4.0;  //  process_noise_pos;		// z
    	Sigt.at<double>(3,3) = 0;  //  process_noise_vel;		// x'
    	Sigt.at<double>(4,4) = 0;  //  process_noise_vel;		// y'
    	Sigt.at<double>(5,5) = 0;  //  process_noise_vel;		// z'
    	Sigt.at<double>(6,6) = 0.5;  //  process_noise_rot;		// rw
    	Sigt.at<double>(7,7) = 0.5;  //  process_noise_rot;		// rx
    	Sigt.at<double>(8,8) = 0.5;  //  process_noise_rot;		// ry
//    	Sigt.at<double>(9,9) = 0.5;  //  process_noise_rot;		// rz
    }

    Mat propagateCovariance( Mat Gt, Mat Sigt_1, Mat Rt ){
    	Mat Gt_trans = Mat(number_of_state_dimensions, number_of_state_dimensions, CV_64F);
    	cv::transpose(Gt,Gt_trans);
		Mat tmp = Gt * Sigt_1;
		Mat Sigt_ = tmp*Gt_trans + Rt;
		return Sigt_;
    }

    Mat measurementUpdate(FTag2Pose observation, std::vector<double> curr_vel, double delta_t ){
    	tf::Quaternion q(observation.orientation_x, observation.orientation_y, observation.orientation_z, observation.orientation_w);
    	tf::Matrix3x3 m(q);
    	double roll, pitch, yaw;
    	m.getRPY(pitch, yaw, roll);

    	if ( abs(roll - miut.at<double>(6,0)) > vc_math::pi )
    	{
    		if ( roll < miut.at<double>(6,0) )
    			roll += 2*vc_math::pi;
    		else
    			miut.at<double>(6,0) += 2*vc_math::pi;
    	}

    	Mat zt = Mat(number_of_observation_dimensions, 1, CV_64F);
    	zt.at<double>(0,0) = observation.position_x;
    	zt.at<double>(1,0) = observation.position_y;
    	zt.at<double>(2,0) = observation.position_z;
    	zt.at<double>(3,0) = curr_vel[0];   //(observation.position_x - miut.at<double>(0,0))/delta_t;
    	zt.at<double>(4,0) = curr_vel[1];   //(observation.position_y - miut.at<double>(1,0))/delta_t;
    	zt.at<double>(5,0) = curr_vel[2];   //(observation.position_z - miut.at<double>(2,0))/delta_t;
    	zt.at<double>(6,0) = roll;
       	zt.at<double>(7,0) = pitch;
       	zt.at<double>(8,0) = yaw;
	    return zt;

//    	Mat zt = Mat(number_of_observation_dimensions, 1, CV_64F);
//    	zt.at<double>(0,0) = observation.position_x;
//    	zt.at<double>(1,0) = observation.position_y;
//    	zt.at<double>(2,0) = observation.position_z;
//    	zt.at<double>(3,0) = (observation.position_x - last_pose_estimate.at<double>(0,0))/delta_t;
//    	zt.at<double>(4,0) = (observation.position_y - last_pose_estimate.at<double>(1,0))/delta_t;
//    	zt.at<double>(5,0) = (observation.position_z - last_pose_estimate.at<double>(2,0))/delta_t;
//    	zt.at<double>(6,0) = observation.orientation_w;
//       	zt.at<double>(7,0) = observation.orientation_x;
//       	zt.at<double>(8,0) = observation.orientation_y;
//	    zt.at<double>(9,0) = observation.orientation_z;
//	    return zt;
    }

//    Mat makeHt( double delta_t ){
//    	Mat Ht = Mat::zeros(number_of_observation_dimensions, number_of_state_dimensions, CV_64F);
//    	Ht.at<double>(0,0) = -1.0/delta_t;
//    	Ht.at<double>(1,1) = -1.0/delta_t;
//    	Ht.at<double>(2,2) = -1.0/delta_t;
//    	return Ht;
//    }
    Mat makeHt( double delta_t ){
    	Mat Ht = Mat::eye(number_of_observation_dimensions, number_of_state_dimensions, CV_64F);
    	Ht.at<double>(3,0) = 1.0/delta_t;
//    	Ht.at<double>(3,3) = -1.0/delta_t;
    	Ht.at<double>(4,1) = 1.0/delta_t;
//    	Ht.at<double>(4,4) = -1.0/delta_t;
    	Ht.at<double>(5,2) = 1.0/delta_t;
//    	Ht.at<double>(5,5) = -1.0/delta_t;;
    	return Ht;
    }

    Mat makeQt( double delta_t ){
    	Mat Qt = Mat::eye(number_of_observation_dimensions,number_of_observation_dimensions, CV_64F);
    	Qt.at<double>(0,0) = observation_noise_pos;  //  *delta_t;
    	Qt.at<double>(1,1) = observation_noise_pos;  //  delta_t;
    	Qt.at<double>(2,2) = observation_noise_pos;  //  *delta_t;
		Qt.at<double>(3,3) = observation_noise_vel;  //  *delta_t;
		Qt.at<double>(4,4) = observation_noise_vel;  //  *delta_t;
		Qt.at<double>(5,5) = observation_noise_vel;  //  *delta_t;
		Qt.at<double>(6,6) = observation_noise_rot;  //  *delta_t;
		Qt.at<double>(7,7) = observation_noise_rot;  //  *delta_t;
		Qt.at<double>(8,8) = observation_noise_rot;  //  *delta_t;
//		Qt.at<double>(9,9) = observation_noise_rot;
		return Qt;
    }

    Mat computeKalmanGain( Mat Ht, Mat Sigt_, Mat Qt ){
    	Mat Ht_trans = Mat(Ht.size(), CV_64F);
    	transpose(Ht, Ht_trans);
    	Mat tmp = Ht*Sigt_*Ht_trans + Qt;
    	Mat tmp_inv = tmp.inv();
    	Mat Kt = Sigt_*Ht_trans*tmp_inv;
    	return Kt;
    }

    void updatePoseEstimate(Mat miut_, Mat Sigt_, Mat Kt, Mat zt, Mat Ht){
    	miut = miut_ + Kt * ( zt - miut_ ); // or miut - zt?
    	miut.at<double>(6,0) = miut.at<double>(6,0) - (2*vc_math::pi) * floor( miut.at<double>(6,0) / (2.0*vc_math::pi) );
    	Mat tmp = Mat::eye(number_of_state_dimensions,number_of_state_dimensions, CV_64F);
      	Sigt = tmp*(Kt*Ht)*Sigt_;
    }

	FTag2Pose getEstimatedPose() {
//		cout << "Filter returns: " << current_pose_estimate << endl;
//		cout << "Roll: " << current_pose_estimate.at<double>(6,0) << endl;
//		cout << "Pitch: " << current_pose_estimate.at<double>(7,0) << endl;
//		cout << "Yaw: " << current_pose_estimate.at<double>(8,0) << endl;

		FTag2Pose pose;
		tf::Quaternion q;
		q = tf::createQuaternionFromRPY(current_pose_estimate.at<double>(7,0), current_pose_estimate.at<double>(8,0), current_pose_estimate.at<double>(6,0));

		pose.position_x = current_pose_estimate.at<double>(0,0);
		pose.position_y = current_pose_estimate.at<double>(1,0);
		pose.position_z = current_pose_estimate.at<double>(2,0);
		pose.orientation_w = q.w();
		pose.orientation_x = q.x();
		pose.orientation_y = q.y();
		pose.orientation_z = q.z();

//		std::cout << "Pose returned by filter :\n" << pose.position_x << ", " << pose.position_y
//												<< ", " << pose.position_z << std::endl;
//		printMatrix(calculateP());
		return pose;
	}

};

#endif /* EKF_H_ */
