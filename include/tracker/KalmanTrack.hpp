/*
 * ObjectHypothesis.h
 *
 *  Created on: 2013-10-23
 *      Author: dacocp
 */

#ifndef KALMANTRACK_H_
#define KALMANTRACK_H_

#include "kalman/ekfilter.hpp"
#include "kalman/kmatrix.hpp"
#include "kalman/kvector.hpp"

#include "common/FTag2.hpp"

#include <ompl/base/spaces/SO3StateSpace.h>
#include <ompl/base/State.h>
#include <ompl/base/ScopedState.h>

#include <random>
#include <iostream>
#include <chrono>
#include <ctime>

#include <cstdlib>
#include <iostream>
#include <fstream>

#include <cmath>



#define MS_PER_FRAME 33.0

using namespace std;

class KalmanTrack: public Kalman::EKFilter<double,1> {

private:

protected:
    double Period;
    Vector previous_state;
    FTag2Pose current_observation;

public:
    static double process_noise_pos;
    static double process_noise_vel;
    static double process_noise_rot;
    static double observation_noise_pos;
    static double observation_noise_rot;
    static unsigned int number_of_state_dimensions;
    static unsigned int number_of_observation_dimensions;
    static unsigned int number_of_process_noise_dimensions;
    static unsigned int number_of_observation_noise_dimensions;

    void printMatrix(const Matrix& M)
    {
    	for (unsigned int i=1; i<=M.nrow(); i++)
    	{
    		for (unsigned int j=1; j<=M.ncol(); j++)
    			std::cout << M(i,j) << ", ";
    		std::cout<< std::endl;
    	}
    	std::cout<< std::endl;
    }

    void makeBaseA()
    {
    	for (unsigned int i=1; i<=number_of_state_dimensions; i++)
    	{
    		for (unsigned int j=1; j<=number_of_state_dimensions; j++)
    		{
    			if ( i == j )
    				A(i,j) = 1.0;
    			A(i,j) = 0.0;
    		}
    	}
    }

    void makeA()
    {
//    	cout << "Making A..." << endl;
    	/* ================== */
    	A(1,1) = 1.0;	A(1,2) = Period;
    	A(2,1) = 0.0;	A(2,2) = 1.0;
    	/* ================== */

    									/* ================== */
    									A(3,3) = 1.0;	A(3,4) = Period;
    									A(4,3) = 0.0;	A(4,4) = 1.0;
    									/* ================== */

    																	/* ================== */
    																	A(5,5) = 1.0; 	A(5,6) = Period;
    																	A(6,5) = 0.0;	A(6,6) = 1.0;
    																		/* ================== */

    	/* ================== */ /* ================== */ /* ================== */
    	A(7,7) = 1.0;
    					A(8,8) = 1.0;
    									A(9,9) = 1.0;
    													A(10,10) = 1.0;
    	/* ================== */ /* ================== */ /* ================== */

//    	std::cout<< std::endl;
//    	for (unsigned int i=1; i<=number_of_state_dimensions; i++)
//    	{
//    		for (unsigned int j=1; j<=number_of_state_dimensions; j++)
//    			std::cout << A(i,j) << ", ";
//    		std::cout<< std::endl;
//    	}
    }

    void makeW()
    {
//    	cout << "Making W..." << endl;
    	for (unsigned int i=1; i<=number_of_state_dimensions; i++)
    		for (unsigned int j=1; j<=number_of_process_noise_dimensions; j++)
    		{
    			if ( i == j )
    				W(i,j) = 1.0;
    			else
    				W(i,j) = 0.0;
    		}
//    	printMatrix(W);
    }

    void makeQ()
    {
//    	cout << "Making Q..." << endl;
    	for (unsigned int i=1; i<=number_of_process_noise_dimensions; i++)
    		for (unsigned int j=1; j<=number_of_process_noise_dimensions; j++)
    		{
    			if( i == j )
    				Q(i,j) = 1.0;
    			else
    				Q(i,j) = 0.0;
    		}
    	Q(1,1) = process_noise_pos;		// x
    	Q(2,2) = process_noise_vel;		// x'
    	Q(3,3) = process_noise_pos;		// y
    	Q(4,4) = process_noise_vel;		// y'
    	Q(5,5) = process_noise_pos;		// z
    	Q(6,6) = process_noise_rot;		// z'
    	Q(7,7) = process_noise_rot;		// rw
    	Q(8,8) = process_noise_rot;		// rx
    	Q(9,9) = process_noise_rot;		// ry
    	Q(10,10) = process_noise_rot;	// rz

//    	printMatrix(Q);
    }

//    void makeBaseH()
//    {
//    	for (unsigned int i=1; i<=number_of_observation_dimensions; i++)
//    		for (unsigned int j=1; j<=number_of_state_dimensions; j++)
//    			H(i,j) = 0.0;
//    }

    void makeH()
    {
//    	cout << "Making H..." << endl;
    	for (unsigned int i=1; i<=number_of_observation_dimensions; i++)
    		for (unsigned int j=1; j<=number_of_state_dimensions; j++)
    			H(i,j) = 0.0;

    	H(1,1) = 1.0;
    	H(2,3) = 1.0;
    	H(3,5) = 1.0;
    	H(4,7) = 1.0;
    	H(5,8) = 1.0;
    	H(6,9) = 1.0;
    	H(7,10) = 1.0;
//    	printMatrix(H);
    }

    void makeV()
    {
//    	cout << "Making V..." << endl;
    	for (unsigned int i=1; i<=number_of_observation_dimensions; i++)
    		for (unsigned int j=1; j<=number_of_observation_noise_dimensions; j++)
    		{
    			if( i == j )
    				V(i,j) = 1.0;
    			else
    				V(i,j) = 0.0;
    		}
//    	printMatrix(V);
    }

    void makeR()
    {
    	for (unsigned int i=1; i<=number_of_observation_noise_dimensions; i++)
    		for (unsigned int j=1; j<=number_of_observation_noise_dimensions; j++)
    		{
    			if( i == j )
    				R(i,j) = 1.0;
    			else
    				R(i,j) = 0.0;
    		}
    	for ( unsigned int i=1; i <= 3; i++ )
    		R(i,i) = observation_noise_pos;
    	for ( unsigned int i=4; i <= 7; i++ )
    		R(i,i) = observation_noise_rot;
//    	printMatrix(R);
    }

    void makeProcess()
    {
//    	previous_state = Vector(x.size());
//    	for (unsigned int i=1; i < previous_state.size(); i++)
//    		previous_state(i) = x(i);
//    	cout << "Making process: "  << endl;
    	Vector x_(number_of_state_dimensions);
    	x_(1) = x(1) + x(2)*Period;
    	x_(2) = 0.0;  //x_(1) - x(1);
    	x_(3) = x(3) + x(4)*Period;
    	x_(4) = 0.0;  // x_(3) - x(3);
    	x_(5) = x(5) + x(6)*Period;
    	x_(6) = 0.0;  // x_(5) - x(5);

    	x_(7) = x(7);
    	x_(8) = x(8);
    	x_(9) = x(9);
    	x_(10) = x(10);

    	x.swap(x_);
//    	cout << "Finished process" << endl;
    }

    void makeMeasure()
    {
//    	cout << "Making measure: "  << endl;
    	z(1) = x(1);
    	z(2) = x(3);
    	z(3) = x(5);
    	z(4) = x(7);
    	z(5) = x(8);
    	z(6) = x(9);
    	z(7) = x(10);
//    	cout << "Finished measure" << endl;
    }

    KalmanTrack(){};
	KalmanTrack(FTag2Pose observation) {
//		The function setDim() sets
//		the number of states,
//		the number of inputs,
//		the number of process noise random variables,
//		the number of measures and
//		the number of measurement noise random variables.
		int number_of_inputs = 0;
        setDim(number_of_state_dimensions, number_of_inputs, number_of_process_noise_dimensions, number_of_observation_dimensions, number_of_observation_noise_dimensions);

        Period = 1;

#define ICP 0.1
#define ICV 0.5
#define ICR 0.1


		static const double _P0[] = {
			/*	x		x'		y		y'		z		z'		rw		rx		ry		rz	*/
				ICP, 	0.0, 	0.0, 	0.0, 	0.0, 	0.0, 	0.0, 	0.0, 	0.0, 	0.0, // x
				0.0, 	ICV, 	0.0, 	0.0, 	0.0, 	0.0, 	0.0, 	0.0, 	0.0, 	0.0, // x'
				0.0, 	0.0, 	ICP, 	0.0, 	0.0, 	0.0, 	0.0, 	0.0, 	0.0, 	0.0, // y
				0.0, 	0.0, 	0.0, 	ICV, 	0.0, 	0.0, 	0.0, 	0.0, 	0.0, 	0.0, // y'
				0.0, 	0.0, 	0.0, 	0.0, 	ICP, 	0.0, 	0.0, 	0.0, 	0.0, 	0.0, // z
				0.0, 	0.0, 	0.0, 	0.0, 	0.0, 	ICV, 	0.0, 	0.0, 	0.0, 	0.0, // z'
				0.0, 	0.0, 	0.0, 	0.0, 	0.0, 	0.0, 	ICR, 	0.0, 	0.0, 	0.0, // rw
				0.0, 	0.0, 	0.0, 	0.0,	0.0, 	0.0, 	0.0, 	ICR, 	0.0, 	0.0, // rx
				0.0, 	0.0, 	0.0, 	0.0, 	0.0, 	0.0, 	0.0, 	0.0, 	ICR, 	0.0, // ry
				0.0, 	0.0, 	0.0, 	0.0,	0.0, 	0.0, 	0.0, 	0.0, 	0.0, 	ICR, // rz
		};
		Matrix P0(number_of_state_dimensions, number_of_state_dimensions, _P0);

//		cout << "Initial P = " << endl;
//		printMatrix(P0);

		Vector x_(number_of_state_dimensions);

		//Initiale estimate
		x_(1) = observation.position_x;
		x_(2) = 0.0;
		x_(3) = observation.position_y;
		x_(4) = 0.0;
		x_(5) = observation.position_z;
		x_(6) = 0.0;
		x_(7) = observation.orientation_w;
		x_(8) = observation.orientation_x;
		x_(9) = observation.orientation_y;
		x_(10) = observation.orientation_z;

		init(x_, P0);
	};

	void step_(FTag2Pose observation) {
//		cout << "Stepping KF" << endl;
		current_observation = observation;

		Vector z_(number_of_observation_dimensions);


		z_(1) = current_observation.position_x;
		z_(2) = current_observation.position_y;
		z_(3) = current_observation.position_z;
		z_(4) = current_observation.orientation_w;
		z_(5) = current_observation.orientation_x;
		z_(6) = current_observation.orientation_y;
		z_(7) = current_observation.orientation_z;

		Vector u_(0);
		step(u_, z_);

//		cout << "Observation:\t";
//		for ( unsigned int i = 1; i <= z_.size(); i++)
//		{
//			std::cout << z_(i) << ", ";
//		}
//		std::cout << endl << "Estimate:   ";
//		for ( unsigned int i = 1; i < x.size(); i++)
//		{
//			std::cout << x(i) << ", ";
//		}
//		cout << endl;

	}

	void step_() {
		Vector u_(0);
		timeUpdateStep(u_);
	}

	FTag2Pose getEstimatedPose() {
		FTag2Pose pose;
//		std::cout << endl << "Estimate:\t";
//		for ( unsigned int i = 1; i <= x.size(); i++)
//		{
//			std::cout << x(i) << ", ";
//		}
//		cout << endl;

		pose.position_x = x(1);
		pose.position_y = x(3);
		pose.position_z = x(5);
		pose.orientation_w = x(7);
		pose.orientation_x = x(8);
		pose.orientation_y = x(9);
		pose.orientation_z = x(10);

//		cout << "P = " << endl;
//		printMatrix(calculateP());
		return pose;
	}
//	virtual ~KalmanTrack();
};

typedef KalmanTrack::Vector Vector;
typedef KalmanTrack::Matrix Matrix;

#endif /* KALMANTRACK_H_ */
