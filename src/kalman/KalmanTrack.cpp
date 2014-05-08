/*
 * ObjectHypothesis.h
 *
 *  Created on: 2013-10-23
 *      Author: dacocp
 */


#include "tracker/KalmanTrack.hpp"

double KalmanTrack::process_noise_pos = 0.1;
double KalmanTrack::process_noise_vel = 1.0;
double KalmanTrack::process_noise_rot = 0.1;
double KalmanTrack::observation_noise_pos = 0.1;
double KalmanTrack::observation_noise_vel = 1.0;
double KalmanTrack::observation_noise_rot = 0.1;
unsigned int KalmanTrack::number_of_state_dimensions = 10;
unsigned int KalmanTrack::number_of_observation_dimensions = 10;
unsigned int KalmanTrack::number_of_process_noise_dimensions = 10;
unsigned int KalmanTrack::number_of_observation_noise_dimensions = 10;

//// Simple uniform distribution of zero mean and unit variance
//float uniformxx(void)
//{
//   return((((float)rand())/(RAND_MAX-1) - 0.5f)* 3.464101615138f);
//}
//
//// Simple approximation of normal dist. by the sum of uniform dist.
//float normalxx()
//{
//  int n = 6;
//  int i;
//  float temp = 0.0;
//
//  for(i = 0; i < n; i++)
//    temp += uniformxx();
//  temp /= sqrt((float)n);
//  return temp;
//}
