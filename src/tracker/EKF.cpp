#include "tracker/EKF.hpp"

double EKF::process_noise_pos = 0.25;
double EKF::process_noise_vel = 5.0;
double EKF::process_noise_rot = 0.1;
double EKF::observation_noise_pos = 0.05;
double EKF::observation_noise_vel = 0.05;
double EKF::observation_noise_rot = 0.05;
unsigned int EKF::number_of_state_dimensions = 9;
unsigned int EKF::number_of_observation_dimensions = 9;
unsigned int EKF::number_of_process_noise_dimensions = 9;
unsigned int EKF::number_of_observation_noise_dimensions = 9;


//double EKF::process_noise_pos = 0.1;
//double EKF::process_noise_vel = 1.0;
//double EKF::process_noise_rot = 0.1;
//double EKF::observation_noise_pos = 0.1;
//double EKF::observation_noise_vel = 1.0;
//double EKF::observation_noise_rot = 0.1;
//unsigned int EKF::number_of_state_dimensions = 10;
//unsigned int EKF::number_of_observation_dimensions = 10;
//unsigned int EKF::number_of_process_noise_dimensions = 10;
//unsigned int EKF::number_of_observation_noise_dimensions = 10;
