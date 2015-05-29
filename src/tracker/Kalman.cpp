#include "tracker/Kalman.hpp"

double Kalman::process_noise_pos = 0.1;
double Kalman::process_noise_rot = 0.05	;
double Kalman::observation_noise_pos = 0.05;
double Kalman::observation_noise_rot = 0.01;
unsigned int Kalman::number_of_state_dimensions = 6;
unsigned int Kalman::number_of_observation_dimensions = 6;
unsigned int Kalman::number_of_process_noise_dimensions = 6;
unsigned int Kalman::number_of_observation_noise_dimensions = 6;
