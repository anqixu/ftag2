/*
 * ParticleFilter.cpp
 *
 *  Created on: 2013-10-23
 *      Author: dacocp
 */

#include "tracker/ParticleFilter.hpp"

#define SILENT_PF

double ParticleFilter::sampling_percent = 0.9;

unsigned int ParticleFilter::number_of_particles = 1000;
double ParticleFilter::position_std = 0.2;
double ParticleFilter::orientation_std = 0.2;
double ParticleFilter::position_noise_std = 0.2;
double ParticleFilter::orientation_noise_std = 0.2;
double ParticleFilter::velocity_noise_std = 0.2;
double ParticleFilter::acceleration_noise_std = 0.2;

ParticleFilter::~ParticleFilter() {
	// TODO Auto-generated destructor stub
}

/*
// TODO only gcc 4.7 supports delegating constructors; need to manually initialize (i.e. write initialize function, then have different constructors call it)
ParticleFilter::ParticleFilter(int numP, std::vector<FTag2Pose> observations,
		ParticleFilter::time_point starting_time_):
				ParticleFilter(numP,observations,0.15,0.15,0.15,0.15,0.01,0.01, starting_time){
}
*/

ParticleFilter::ParticleFilter():
				disable_resampling(false),
				log_sum_of_weights(0.0), log_max_weight(0.0),
				current_time_step_ms(0) { };


ParticleFilter::ParticleFilter(FTag2Pose observation) :
				log_sum_of_weights(0.0), log_max_weight(0.0),
				current_time_step_ms(0){

	std::chrono::duration<int,std::milli> start_delay(50);

	std::chrono::milliseconds ms_(100);
	//unsigned long long ms = ms_.count();

	starting_time = ParticleFilter::clock::now() - std::chrono::milliseconds(100);
	current_time = starting_time;

//	std::chrono::milliseconds st_ = std::chrono::duration_cast<std::chrono::milliseconds>(starting_time - starting_time_);

//	std::cout << "Creating PF (NP = " << number_of_particles << ")" << std::endl;

	weights.resize(number_of_particles);
	particles.resize(number_of_particles);
	srand(time(NULL));
//	std::cout << "Particles: " << std::endl;
	for ( unsigned int i=0; i < number_of_particles; i++ )
	{
		weights[i] = 1.0/number_of_particles;
		particles[i] = ObjectHypothesis(observation, true);
		//std::cout <<  "Pose_x: " << observations[k].position_x << std::endl;
		//std::cout <<  "Pose_y: " << observations[k].position_y << std::endl;
		//std::cout <<  "Pose_z: " << observations[k].position_z << std::endl;
		//std::cout <<  "Part Pose_x: " << particles[i].getPose().position_x << std::endl;
		//std::cout <<  "Part Pose_y: " << particles[i].getPose().position_y << std::endl;
		//std::cout <<  "Part Pose_z: " << particles[i].getPose().position_z << std::endl;
	}
//	std::cout << "Cloud created" << std::endl;
	//cv::waitKey();

	disable_resampling = false;
}

void ParticleFilter::updateParameters(int numP, double position_std_, double orientation_std_,
		double position_noise_std_, double orientation_noise_std_, double velocity_noise_std_, double acceleration_noise_std_ ){
//	std::lock_guard<std::mutex> lock(paramsMutex);
	number_of_particles = numP;
	position_std = position_std_;
	orientation_std = orientation_std_;
	position_noise_std = position_noise_std_;
	orientation_noise_std = orientation_noise_std_;
	velocity_noise_std = velocity_noise_std_;
	acceleration_noise_std = acceleration_noise_std_;
//	paramsMut.unlock();
#ifndef SILENT_PF
	std::cout << "Params: " << std::endl << "Num. paritlces: " << number_of_particles << std::endl;
	std::cout << "Position STD: " << position_std << std::endl;
	std::cout << "Orientation STD: " << orientation_std << std::endl;
	std::cout << "Position noise STD: " << position_noise_std << std::endl;
	std::cout << "Orientation noise STD: " << orientation_noise_std << std::endl;
#endif
}

void ParticleFilter::step(FTag2Pose observation)
{
//	cout << "Steping PF with obs." << endl;
	new_observations.clear();
	new_observations.push_back(observation);
	step();
}

void ParticleFilter::step()
{
//	cout << "Steping PF without obs." << endl;
//	paramsMutex.lock();
	motionUpdate();
	measurementUpdate(new_observations);
	normalizeWeights();
	computeMeanPose();
	resample();

	/* Clear processed observations */
	new_observations.clear();
//	paramsMutex.unlock();
}

void ParticleFilter::motionUpdate( ) {
	time_point new_time = ParticleFilter::clock::now();
	std::chrono::milliseconds current_time_step_ = std::chrono::duration_cast<std::chrono::milliseconds>(new_time - current_time);
	unsigned long long current_time_step_ms = current_time_step_.count();

	current_time = new_time;
	for( ObjectHypothesis& particle: particles )
	{
		particle.motionUpdate(position_noise_std, orientation_noise_std, velocity_noise_std, acceleration_noise_std, current_time_step_ms);
	}
	//cout << "Part i: " << pose.position_x << ", " << pose.position_y << ", " << pose.position_z << endl;

//	cout << "Curr. t.step: " << current_time_step_ms << endl;
//	cout << "Pos STD ORIG: " << position_noise_std << endl;
//	cout << "Orient STD ORIG: " << orientation_noise_std << endl;
}

void ParticleFilter::measurementUpdate(std::vector<FTag2Pose> observations) {
	if ( observations.size() == 0 )
	{
		disable_resampling = true;
		return;
	}
	disable_resampling = false;

//	cout << "Obs. update. Stds: ( " << position_std << ", " << orientation_std << " )" << endl;
	for (ObjectHypothesis& particle: particles) {
		particle.measurementUpdate(observations,position_std,orientation_std);
	}
}

void ParticleFilter::normalizeWeights(){
	if ( disable_resampling == true )
		return;

	double log_min_weight = std::numeric_limits<double>::infinity();
	log_max_weight = -std::numeric_limits<double>::infinity();
	for( ObjectHypothesis& particle: particles )
	{
		if ( log_min_weight > particle.getLogWeight() )
			log_min_weight = particle.getLogWeight();
		if ( log_max_weight < particle.getLogWeight() )
			log_max_weight = particle.getLogWeight();
	}
#ifndef SILENT_PF
	std::cout << "log max weight: " << log_max_weight << std::endl;
	std::cout << "log min weight: " << log_min_weight << std::endl;
#endif
	log_sum_of_weights = 0.0;
	for( ObjectHypothesis& particle: particles )
	{
		particle.setLogWeight(particle.getLogWeight() - log_max_weight);
		log_sum_of_weights += exp(particle.getLogWeight());
	}
	log_sum_of_weights = log(log_sum_of_weights);
#ifndef SILENT_PF
	std::cout << "sum of weights: " << log_sum_of_weights << std::endl;
	std::cout << "log sum of weights: " << log_sum_of_weights << std::endl;
#endif
	for( ObjectHypothesis& particle: particles )
	{
		particle.setLogWeight(particle.getLogWeight() - log_sum_of_weights);
	}
	log_max_weight = - log_sum_of_weights;
#ifndef SILENT_PF
	std::cout << "log max weight: " << log_max_weight << std::endl;
#endif
	double sum_w = 0.0;
	double sum_squared_w = 0.0;
	for ( ObjectHypothesis& particle: particles )
	{
		double w_ = exp(particle.getLogWeight());
		sum_w += w_;
		sum_squared_w += w_ * w_;
	}
	double mean_weight = sum_w/number_of_particles;
#ifndef SILENT_PF
	double Neff = 1.0 / sum_squared_w;
#endif
	double sum_square_diffs = 0.0;
	for ( ObjectHypothesis& particle: particles )
	{
		sum_square_diffs += (exp(particle.getLogWeight()) - mean_weight) * (exp(particle.getLogWeight()) - mean_weight);
	}
#ifndef SILENT_PF
	double weights_std = sqrt(sum_square_diffs / number_of_particles);
	std::cout << "REAL SUM OF WEIGHTS: " << sum_w << std::endl;
	std::cout << "MEAN WEIGHT: " << mean_weight << std::endl;
	std::cout << "WEIGHT STD: " << weights_std << std::endl;
	std::cout << "Effective sample size: " << Neff << std::endl;
#endif
}

void ParticleFilter::resample(){
	if ( disable_resampling == true )
		return;

	std::vector< ObjectHypothesis > newParticles(number_of_particles);

	std::vector<double> weights(number_of_particles);
	std::vector<double> cummulative_weights(number_of_particles+1);
	cummulative_weights[0] = 0.0;
	for ( unsigned int i=0; i<number_of_particles; i++ )
	{
		weights[i] = exp(particles[i].getLogWeight());
		cummulative_weights[i+1] = cummulative_weights[i] + weights[i];
	}

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> discreteUnif(0, number_of_particles-1);

	std::uniform_real_distribution<double> continuousUnif(0,2*exp(log_max_weight));

	unsigned int index = discreteUnif(gen);
	double beta = 0.0;
	for ( unsigned int i=0; i<number_of_particles; i++ )
	{
		beta += continuousUnif(gen);
		double cumm_weight_index = cummulative_weights[index];
		while( beta + cumm_weight_index > cummulative_weights[index+1] )
		{
			beta -= weights[index];
			index = (index+1) % number_of_particles;
			cumm_weight_index = cummulative_weights[index];
		}
		newParticles[i] = particles[index];
	}
	for ( unsigned int i=0; i<number_of_particles; i++ )
		particles[i] = newParticles[i];
}

FTag2Pose ParticleFilter::computeMeanPose(){
	FTag2Pose tracked_pose;

	double current_weight = exp(particles[0].getLogWeight());
//	std::cout << "Mean log W: " << 0 << ": " << particles[0].getLogWeight() << std::endl;
//	std::cout << "Mean W: " << 0 << ": " << current_weight << std::endl;
	tracked_pose.position_x = particles[0].getPose().position_x * current_weight;
	tracked_pose.position_y = particles[0].getPose().position_y * current_weight;
	tracked_pose.position_z = particles[0].getPose().position_z * current_weight;
	tracked_pose.orientation_x = particles[0].getPose().orientation_x * current_weight;
	tracked_pose.orientation_y = particles[0].getPose().orientation_y * current_weight;
	tracked_pose.orientation_z = particles[0].getPose().orientation_z * current_weight;
	tracked_pose.orientation_w = particles[0].getPose().orientation_w * current_weight;
#ifndef SILENT_PF
	std::cout << "Pose x: " << tracked_pose.position_x << std::endl;
	std::cout << "Pose y: " << tracked_pose.position_y << std::endl;
	std::cout << "Pose z: " << tracked_pose.position_z << std::endl;
#endif
	for ( unsigned int i=1; i<number_of_particles; i++ )
	{
		current_weight = exp(particles[i].getLogWeight());
//		std::cout << "Mean log W: " << i << ": " << particles[i].getLogWeight() << std::endl;
//		std::Z << "Mean W: " << i << ": " << current_weight << std::endl;
		tracked_pose.position_x += particles[i].getPose().position_x * current_weight;
		tracked_pose.position_y += particles[i].getPose().position_y * current_weight;
		tracked_pose.position_z += particles[i].getPose().position_z * current_weight;
		tracked_pose.orientation_x += particles[i].getPose().orientation_x * current_weight;
		tracked_pose.orientation_y += particles[i].getPose().orientation_y * current_weight;
		tracked_pose.orientation_z += particles[i].getPose().orientation_z * current_weight;
		tracked_pose.orientation_w += particles[i].getPose().orientation_w * current_weight;
	}

//	tf::Quaternion rMat(tracked_pose.orientation_x,tracked_pose.orientation_y,tracked_pose.orientation_z,tracked_pose.orientation_w);
//	static tf::TransformBroadcaster br;
//	tf::Transform transform;
//	transform.setOrigin( tf::Vector3( tracked_pose.position_x, tracked_pose.position_y, tracked_pose.position_z ) );
//	transform.setRotation( rMat );
//	br.sendTransform( tf::StampedTransform( transform, ros::Time::now(), "camera", "track" ) );

	estimated_pose = tracked_pose;
	return tracked_pose;
}

FTag2Pose ParticleFilter::computeModePose(){
	FTag2Pose tracked_pose;

	log_max_weight = -std::numeric_limits<double>::infinity();
	unsigned int i=0, max_index=0;
	for( ObjectHypothesis& particle: particles )
	{
		if ( log_max_weight < particle.getLogWeight() )
		{
			log_max_weight = particle.getLogWeight();
			max_index = i;
		}
		i++;
	}

	tracked_pose.position_x = particles[max_index].getPose().position_x;
	tracked_pose.position_y = particles[max_index].getPose().position_y;
	tracked_pose.position_z = particles[max_index].getPose().position_z;
	tracked_pose.orientation_x = particles[max_index].getPose().orientation_x;
	tracked_pose.orientation_y = particles[max_index].getPose().orientation_y;
	tracked_pose.orientation_z = particles[max_index].getPose().orientation_z;
	tracked_pose.orientation_w = particles[max_index].getPose().orientation_w;

//	tf::Quaternion rMat(tracked_pose.orientation_x,tracked_pose.orientation_y,tracked_pose.orientation_z,tracked_pose.orientation_w);
//	static tf::TransformBroadcaster br;
//	tf::Transform transform;
//	transform.setOrigin( tf::Vector3( tracked_pose.position_x, tracked_pose.position_y, tracked_pose.position_z ) );
//	transform.setRotation( rMat );
//	br.sendTransform( tf::StampedTransform( transform, ros::Time::now(), "camera", "track" ) );

	estimated_pose = tracked_pose;
	return tracked_pose;
}


void ParticleFilter::publishTrackedPose(int marker_id){
	std::ostringstream frameName;
	frameName << "Track_" << marker_id;
	tf::Quaternion rMat(estimated_pose.orientation_x,estimated_pose.orientation_y,estimated_pose.orientation_z,estimated_pose.orientation_w);
	static tf::TransformBroadcaster br;
	tf::Transform transform;
	transform.setOrigin( tf::Vector3( estimated_pose.position_x, estimated_pose.position_y, estimated_pose.position_z ) );
	transform.setRotation( rMat );
	br.sendTransform( tf::StampedTransform( transform, ros::Time::now(), "camera", frameName.str() ) );
}


void ParticleFilter::displayParticles(int frame_id){
	for ( unsigned int i=0; i<number_of_particles; i++ )
	{
		std::ostringstream frameName;
		frameName << "T" << frame_id << "P" << i;

		tf::Quaternion rMat(particles[i].getPose().orientation_x,particles[i].getPose().orientation_y,particles[i].getPose().orientation_z,particles[i].getPose().orientation_w);
		static tf::TransformBroadcaster br;
		tf::Transform transform;
		transform.setOrigin( tf::Vector3( particles[i].getPose().position_x, particles[i].getPose().position_y, particles[i].getPose().position_z ) );
		transform.setRotation( rMat );
		br.sendTransform( tf::StampedTransform( transform, ros::Time::now(), "camera", frameName.str() ) );
//		std::vector<cv::Vec2i> corners = particles[i].getCorners();

//		std::cout << "Corners of " << i << ": { (" << corners[0][0] << ", " << corners[0][1] << "), (" << corners[1][0] << ", "
//								<< corners[1][1] << "), (" << corners[2][0] << ", " << corners[2][1] << "), (" << corners[3][0] << ", "
//								<< corners[3][1] << ") }" << std::endl;

//		cv::line( img, cv::Point((int)corners[0][0], (int)corners[0][1]), cv::Point((int)corners[1][0], (int)corners[1][1]), cv::Scalar(255,0,0), 1, 8 );
//		cv::line( img, cv::Point((int)corners[1][0], (int)corners[1][1]), cv::Point((int)corners[2][0], (int)corners[2][1]), cv::Scalar(255,0,0), 1, 8 );
//		cv::line( img, cv::Point((int)corners[2][0], (int)corners[2][1]), cv::Point((int)corners[3][0], (int)corners[3][1]), cv::Scalar(255,0,0), 1, 8 );
//		cv::line( img, cv::Point((int)corners[3][0], (int)corners[3][1]), cv::Point((int)corners[0][0], (int)corners[0][1]), cv::Scalar(255,0,0), 1, 8 );
		//PC.drawObject(img);
	}
//	std::cout << "Finished creating image" << std::endl;
//	cv::imshow("Particles", img);
//	std::cout << "Finished drawing image" << std::endl;
}
