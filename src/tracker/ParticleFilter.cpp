/*
 * ParticleFilter.cpp
 *
 *  Created on: 2013-10-23
 *      Author: dacocp
 */

#include "tracker/ParticleFilter.hpp"

double ParticleFilter::sampling_percent = 0.9;

ParticleFilter::~ParticleFilter() {
	// TODO Auto-generated destructor stub
}

ParticleFilter::ParticleFilter(int numP, double tagSize, std::vector<FTag2Marker> detections, double position_std, double orientation_std, double position_noise_std, double orientation_noise_std){

	this->tagSize = tagSize;
	number_of_particles = numP;

	int numDetections = detections.size();

	cout << "Creating PF" << endl;

	weights.resize(number_of_particles);
	particles.resize(number_of_particles);
	srand(time(NULL));
	cout << "Particles: " << endl;
	for ( unsigned int i=0; i < number_of_particles; i++ )
	{
		int k = i%numDetections;
		weights[i] = 1.0/number_of_particles;
		particles[i] = ObjectHypothesis(detections.at(k),position_std, orientation_std, position_noise_std, orientation_noise_std, true);
		cout <<  "Pose_x: " << detections[k].position_x << endl;
		cout <<  "Pose_y: " << detections[k].position_y << endl;
		cout <<  "Pose_z: " << detections[k].position_z << endl;
		cout <<  "Part Pose_x: " << particles[i].getPose().position_x << endl;
		cout <<  "Part Pose_y: " << particles[i].getPose().position_y << endl;
		cout <<  "Part Pose_z: " << particles[i].getPose().position_z << endl;
	}
	cout << "Cloud created" << endl;

	disable_resampling = false;
}

void ParticleFilter::motionUpdate() {
	for( unsigned int i=0; i < number_of_particles; i++ )
		particles[i].motionUpdate();
}

void ParticleFilter::measurementUpdate(std::vector<FTag2Marker> detections) {
	if ( detections.size() == 0 )
	{
		disable_resampling = true;
		return;
	}
	disable_resampling = false;

	for (ObjectHypothesis& particle: particles) {
		particle.measurementUpdate(detections);
	}
}

void ParticleFilter::displayParticles(){
	for ( unsigned int i=0; i<number_of_particles; i++ )
	{
		tf::Quaternion rMat(particles[i].getPose().orientation_x,particles[i].getPose().orientation_y,particles[i].getPose().orientation_z,particles[i].getPose().orientation_w);
		static tf::TransformBroadcaster br;
		tf::Transform transform;
		transform.setOrigin( tf::Vector3( particles[i].getPose().position_x, particles[i].getPose().position_y, particles[i].getPose().position_z ) );
		transform.setRotation( rMat );
		std::ostringstream frameName;
		frameName << "Particle_" << i;
		br.sendTransform( tf::StampedTransform( transform, ros::Time::now(), "camera", frameName.str() ) );
//		std::vector<cv::Vec2i> corners = particles[i].getCorners();

//		cout << "Corners of " << i << ": { (" << corners[0][0] << ", " << corners[0][1] << "), (" << corners[1][0] << ", "
//								<< corners[1][1] << "), (" << corners[2][0] << ", " << corners[2][1] << "), (" << corners[3][0] << ", "
//								<< corners[3][1] << ") }" << endl;

//		cv::line( img, cv::Point((int)corners[0][0], (int)corners[0][1]), cv::Point((int)corners[1][0], (int)corners[1][1]), cv::Scalar(255,0,0), 1, 8 );
//		cv::line( img, cv::Point((int)corners[1][0], (int)corners[1][1]), cv::Point((int)corners[2][0], (int)corners[2][1]), cv::Scalar(255,0,0), 1, 8 );
//		cv::line( img, cv::Point((int)corners[2][0], (int)corners[2][1]), cv::Point((int)corners[3][0], (int)corners[3][1]), cv::Scalar(255,0,0), 1, 8 );
//		cv::line( img, cv::Point((int)corners[3][0], (int)corners[3][1]), cv::Point((int)corners[0][0], (int)corners[0][1]), cv::Scalar(255,0,0), 1, 8 );
		//PC.drawObject(img);
	}
//	cout << "Finished creating image" << endl;
//	cv::imshow("Particles", img);
//	cout << "Finished drawing image" << endl;
}

void ParticleFilter::normalizeWeights(){
	log_max_weight = -std::numeric_limits<double>::infinity();
	for( ObjectHypothesis& particle: particles )
	{
		if ( log_max_weight < particle.getLogWeight() )
			log_max_weight = particle.getLogWeight();
	}

	for( ObjectHypothesis& particle: particles )
	{
		particle.setLogWeight(particle.getLogWeight() - log_max_weight);
		log_sum_of_weights += exp(particle.getLogWeight());
	}
	log_sum_of_weights = log(log_sum_of_weights);

	for( ObjectHypothesis& particle: particles )
	{
		particle.setLogWeight(particle.getLogWeight() - log_sum_of_weights);
	}
	log_max_weight = - log_sum_of_weights;
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

FTag2Marker ParticleFilter::computeMeanPose(){

	//cout << "Mean: P" << 0 << ": " << particles[0].getLogWeight() << endl;
	//cout << "Norm. Mean: P" << 0 << ": " << particles[0].getLogWeight()/sum_of_weights << endl;
	FTag2Marker tracked_pose;
	double current_weight = exp(particles[0].getLogWeight());
	tracked_pose.position_x = particles[0].getPose().position_x * current_weight;
	tracked_pose.position_y = particles[0].getPose().position_y * current_weight;
	tracked_pose.position_z = particles[0].getPose().position_z * current_weight;
	tracked_pose.orientation_x = particles[0].getPose().orientation_x * current_weight;
	tracked_pose.orientation_y = particles[0].getPose().orientation_y * current_weight;
	tracked_pose.orientation_z = particles[0].getPose().orientation_z * current_weight;
	tracked_pose.orientation_w = particles[0].getPose().orientation_w * current_weight;
	for ( unsigned int i=1; i<number_of_particles; i++ )
	{
		current_weight = exp(particles[i].getLogWeight());
		//cout << "Mean: Normalized P" << i << ": " << particles[i].getLogWeight() << endl;
		tracked_pose.position_x += particles[i].getPose().position_x * current_weight;
		tracked_pose.position_y += particles[i].getPose().position_y * current_weight;
		tracked_pose.position_z += particles[i].getPose().position_z * current_weight;
		tracked_pose.orientation_x += particles[i].getPose().orientation_x * current_weight;
		tracked_pose.orientation_y += particles[i].getPose().orientation_y * current_weight;
		tracked_pose.orientation_z += particles[i].getPose().orientation_z * current_weight;
		tracked_pose.orientation_w += particles[i].getPose().orientation_w * current_weight;
	}
	tf::Quaternion rMat(tracked_pose.orientation_x,tracked_pose.orientation_y,tracked_pose.orientation_z,tracked_pose.orientation_w);
	static tf::TransformBroadcaster br;
	tf::Transform transform;
	transform.setOrigin( tf::Vector3( tracked_pose.position_x, tracked_pose.position_y, tracked_pose.position_z ) );
	transform.setRotation( rMat );
	br.sendTransform( tf::StampedTransform( transform, ros::Time::now(), "camera", "track" ) );

	return tracked_pose;
}
