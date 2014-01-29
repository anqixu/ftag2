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

ParticleFilter::ParticleFilter(int numP, double tagSize, std::vector<FTag2Marker> detections){

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
		particles[i] = ObjectHypothesis(detections.at(k),true);
		cout <<  "Pose_x: " << detections[k].pose_x << endl;
		cout <<  "Pose_y: " << detections[k].pose_y << endl;
		cout <<  "Pose_z: " << detections[k].pose_z << endl;
		cout <<  "Part Pose_x: " << particles[i].getPose().pose_x << endl;
		cout <<  "Part Pose_y: " << particles[i].getPose().pose_y << endl;
		cout <<  "Part Pose_z: " << particles[i].getPose().pose_z << endl;
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

	max_weight = 0.0;
	sum_of_weights = 0.0;
	for( unsigned int i=0; i < number_of_particles; i++ )
	{
		sum_of_weights += particles[i].measurementUpdate(detections);
		if ( max_weight < particles[i].getWeight() )
			max_weight = particles[i].getWeight();
		//cout << "P" << i << ": " << particles[i].getWeight() << endl;
	}
}

void ParticleFilter::displayParticles(){
	for ( unsigned int i=0; i<number_of_particles; i++ )
	{
		tf::Quaternion rMat(particles[i].getPose().orientation_x,particles[i].getPose().orientation_y,particles[i].getPose().orientation_z,particles[i].getPose().orientation_w);
		static tf::TransformBroadcaster br;
		tf::Transform transform;
		transform.setOrigin( tf::Vector3( particles[i].getPose().pose_x, particles[i].getPose().pose_y, particles[i].getPose().pose_z ) );
		transform.setRotation( rMat );
		std::string frameName = "Particle_";
		frameName += i;
		br.sendTransform( tf::StampedTransform( transform, ros::Time::now(), "camera", frameName ) );
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
	double sum = 0.0;
	for ( unsigned int i=0; i<number_of_particles; i++ )
		sum += particles[i].getWeight();
	for ( unsigned int i=0; i<number_of_particles; i++ )
	{
		particles[i].setWeight(particles[i].getWeight()/sum);
		max_weight = max_weight/sum;
		//cout << "Normalized P" << i << ": " << particles[i].getWeight() << endl;
	}
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
		weights[i] = particles[i].getWeight();
		cummulative_weights[i+1] = cummulative_weights[i] + weights[i];
	}

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> discreteUnif(0, number_of_particles-1);

	std::uniform_real_distribution<double> continuousUnif(0,2*max_weight);

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

	//cout << "Mean: P" << 0 << ": " << particles[0].getWeight() << endl;
	//cout << "Norm. Mean: P" << 0 << ": " << particles[0].getWeight()/sum_of_weights << endl;
	FTag2Marker tracked_pose;
	tracked_pose.pose_x = particles[0].getPose().pose_x * particles[0].getWeight()/sum_of_weights;
	tracked_pose.pose_y = particles[0].getPose().pose_y * particles[0].getWeight()/sum_of_weights;;
	tracked_pose.pose_z = particles[0].getPose().pose_z * particles[0].getWeight()/sum_of_weights;;
	tracked_pose.orientation_x = particles[0].getPose().orientation_x * particles[0].getWeight()/sum_of_weights;;
	tracked_pose.orientation_y = particles[0].getPose().orientation_y * particles[0].getWeight()/sum_of_weights;;
	tracked_pose.orientation_z = particles[0].getPose().orientation_z * particles[0].getWeight()/sum_of_weights;;
	tracked_pose.orientation_w = particles[0].getPose().orientation_w * particles[0].getWeight()/sum_of_weights;;
	for ( unsigned int i=1; i<number_of_particles; i++ )
	{
		//cout << "Mean: Normalized P" << i << ": " << particles[i].getWeight() << endl;
		tracked_pose.pose_x += particles[i].getPose().pose_x * particles[i].getWeight()/sum_of_weights;;
		tracked_pose.pose_y += particles[i].getPose().pose_y * particles[i].getWeight()/sum_of_weights;;
		tracked_pose.pose_z += particles[i].getPose().pose_z * particles[i].getWeight()/sum_of_weights;;
		tracked_pose.orientation_x += particles[i].getPose().orientation_x * particles[i].getWeight()/sum_of_weights;;
		tracked_pose.orientation_y += particles[i].getPose().orientation_y * particles[i].getWeight()/sum_of_weights;;
		tracked_pose.orientation_z += particles[i].getPose().orientation_z * particles[i].getWeight()/sum_of_weights;;
		tracked_pose.orientation_w += particles[i].getPose().orientation_w * particles[i].getWeight()/sum_of_weights;;
	}
	tf::Quaternion rMat(tracked_pose.orientation_x,tracked_pose.orientation_y,tracked_pose.orientation_z,tracked_pose.orientation_w);
	static tf::TransformBroadcaster br;
	tf::Transform transform;
	transform.setOrigin( tf::Vector3( tracked_pose.pose_x, tracked_pose.pose_y, tracked_pose.pose_z ) );
	transform.setRotation( rMat );
	br.sendTransform( tf::StampedTransform( transform, ros::Time::now(), "camera", "track" ) );

	return tracked_pose;
}
