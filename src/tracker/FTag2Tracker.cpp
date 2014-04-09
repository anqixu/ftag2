/*
 * Ftag2Tracker.cpp
 *
 *  Created on: Mar 4, 2014
 *      Author: dacocp
 */

#include "tracker/FTag2Tracker.hpp"

using namespace std;

bool compareMarkerFilters( MarkerFilter a, MarkerFilter b) {
	double sum_a = 0.0, sum_b = 0.0;
	for ( double d : a.getHypothesis().payload.phaseVariances )
		sum_a += sqrt(d);
	for ( double d : b.getHypothesis().payload.phaseVariances )
		sum_b += sqrt(d);
	return sum_a < sum_b;
}

double markerDistance( FTag2Pose m1, FTag2Pose m2 ) {
	return sqrt( ( m1.position_x - m2.position_x )*( m1.position_x - m2.position_x ) +
			( m1.position_y - m2.position_y )*( m1.position_y - m2.position_y ) +
			( m1.position_z - m2.position_z )*( m1.position_z - m2.position_z ) );
}

void FTag2Tracker::correspondence(std::vector<FTag2Marker> detectedTags){

	std::sort(filters.begin(), filters.end(), compareMarkerFilters);

//	int i=0;
//	for ( FTag2Marker &tag: detectedTags) {
//		std::cout << "Tag " << i << " payload std: " << tag.payload.sumOfStds() << endl;
//	}

	vector <MarkerFilter>::iterator it1 = filters.begin();
	while( it1 != filters.end() )
	{
		FTag2Marker hypothesis = it1->getHypothesis();
		FTag2Marker found_match;
		double min_dist = std::numeric_limits<double>::max();
		bool found = false;
		vector <FTag2Marker>::iterator it2 = detectedTags.begin();
		vector <FTag2Marker>::iterator temp;
		while ( it2 != detectedTags.end() )
		{
			FTag2Marker tag = *it2;
			if (tag.payload.withinPhaseRange(hypothesis.payload))
			{
				double d = markerDistance(tag.pose,hypothesis.pose);
				if ( d < min_dist )
				{
					min_dist = d;
					found_match = tag;
					found = true;
					temp = it2;
				}
			}
			it2++;
		}
		if ( found )
		{
			filters_with_match.push_back(*it1);
			detection_matches.push_back(found_match);
			detectedTags.erase(temp);
		}
		else {
			if ( it1->get_frames_without_detection() > MAX_FRAMES_NO_DETECTION )
			{
				ready_to_be_killed.push_back(*it1);
			}
			else
			{
				not_matched.push_back(*it1);
			}
		}
	    it1 = filters.erase(it1);
	}
	to_be_spawned = detectedTags;
	detectedTags.clear();
}

void FTag2Tracker::step(std::vector<FTag2Marker> detectedTags)
{
//	cout << "BEFORE correspondence: Num. Filters: " << filters.size() << endl;

//	for ( FTag2Marker f: detectedTags )
//	{
//		std::cout << "Detected tags " << i << ": Variances: ";
//		for ( double d: f.payload.phaseVariances )
//		{
//			cout << d << " ,";
//		}
//		cout << endl;
//	}
	correspondence( detectedTags );

//	cout << "*** After correspondence:  ***" << endl;
//	cout << "Filters: " << filters.size() << endl;
//	cout << filters_with_match.size() << " filters matched" << endl;
//	cout << to_be_spawned.size() << " filters spawned" << endl;
//	cout << not_matched.size() << " filters not matched" << endl;
//	cout << ready_to_be_killed.size() << " to be killed" << endl;


	/* UPDATE FILTERS: FILTERS WITH MATCHING DETECTED TAG */
	while ( !filters_with_match.empty() && !detection_matches.empty() )
	{
//		cout << "UPDATING MATCHED FILTERS" << endl;
		filters_with_match.back().step(detection_matches.back());
		filters.push_back( filters_with_match.back() );
		filters_with_match.pop_back();
		detection_matches.pop_back();
	}

	/* SPAWN NEW FILTERS: DETECTED TAGS WITH NO CORRESPONDING FILTER */
	while ( !to_be_spawned.empty() )
	{
//		cout << "SPAWNING FILTER" << endl;
		MarkerFilter MF(to_be_spawned.back());
		MF.step(to_be_spawned.back());
		to_be_spawned.pop_back();
		filters.push_back(MF);
	}

	/* UPDATE FILTERS: FILTERS WITH NO MATCHING DETECTED TAGS */
	while ( !not_matched.empty() )
	{
//		cout << "UPDATING NOT MATCHED FILTER" << endl;
		not_matched.back().step();
		filters.push_back(not_matched.back());
		not_matched.pop_back();
	}

	/* KILL FILTERS: FILTERS WITH NO MATCHING TAGS FOR MANY CONSECUTIVE FRAMES */
	while ( !ready_to_be_killed.empty() )
	{
//		cout << "KILLING FILTER" << endl;
		/* TODO: Properly kill the filters */
		ready_to_be_killed.clear();
	}
//	std::cout << "After current iteration, " << filters.size() << " filters exist." << endl;
}

void FTag2Tracker::updateParameters(int numberOfParticles_, double position_std_, double orientation_std_, double position_noise_std_, double orientation_noise_std_, double velocity_noise_std_, double acceleration_noise_std_)
{
//	std::cout << "UPDATING PARAMETERS IN FTAG2TRACKER" << std::endl;
	for ( MarkerFilter f: filters )
		f.updateParameters(numberOfParticles_, position_std_, orientation_std_, position_noise_std_, orientation_noise_std_, velocity_noise_std_, acceleration_noise_std_);
}

