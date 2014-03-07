/*
 * Ftag2Tracker.cpp
 *
 *  Created on: Mar 4, 2014
 *      Author: dacocp
 */

#include "tracker/FTag2Tracker.hpp"

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

void FTag2Tracker::director(std::vector<FTag2Marker> detectedTags)
{
	correspondence( detectedTags );

	/* UPDATE FILTERS: FILTERS WITH MATCHING DETECTED TAG */
	while ( !filters_with_match.empty() && !detection_matches.empty() )
	{
		filters_with_match.back().step(detection_matches.back());
		filters.push_back( filters_with_match.back() );
		filters_with_match.pop_back();
		detection_matches.pop_back();
	}

	/* SPAWN NEW FILTERS: DETECTED TAGS WITH NO CORRESPONDING FILTER */
	while ( !to_be_spawned.empty() )
	{
		MarkerFilter MF(to_be_spawned.back());
		MF.step(to_be_spawned.back());
		to_be_spawned.pop_back();
		filters.push_back(MF);
	}

	/* UPDATE FILTERS: FILTERS WITH NO MATCHING DETECTED TAGS */
	while ( !not_matched.empty() )
	{
		not_matched.end()->step();
		filters.push_back(not_matched.back());
		not_matched.pop_back();
	}

	/* KILL FILTERS: FILTERS WITH NO MATCHING TAGS FOR MANY CONSECUTIVE FRAMES */
	while ( !ready_to_be_killed.empty() )
	{
		/* TODO: Propperly kill the filters */
		ready_to_be_killed.clear();
	}
}

