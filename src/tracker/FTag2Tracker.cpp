/*
 * Ftag2Tracker.cpp
 *
 *  Created on: Mar 4, 2014
 *      Author: dacocp
 */

#include "tracker/FTag2Tracker.hpp"

bool compareMarkerFilters( MarkerFilter a, MarkerFilter b) { return a.getSumOfStds() < b.getSumOfStds(); }
double markerDistance( FTag2Marker m1, FTag2Marker m2 ) {
	return sqrt( ( m1.position_x - m2.position_x )*( m1.position_x - m2.position_x ) +
			( m1.position_y - m2.position_y )*( m1.position_y - m2.position_y ) +
			( m1.position_z - m2.position_z )*( m1.position_z - m2.position_z ) );
}

void FTag2Tracker::correspondence(std::vector<FTag2Marker> detectedTags_){
	detectedTags = detectedTags_;
	std::sort(filters.begin(), filters.end(), compareMarkerFilters);

	for ( MarkerFilter filter : filters )
	{
		FTag2Marker hypothesis = filter.getHypothesis();
		std::vector<FTag2Marker> candidates;
		for ( FTag2Marker tag : detectedTags )
		{
			if ( tag.withinPhaseRange(hypothesis) ){
				candidates.push_back( tag );
			}
		}

		FTag2Marker found;
		double min_dist = std::numeric_limits<double>::max();
		unsigned int max_idx = 0;
		for ( FTag2Marker candidate: candidates )
		{
			double d = markerDistance(candidate,hypothesis);
			if ( d < min_dist )
			{
				min_dist = d;
				found = candidate;
				max_idx++;
			}
		}

		filters_with_match.push_back(filter);
		detection_matches.push_back(found);
	}
}

void FTag2Tracker::director(std::vector<FTag2Marker> detectedTags_)
{
}

