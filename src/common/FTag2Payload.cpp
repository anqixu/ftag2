#include "common/FTag2Payload.hpp"

#define N_SIGMA 3

using namespace std;

/* TODO: Check the function */
bool FTag2Payload::withinPhaseRange( FTag2Payload& marker ) {
	bool is_within = true;
	for ( int r=0 ; r<phases.rows; r++ )
	{
		for ( int c=0 ; c<phases.cols; c++ )
		{
			double phObs = phases.at<double>(r,c);
			double phFilt = marker.phases.at<double>(r,c);
			if ( phFilt < phObs-N_SIGMA*phaseVariances[c] || phFilt > phObs+N_SIGMA*phaseVariances[c] )
			{
				is_within = false;
				break;
			}
		}
	}
	return is_within;
}
