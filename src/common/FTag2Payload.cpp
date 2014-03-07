#include "common/FTag2Payload.hpp"

#define N_SIGMA 3

using namespace std;

/* TODO: Write the function */
bool FTag2Payload::withinPhaseRange( FTag2Payload& marker ) {
	bool is_within = true;
	for ( int i=0 ; i<phases.cols; i++ )
	{
		double phObs = phases.at<double>(i);
		double phFilt = marker.phases.at<double>(i);
		if ( phFilt < phObs-N_SIGMA*phaseVariances[i%5] || phFilt > phObs+N_SIGMA*phaseVariances[i%5] )
		{
			is_within = false;
			break;
		}
	}
	return is_within;
}
