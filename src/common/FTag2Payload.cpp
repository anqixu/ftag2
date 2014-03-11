#include "common/FTag2Payload.hpp"

#define N_SIGMA 10

using namespace std;

/* TODO: Check the function */
bool FTag2Payload::withinPhaseRange( const FTag2Payload& marker ) {
	bool is_within = true;
	for ( int ray=0 ; ray<phases.rows; ray++ )
	{
		for ( int freq=0 ; freq<phases.cols; freq++ )
		{
			double phObs = phases.at<double>(ray,freq);
			double phFilt = marker.phases.at<double>(ray,freq);
//			cout << "Obs phase: " << phObs << "\t Filt. phase: " << phFilt << endl;
			if ( phFilt < phObs-N_SIGMA*phaseVariances[freq] || phFilt > phObs+N_SIGMA*phaseVariances[freq] )
			{
				is_within = false;
				break;
			}
		}
	}
	return is_within;
}
