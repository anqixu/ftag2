#include "common/FTag2.hpp"
#include "common/BaseCV.hpp"


#define WITHIN_PHASE_RANGE_N_SIGMA 3
#define WITHIN_PHASE_RANGE_ALLOWED_MISSMATCHES 10


using namespace std;


double FTag2Pose::getAngleFromCamera() {
  cv::Mat rotMat = vc_math::quat2RotMat(orientation_w, orientation_x, orientation_y, orientation_z);
  return acos(rotMat.at<double>(2, 2));
};


/* TODO: Check the function */
bool FTag2Payload::withinPhaseRange( const FTag2Payload& marker ) {
//  bool is_within = true;
  unsigned int count_missmatches = 0;
  for ( int ray=0 ; ray<phases.rows; ray++ )
  {
    for ( int freq=0 ; freq<phases.cols; freq++ )
    {
//      cout << "Phase variance (" << ray << ", " << freq << ") = " << phaseVariances[freq] << endl;
      double phObs = phases.at<double>(ray,freq);
      double phFilt = marker.phases.at<double>(ray,freq);
      if (phFilt < 0)
        phFilt += 360;
      else
        phFilt = fmod(phFilt,360.0);
//      cout << "Obs phase: " << phObs << "\t Filt. phase: " << phFilt << endl;
      double phMin = phObs-WITHIN_PHASE_RANGE_N_SIGMA*sqrt(phaseVariances[freq]);
      if ( phMin < 0 )
        phMin += 360;
      double phMax = fmod(phObs+WITHIN_PHASE_RANGE_N_SIGMA*sqrt(phaseVariances[freq]),360.0);
//      cout << "Min/max phase (wrapped around 360): (" << phMin << ", " << phMax << ")" << endl;
      if ( phFilt < phMin || phFilt > phMax )
      {
        count_missmatches ++;
//        is_within = false;
//        break;
      }
    }
  }
  if ( count_missmatches > WITHIN_PHASE_RANGE_ALLOWED_MISSMATCHES )
  {
//    cout << "Did not match!" << endl;
    return false;
  }
//  cout << "Matched!" << endl;
  return true;
//  return is_within;
};
