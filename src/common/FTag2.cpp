#include "common/FTag2.hpp"
#include "common/BaseCV.hpp"


using namespace std;

double FTag2Pose::getAngleFromCamera() {
  cv::Mat rotMat = vc_math::quat2RotMat(orientation_w, orientation_x, orientation_y, orientation_z);
  return acos(rotMat.at<double>(2, 2));
};


double FTag2Payload::WITHIN_PHASE_RANGE_N_SIGMA = 10.0;
int FTag2Payload::WITHIN_PHASE_RANGE_ALLOWED_MISSMATCHES = 10;
double FTag2Payload::WITHIN_PHASE_RANGE_THRESHOLD = 50.0;


// streamlined implementation using WITHIN_PHASE_RANGE_THRESHOLD
bool FTag2Payload::withinPhaseRange(const FTag2Payload& other) {
  if (type != other.type ||
      phases.rows != other.phases.rows ||
      phases.cols != other.phases.cols) {
    throw std::string("tag type mismatch (from withinPhaseRange)");
    return false;
  }

  double* thisPhases = (double*) phases.data;
  double* otherPhases = (double*) other.phases.data;
  const unsigned int numPhases = phases.rows * phases.cols;

  double avgPhaseDiff = 0.0;
  for (unsigned int i = 0; i < numPhases; i++, thisPhases++, otherPhases++) {
    avgPhaseDiff += vc_math::angularDist(*thisPhases, *otherPhases, 360.0);
  }
  avgPhaseDiff /= numPhases;
  return (avgPhaseDiff < WITHIN_PHASE_RANGE_THRESHOLD);
};


/*
// original implementation using WITHIN_PHASE_RANGE_THRESHOLD
bool FTag2Payload::withinPhaseRange( const FTag2Payload& marker ) {
  cout << "WPRT: " << WITHIN_PHASE_RANGE_THRESHOLD << endl;
  double avg_abs_diff = 0.0;
  int k = 0;
  for ( int ray=0 ; ray<phases.rows; ray++ )
  {
    for ( int freq=0 ; freq<phases.cols; freq++ )
    {
//      cout << "Phase variance (" << ray << ", " << freq << ") = " << phaseVariances[freq] << endl;
      double phObs = phases.at<double>(ray,freq);
      double phFilt = marker.phases.at<double>(ray,freq);

      if (phObs < 0)
    	  phObs += 360;
      else
    	  phObs = fmod(phObs,360.0);

      if (phFilt < 0)
        phFilt += 360;
      else
        phFilt = fmod(phFilt,360.0);

      double abs_diff = fabs(phObs-phFilt);
      avg_abs_diff += abs_diff;
      k++;
//      cout << "Obs phase: " << phObs << "\t Filt. phase: " << phFilt << endl;
    }
  }
  avg_abs_diff /= k;
  //cout << "k = " << k << "\tAvg. phase diff = " << avg_abs_diff << endl;
  if ( avg_abs_diff > WITHIN_PHASE_RANGE_THRESHOLD )
  {
//    cout << "Did not match!" << endl;
    return false;
  }
//  cout << "Matched!" << endl;
  return true;
//  return is_within;
};
*/

/*
// Old version, using WITHIN_PHASE_RANGE_N_SIGMA and WITHIN_PHASE_RANGE_ALLOWED_MISSMATCHES
bool FTag2Payload::withinPhaseRange( const FTag2Payload& marker ) {
  bool is_within = true;
   int count_missmatches = 0;
  cout << "WPRNS: " << WITHIN_PHASE_RANGE_N_SIGMA << endl;
  cout << "WPRAM: " << WITHIN_PHASE_RANGE_ALLOWED_MISSMATCHES << endl;
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
        if ( count_missmatches >= WITHIN_PHASE_RANGE_ALLOWED_MISSMATCHES )
        {
        	is_within = false;
        	break;
        }
      }
    }
  }
  return is_within;
};
*/
