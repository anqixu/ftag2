#include "common/FTag2.hpp"
#include "common/BaseCV.hpp"


using namespace std;


// DEPRECATED
double FTag2Pose::computeOutOfTagPlaneAngle() {
  // TODO: 0 verify correctness since convention changed to tag-in-camera frame transform (it SHOULD be the same, since this angle is commutative)
  cv::Mat rotMat = vc_math::quat2RotMat(orientation_w, orientation_x, orientation_y, orientation_z);
  return acos(rotMat.at<double>(2, 2));
};


double FTag2Payload::WITHIN_PHASE_RANGE_THRESHOLD = 50.0;


// streamlined implementation using WITHIN_PHASE_RANGE_THRESHOLD
bool FTag2Payload::withinPhaseRange(const FTag2Payload& other) {
  if (type != other.type ||
      phases.rows != other.phases.rows ||
      phases.cols != other.phases.cols) {
    cout << "ERROR! withinPhaseRange tag type mismatch!" << endl <<
        "- type: " << type << " | " << other.type << endl <<
        "- phases.rows: " << phases.rows << " | " << other.phases.rows << endl <<
        "- phases.cols: " << phases.cols << " | " << other.phases.cols << endl << endl;

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
