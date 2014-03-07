#ifndef PAYLOADFILTER_HPP_
#define PAYLOADFILTER_HPP_


#include <chrono>
#include "common/VectorAndCircularMath.hpp"


class PayloadFilter {
  // NOTE: all thetas are expressed in degrees

public:
  // TODO: 2 integrate with David's usage, and put into common header
  typedef std::chrono::system_clock clock;
  typedef std::chrono::time_point<std::chrono::system_clock> time_point;

  // TODO: 2 take these consts from other places
  constexpr static int NUM_RAYS = 6;
  constexpr static int NUM_FREQS = 5;

protected:
  double minTimeBetweenSteps;
  time_point lastStepTime;

  unsigned long long numObservations;
  std::vector<double> sumInverseVars;
  cv::Mat sumWeightedCosTheta;
  cv::Mat sumWeightedSinTheta;

  // TODO: switch to FTag2Payload
  cv::Mat filteredTheta;
  std::vector<double> filteredThetaVariances;


public:
  PayloadFilter(double _minTimeBetweenSteps = 0.0) :
      minTimeBetweenSteps(_minTimeBetweenSteps),
      lastStepTime(clock::now()),
      numObservations(0),
      sumInverseVars(NUM_FREQS, 0.0),
      filteredThetaVariances(NUM_FREQS, -1.0) {
    reset();
  };


  ~PayloadFilter() {};


  void reset() {
    numObservations = 0;
    for (double& v: sumInverseVars) { v = 0.0; }
    for (double& v: filteredThetaVariances) { v = -1.0; }
    sumWeightedCosTheta = cv::Mat::zeros(NUM_RAYS, NUM_FREQS, CV_64FC1);
    sumWeightedSinTheta = cv::Mat::zeros(NUM_RAYS, NUM_FREQS, CV_64FC1);
    filteredTheta = cv::Mat::zeros(NUM_RAYS, NUM_FREQS, CV_64FC1);
  };


  void setParams(double newMinTimeBetweenSteps) {
    minTimeBetweenSteps = newMinTimeBetweenSteps;
  };


  void step(const cv::Mat thetaObs, const std::vector<double> thetaVarObs) {
    // Check if we should update estimate based on time elapsed
    time_point now = clock::now();
    if (numObservations > 0 &&
        (now - lastStepTime).count() < minTimeBetweenSteps) {
      return;
    }

    // Integrate latest observations
    for (int freq = 0; freq < NUM_FREQS; freq++) {
      sumInverseVars[freq] += 1.0/thetaVarObs[freq];
    }
    // TODO: 2 make following code more efficient
    for (int ray = 0; ray < NUM_RAYS; ray++) {
      for (int freq = 0; freq < NUM_FREQS; freq++) {
        double currThetaObsRad = thetaObs.at<double>(ray, freq)*vc_math::degree;
        sumWeightedCosTheta.at<double>(ray, freq) += 1.0/thetaVarObs[freq] * cos(currThetaObsRad);
        sumWeightedSinTheta.at<double>(ray, freq) += 1.0/thetaVarObs[freq] * sin(currThetaObsRad);
      }
    }
    numObservations += 1;
    lastStepTime = now;
  };


  cv::Mat getFilteredTheta() {
    if (numObservations > 0) {
      for (int ray = 0; ray < NUM_RAYS; ray++) {
        for (int freq = 0; freq < NUM_FREQS; freq++) {
          filteredTheta.at<double>(ray, freq) = vc_math::radian *
              atan2(sumWeightedSinTheta.at<double>(ray, freq)/sumInverseVars[freq],
                  sumWeightedCosTheta.at<double>(ray, freq)/sumInverseVars[freq]);
        }
      }
    }
    return filteredTheta;
  };


  std::vector<double> getFilteredVars() {
    if (numObservations > 0) {
      for (int freq = 0; freq < NUM_FREQS; freq++) {
        filteredThetaVariances[freq] = 1.0/sumInverseVars[freq];
      }
    }
    return filteredThetaVariances;
  };
};


#endif /* PAYLOADFILTER_HPP_ */
