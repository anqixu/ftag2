#ifndef PAYLOADFILTER_HPP_
#define PAYLOADFILTER_HPP_

#include <chrono>
#include "common/VectorAndCircularMath.hpp"
#include "common/FTag2Payload.hpp"


// TODO: 0 test this class
class PayloadFilter {
  // NOTE: all thetas are expressed in degrees

public:
  // TODO: 2 integrate with David's usage, and put into common header
  typedef std::chrono::system_clock clock;
  typedef std::chrono::time_point<std::chrono::system_clock> time_point;

protected:
  double minTimeBetweenSteps;
  time_point lastStepTime;

  unsigned long long numObservations;
  std::vector<double> sumInverseVars;
  cv::Mat sumWeightedCosTheta;
  cv::Mat sumWeightedSinTheta;

  FTag2Payload filteredPayload;


public:
  PayloadFilter(double _minTimeBetweenSteps = 0.0) :
      minTimeBetweenSteps(_minTimeBetweenSteps),
      lastStepTime(clock::now()),
      numObservations(0) {
  };


  ~PayloadFilter() {};


  void reset() {
    numObservations = 0;
    for (double& v: sumInverseVars) { v = 0.0; }
    for (double& v: filteredPayload.phaseVariances) { v = -1.0; }
  };


  void setParams(double newMinTimeBetweenSteps) {
    minTimeBetweenSteps = newMinTimeBetweenSteps;
  };


  void step() {};


  void step(const FTag2Payload& tag) {
    // Check if we should update estimate based on time elapsed
    time_point now = clock::now();
    if (numObservations > 0 &&
        (now - lastStepTime).count() < minTimeBetweenSteps) {
      return;
    }

    // Initialize local storage based on tag information
    const int numRays = tag.phases.rows;
    const int numFreqs = tag.phases.cols;
    if (numObservations == 0) { initializeMatrices(numRays, numFreqs); }

    // Integrate latest observations
    for (int freq = 0; freq < numFreqs; freq++) {
      sumInverseVars[freq] += 1.0/tag.phaseVariances[freq];
      filteredPayload.phaseVariances[freq] = 1.0/sumInverseVars[freq];
    }
    // TODO: 2 make following code more efficient
    for (int ray = 0; ray < numRays; ray++) {
      for (int freq = 0; freq < numFreqs; freq++) {
        double currThetaObsRad = tag.phases.at<double>(ray, freq)*vc_math::degree;
        sumWeightedCosTheta.at<double>(ray, freq) += 1.0/tag.phaseVariances[freq] * std::cos(currThetaObsRad);
        sumWeightedSinTheta.at<double>(ray, freq) += 1.0/tag.phaseVariances[freq] * std::sin(currThetaObsRad);
        filteredPayload.phases.at<double>(ray, freq) = vc_math::radian *
            atan2(sumWeightedSinTheta.at<double>(ray, freq)/sumInverseVars[freq],
                sumWeightedCosTheta.at<double>(ray, freq)/sumInverseVars[freq]);
      }
    }
    numObservations += 1;
    lastStepTime = now;
  };


  FTag2Payload& getFilteredPayload() { return filteredPayload; };


protected:
  void initializeMatrices(int numRays, int numFreqs) {
    sumWeightedCosTheta = cv::Mat::zeros(numRays, numFreqs, CV_64FC1);
    sumWeightedSinTheta = cv::Mat::zeros(numRays, numFreqs, CV_64FC1);
    sumInverseVars.clear();
    for (int freq = 0; freq < numFreqs; freq++) sumInverseVars.push_back(0);
    filteredPayload.phases = cv::Mat::zeros(numRays, numFreqs, CV_64FC1);
    filteredPayload.phaseVariances.clear();
    for (int freq = 0; freq < numFreqs; freq++) filteredPayload.phaseVariances.push_back(-1.0);
  };
};


#endif /* PAYLOADFILTER_HPP_ */
