#ifndef FTAG2_HPP_
#define FTAG2_HPP_


#include <opencv2/core/core.hpp>
#include <vector>

#include "tracker/ParticleFilter.hpp"

struct FTag2 {
  const static unsigned int MAX_NUM_FREQS;
  const static unsigned int PSK_SIZE;
  const static std::vector<unsigned int> PSK_SIG;

  unsigned int ID;

  std::vector<cv::Mat> rays;
  cv::Mat img;

  bool hasSignature;
  int imgRotDir; // counter-clockwise degrees

  // TEMP VARS
  cv::Mat horzRays;
  cv::Mat vertRays;
  cv::Mat horzMagSpec;
  cv::Mat vertMagSpec;
  cv::Mat horzPhaseSpec;
  cv::Mat vertPhaseSpec;

  cv::Mat horzMags;
  cv::Mat vertMags;
  cv::Mat horzPhases;
  cv::Mat vertPhases;
  cv::Mat PSK;

  ParticleFilter PF;

  FTag2() : ID(0), hasSignature(false), imgRotDir(0) {};
};


#endif /* FTAG2_HPP_ */
