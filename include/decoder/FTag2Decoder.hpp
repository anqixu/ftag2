#ifndef FTAG2DECODER_HPP_
#define FTAG2DECODER_HPP_


#include <opencv2/core/core.hpp>
#include "common/FTag2Marker.hpp"
#include "detector/FTag2Detector.hpp"


class FTag2Decoder {
public:
  static FTag2Marker6S5F3B decodeTag(const cv::Mat quadImg, const Quad& quad,
      double markerWidthM,
      const cv::Mat cameraIntrinsic, const cv::Mat cameraDistortion,
      double quadMaxStripAvgDiff,
      PhaseVariancePredictor& phaseVariancePredictor);

  static void analyzeRays(const cv::Mat& img, FTag2Marker* tag);

  static bool checkSignature(FTag2Marker* tag);

  static void flipPhases(const cv::Mat& phasesSrc, cv::Mat& phasesFlipped);

  static void flipPSK(const cv::Mat& pskSrc, cv::Mat& pskFlipped, unsigned int pskSize);

  static char computeXORChecksum(long long bitChunk, unsigned int numBits);

  static long long _extractSigBits(const cv::Mat& phases, bool flipped, unsigned int pskSize); // TEMP FUNCTION

  static unsigned char bin2gray(unsigned char num);

  static unsigned char gray2bin(unsigned char num);

  static unsigned char adjustPSK(double phaseDeg, unsigned int pskSize);

  static cv::Mat decodePhases(const cv::Mat phases,
      const std::vector<double> phaseVars, const std::vector<int> bitsPerFreq,
      double nStdThresh, bool grayCode = false);
};


#endif /* FTAG2DECODER_HPP_ */
