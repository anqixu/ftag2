#ifndef FTAG2DECODER_HPP_
#define FTAG2DECODER_HPP_


#include <opencv2/core/core.hpp>
#include "common/FTag2.hpp"
#include "detector/FTag2Detector.hpp"


class FTag2Decoder {
public:
  static FTag2Marker decodeQuad(const cv::Mat quadImg, const Quad& quad,
      double markerWidthM,
      const cv::Mat cameraIntrinsic, const cv::Mat cameraDistortion,
      double quadMaxStripAvgDiff,
      double tagBorderMeanMaxThresh, double tagBorderStdMaxThresh,
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

  static void decodePayload(FTag2Payload& tag, double nStdThresh);

  static int davinqiDist(const FTag2Payload& tag1, const FTag2Payload& tag2);
};


#endif /* FTAG2DECODER_HPP_ */
