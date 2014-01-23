#ifndef FTAG2DECODER_HPP_
#define FTAG2DECODER_HPP_


#include <opencv2/core/core.hpp>
#include "common/FTag2Marker.hpp"


class FTag2Decoder {
public:
  static void analyzeRays(const cv::Mat& img, FTag2Marker* tag);

  static void flipPhases(const cv::Mat& phasesSrc, cv::Mat& phasesFlipped);

  static void flipPSK(const cv::Mat& pskSrc, cv::Mat& pskFlipped, unsigned int pskSize);

  static char computeXORChecksum(long long bitChunk, unsigned int numBits);

  static long long _extractSigBits(const cv::Mat& phases, bool flipped, unsigned int pskSize); // TEMP FUNCTION

  static unsigned char bin2gray(unsigned char num);

  static unsigned char gray2bin(unsigned char num);

  static unsigned char adjustPSK(double phaseDeg, unsigned int pskSize);
};


#endif /* FTAG2DECODER_HPP_ */
