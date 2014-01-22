#ifndef FTAG2DECODER_HPP_
#define FTAG2DECODER_HPP_


#include <opencv2/core/core.hpp>
#include "common/FTag2Marker.hpp"


class FTag2Decoder {
public:
  static void extractPayloadFromTag(FTag2Marker* tag);

  static void flipPSK(const cv::Mat& pskSrc, cv::Mat& pskFlipped, unsigned int pskSize);

  static char computeXORChecksum(long long bitChunk, unsigned int numBits);

  static long long _extractSigBits(const cv::Mat& tagPSK, bool flipped); // TEMP FUNCTION
};


#endif /* FTAG2DECODER_HPP_ */
