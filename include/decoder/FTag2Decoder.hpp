#ifndef FTAG2DECODER_HPP_
#define FTAG2DECODER_HPP_


#include <opencv2/core/core.hpp>
#include "common/FTag2.hpp"
#include "common/VectorAndCircularMath.hpp"


class FTag2Decoder {
public:
  FTag2Decoder();
  ~FTag2Decoder();

  static FTag2 decodeTag(cv::Mat img);
};


#endif /* FTAG2DECODER_HPP_ */
