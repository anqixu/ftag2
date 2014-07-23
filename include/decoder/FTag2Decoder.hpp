#ifndef FTAG2DECODER_HPP_
#define FTAG2DECODER_HPP_


#include <opencv2/core/core.hpp>
#include "common/FTag2.hpp"
#include "detector/FTag2Detector.hpp"


// TODO: 1 switch from class fn to C fn
class FTag2Decoder {
public:
  static FTag2Marker decodeQuad(const cv::Mat quadImg, const Quad& quad,
      int tagType,
      double markerWidthM,
      unsigned int numSamplesPerRow,
      const cv::Mat cameraIntrinsic, const cv::Mat cameraDistortion,
      double quadMaxStripAvgDiff,
      double tagBorderMeanMaxThresh, double tagBorderStdMaxThresh,
      PhaseVariancePredictor& phaseVariancePredictor);

  static void decodePayload(FTag2Payload& tag, double nStdThresh);

  static int davinqiDist(const FTag2Payload& tag1, const FTag2Payload& tag2);
};


#endif /* FTAG2DECODER_HPP_ */
