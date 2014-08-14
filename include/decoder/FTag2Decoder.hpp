#ifndef FTAG2DECODER_HPP_
#define FTAG2DECODER_HPP_


#include <opencv2/core/core.hpp>
#include "common/FTag2.hpp"
#include "detector/FTag2Detector.hpp"


FTag2Marker decodeQuad(const cv::Mat quadImg, const Quad& quad,
    int tagType,
    double markerWidthM,
    unsigned int numSamplesPerRow,
    const cv::Mat cameraIntrinsic, const cv::Mat cameraDistortion,
    double quadMaxStripAvgDiff,
    double tagBorderMeanMaxThresh, double tagBorderStdMaxThresh,
    PhaseVariancePredictor& phaseVariancePredictor);


void decodePayload(FTag2Payload& tag, double nStdThresh);


#endif /* FTAG2DECODER_HPP_ */
