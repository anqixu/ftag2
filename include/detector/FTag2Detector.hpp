#ifndef FTAG2DETECTOR_HPP_
#define FTAG2DETECTOR_HPP_


#include <opencv2/core/core.hpp>
#include <list>
#include "common/VectorAndCircularMath.hpp"
#include "common/BaseCV.hpp"


/**
 * Detects line segments (represented as [endA.x, endA.y, endB.x, endB.y])
 * by grouping connected edge elements based on similar edgel directions.
 *
 * @params grayImg: 1-channel image (of type CV_8UC1)
 * @params sobelThreshHigh: maximum hysteresis threshold for Sobel edge detector
 * @params sobelThreshLow: minimum hysteresis threshold for Sobel edge detector
 * @params sobelBlurWidth: width of blur mask for Sobel edge detector (must be 3, 5, or 7)
 * @params ccMinNumEdgels: minimum number of edgels
 * @params angleMargin: in radians
 *
 * TODO: 1 tune params: sobelBlurWidth==3 removes most unwanted edges but fails when tag is moving; sobelBlurWidth==5 gets more spurious edges in general but detects tag boundary when moving
 */
std::list<cv::Vec4i> detectLineSegments(cv::Mat grayImg,
    int sobelThreshHigh = 100, int sobelThreshLow = 30, int sobelBlurWidth = 3,
    unsigned int ccMinNumEdgels = 50, double angleMargin = 20.0*vc_math::degree,
    unsigned int segmentMinNumEdgels = 15);


/**
 * Draws line segments over provided image
 */
void drawLineSegments(cv::Mat img, const std::list<cv::Vec4i> lineSegments);


class FTag2Detector : public BaseCV {
public:
  FTag2Detector();
  ~FTag2Detector();
};


#endif /* FTAG2DETECTOR_HPP_ */
