#ifndef FTAG2DETECTOR_HPP_
#define FTAG2DETECTOR_HPP_


#include <opencv2/core/core.hpp>
#include <list>
#include <vector>
#include <cmath>
#include "common/VectorAndCircularMath.hpp"
#include "common/BaseCV.hpp"


struct Quad {
  std::vector<cv::Point2f> corners;
  double area;

  void updateArea() {
    double lenA = vc_math::dist(corners[0], corners[1]);
    double lenB = vc_math::dist(corners[1], corners[2]);
    double lenC = vc_math::dist(corners[2], corners[3]);
    double lenD = vc_math::dist(corners[3], corners[0]);
    double angleAD = std::acos(vc_math::dot(corners[1], corners[0], corners[0], corners[3])/lenA/lenD);
    double angleBC = std::acos(vc_math::dot(corners[1], corners[2], corners[2], corners[3])/lenB/lenC);
    area = 0.5*(lenA*lenD*std::sin(angleAD) + lenB*lenC*std::sin(angleBC));
  };

  Quad() : corners(4, cv::Point2d(0, 0)), area(-1.0) {};
};


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
 */
std::vector<cv::Vec4i> detectLineSegments(cv::Mat grayImg,
    int sobelThreshHigh = 100, int sobelThreshLow = 30, int sobelBlurWidth = 3,
    unsigned int ccMinNumEdgels = 50, double angleMargin = 20.0*vc_math::degree,
    unsigned int segmentMinNumEdgels = 15);


std::vector<cv::Vec4i> detectLineSegmentsHough(cv::Mat grayImg,
    int sobelThreshHigh, int sobelThreshLow, int sobelBlurWidth,
    double houghRhoRes, double houghThetaRes,
    double houghEdgelThetaMargin,
    double houghRhoBlurRange, double houghThetaBlurRange,
    double houghRhoNMSRange, double houghThetaNMSRange,
    double houghMinAccumValue,
    double houghMaxDistToLine,
    double houghMinSegmentLength, double houghMaxSegmentGap);


std::list<Quad> detectQuads(const std::vector<cv::Vec4i>& segments,
    double intSegMinAngle = 30.0*vc_math::degree,
    double minEndptDist = 4.0);


inline void drawQuads(cv::Mat img, std::list<Quad> quads) {
  for (Quad& quad: quads) {
    cv::line(img, quad.corners[0], quad.corners[1], CV_RGB(0, 255, 0), 3);
    cv::line(img, quad.corners[0], quad.corners[1], CV_RGB(255, 0, 255), 1);
    cv::line(img, quad.corners[1], quad.corners[2], CV_RGB(0, 255, 0), 3);
    cv::line(img, quad.corners[1], quad.corners[2], CV_RGB(255, 0, 255), 1);
    cv::line(img, quad.corners[2], quad.corners[3], CV_RGB(0, 255, 0), 3);
    cv::line(img, quad.corners[2], quad.corners[3], CV_RGB(255, 0, 255), 1);
    cv::line(img, quad.corners[3], quad.corners[0], CV_RGB(0, 255, 0), 3);
    cv::line(img, quad.corners[3], quad.corners[0], CV_RGB(255, 0, 255), 1);
  }
};


cv::Mat extractQuadImg(cv::Mat img, Quad& quad);


void OpenCVCanny( cv::InputArray _src, cv::OutputArray _dst,
                double low_thresh, double high_thresh,
                int aperture_size, cv::Mat& dx, cv::Mat& dy, bool L2gradient = false );

class FTag2Detector : public BaseCV {
public:
  FTag2Detector();
  ~FTag2Detector();
};


#endif /* FTAG2DETECTOR_HPP_ */
