#ifndef FTAG2DETECTOR_HPP_
#define FTAG2DETECTOR_HPP_


#include <opencv2/core/core.hpp>
#include <list>
#include <vector>
#include <mutex>
#include <cmath>
#include "common/VectorAndCircularMath.hpp"
#include "common/BaseCV.hpp"
#include "common/FTag2Marker.hpp"


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

    // Check for non-simple quads
    if (vc_math::isIntersecting(corners[0], corners[1], corners[2], corners[3]) ||
        vc_math::isIntersecting(corners[1], corners[2], corners[3], corners[0])) {
      area *= -1;
    }
  };

  static bool compareArea(const Quad& first, const Quad& second) {
    return first.area > second.area;
  };

  bool checkMinWidth(double w) {
    return ((vc_math::dist(corners[0], corners[1]) >= w) &&
        (vc_math::dist(corners[1], corners[2]) >= w) &&
        (vc_math::dist(corners[2], corners[3]) >= w) &&
        (vc_math::dist(corners[3], corners[0]) >= w));
  };

  Quad() : corners(4, cv::Point2f(0, 0)), area(-1.0) {};
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
    double maxEndptDist = 6.0);


std::list<Quad> detectQuadsNew(const std::vector<cv::Vec4i>& segments,
    double intSegMinAngle = 30.0*vc_math::degree,
    double maxTIntDistRatio = 0.25,
    double maxEndptDistRatio = 0.1,
    double maxCornerGapEndptDistRatio = 0.2,
    double maxEdgeGapDistRatio = 0.5,
    double maxEdgeGapAlignAngle = 15.0*vc_math::degree,
    double minQuadWidth = 15.0);


inline void drawQuad(cv::Mat img, const std::vector<cv::Point2f>& corners) {
  cv::line(img, corners[0], corners[1], CV_RGB(0, 255, 0), 3);
  cv::line(img, corners[0], corners[1], CV_RGB(255, 0, 255), 1);
  cv::line(img, corners[1], corners[2], CV_RGB(0, 255, 0), 3);
  cv::line(img, corners[1], corners[2], CV_RGB(255, 0, 255), 1);
  cv::line(img, corners[2], corners[3], CV_RGB(0, 255, 0), 3);
  cv::line(img, corners[2], corners[3], CV_RGB(255, 0, 255), 1);
  cv::line(img, corners[3], corners[0], CV_RGB(0, 255, 0), 3);
  cv::line(img, corners[3], corners[0], CV_RGB(255, 0, 255), 1);
};


inline void drawTag(cv::Mat img, const std::vector<cv::Point2f>& corners) {
  cv::line(img, corners[0], corners[1], CV_RGB(0, 0, 255), 3);
  cv::line(img, corners[0], corners[1], CV_RGB(255, 255, 0), 1);
  cv::line(img, corners[1], corners[2], CV_RGB(0, 0, 255), 3);
  cv::line(img, corners[1], corners[2], CV_RGB(255, 255, 0), 1);
  cv::line(img, corners[2], corners[3], CV_RGB(0, 0, 255), 3);
  cv::line(img, corners[2], corners[3], CV_RGB(255, 255, 0), 1);
  cv::line(img, corners[3], corners[0], CV_RGB(0, 0, 255), 3);
  cv::line(img, corners[3], corners[0], CV_RGB(255, 255, 0), 1);
  cv::circle(img, corners[0], 5, CV_RGB(255, 0, 0));
  cv::circle(img, corners[0], 3, CV_RGB(0, 255, 255));
};


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


/**
 * oversample: extract approximately 1 pixel more from each of the sides
 */
cv::Mat extractQuadImg(const cv::Mat img, const Quad& quad, unsigned int minWidth = 8,
    bool oversample = true, bool grayscale = true);


/**
 * Removes columns and/or rows of white pixels from borders of an extracted
 * tag image, caused by rounding accuracy / oversampling in extractQuadImg
 */
cv::Mat trimFTag2Quad(cv::Mat tag, double maxStripAvgDiff = 12.0);


cv::Mat cropFTag2Border(cv::Mat tag, unsigned int numRays = 6, unsigned int borderBlocks = 1);


cv::Mat extractHorzRays(cv::Mat croppedTag, unsigned int numSamples = 1,
    unsigned int numRays = 6, bool markRays = false);


inline cv::Mat extractVertRays(cv::Mat croppedTag, unsigned int numSamples = 1,
    unsigned int numRays = 6, bool markRays = false) {
  return extractHorzRays(croppedTag.t(), numSamples, numRays, markRays);
};


/**
 * quadSizeM in meters
 * cameraIntrinsic should be a 3x3 matrix of doubles
 * cameraDistortion should be a 5x1 matrix of doubles
 * tx, ty, tz are in meters
 */
void solvePose(const std::vector<cv::Point2f> cornersPx, double quadSizeM,
    cv::Mat cameraIntrinsic, cv::Mat cameraDistortion,
    double& tx, double &ty, double& tz,
    double& rw, double& rx, double& ry, double& rz);


void OpenCVCanny( cv::InputArray _src, cv::OutputArray _dst,
                double low_thresh, double high_thresh,
                int aperture_size, cv::Mat& dx, cv::Mat& dy, bool L2gradient = false );


bool validateTagBorder(cv::Mat tag, double meanPxMaxThresh = 80.0, double stdPxMaxThresh = 30.0,
    unsigned int numRays = 6, unsigned int borderBlocks = 1);


/**
 * This class predicts the variance of encoded phases (in degrees) inside a
 * FTag2 marker, using a linear regression model incorporating a constant bias,
 * the marker's distance and angle from the camera, and the frequency of each
 * encoded phase.
 */
class PhaseVariancePredictor {
protected:
  std::mutex paramsMutex;

  double weight_r; // norm of XY components of position
  double weight_z; // projective distance from camera, in camera's ray vector
  double weight_angle; // angle between tag's normal vector and camera's ray vector (in degrees)
  double weight_freq; // encoding frequency of phase
  double weight_bias; // constant bias

public:
  PhaseVariancePredictor() : weight_r(0), weight_z(0), weight_angle(0),
      weight_freq(0), weight_bias(0) {};

  void updateParams(double w_r, double w_z, double w_a, double w_f, double w_b) {
    paramsMutex.lock();
    weight_r = w_r;
    weight_z = w_z;
    weight_angle = w_a;
    weight_freq = w_f;
    weight_bias = w_b;
    paramsMutex.unlock();
  };

  void predict(FTag2Marker* tag) {
    double r = sqrt(tag->pose.position_x*tag->pose.position_x + tag->pose.position_y*tag->pose.position_y);
    double z = tag->pose.position_z;
    double angle = tag->pose.getAngleFromCamera()*vc_math::radian;
    paramsMutex.lock();
    for (unsigned int freq = 1; freq <= tag->payload.phaseVariances.size(); freq++) {
      tag->payload.phaseVariances[freq-1]= weight_bias + weight_r*r + weight_z*z +
          weight_angle*angle + weight_freq*freq;
    }
    paramsMutex.unlock();
  };
};


#endif /* FTAG2DETECTOR_HPP_ */
