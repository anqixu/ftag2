#ifndef FTAG2DETECTOR_HPP_
#define FTAG2DETECTOR_HPP_


#include <opencv2/core/core.hpp>
#include <list>
#include <vector>
#include <mutex>
#include <cmath>
#include "common/VectorAndCircularMath.hpp"
#include "common/BaseCV.hpp"
#include "common/FTag2.hpp"


struct Quad {
  std::vector<cv::Point2f> corners; // assumed stored in clockwise order (in image space)
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
 * DEFAULT value for segmentMinNumEdgels is chosen with the assumption that
 * there are up to 5Hz information in each FTag2 horizontal slice, which require
 * a minimum of 10 horizontal pixels to represent. Appending on the 2/8 border
 * slices, and rounding up, results in a minimum of 15 pixels. As for vertical
 * pixels, there must be at least 8 (rounding up to 10) pixels available,
 * in order to recover the 6 horizontal slices within a tag.
 *
 * Furthermore, even under the mildest of blurring settings, there is no
 * guarantee that 2 line segments of an actual object corner will be connected.
 * Therefore, the DEFAULT value for ccMinNumEdgels is chosen to be equal to
 * the one for segmentMinNumEdgels.
 *
 * @params grayImg: 1-channel image (of type CV_8UC1)
 * @params cannyBlurWidth: pre-Canny Gaussian blur width [DEFAULT: minimal noise filtering setting]
 * @params cannyApertureSize: filter width for Sobel edge detector (must be 3, 5, or 7) [DEFAULT: finest edgel setting]
 * @params cannyThreshHigh: maximum hysteresis threshold to filter Sobel edge response [DEFAULT: heuristic]
 * @params cannyThreshLow: minimum hysteresis threshold to filter Sobel edge response [DEFAULT: heuristic]
 * @params ccMinNumEdgels: minimum number of edgels in each accepted connected component [DEFAULT: see comment above]
 * @params angleMargin: maximum angular range (in radians) when connecting edgels into line segments [DEFAULT: conservative heuristic]
 * @params segmentMinNumEdgels: minimum number of edgels in each connected line segment, within a connected component [DEFAULT: see comment above]
 */
std::vector<cv::Vec4i> detectLineSegments(cv::Mat grayImg,
    unsigned int cannyBlurWidth = 3,
    int cannyApertureSize = 3,
    int cannyThreshHigh = 100,
    int cannyThreshLow = 30,
    unsigned int ccMinNumEdgels = 10,
    double angleMargin = 20.0*vc_math::degree,
    unsigned int segmentMinNumEdgels = 10);


/**
 * DEPRECATED: using a fixed-width rho/alpha grid sampling in Hough space
 * is ill-advised since it fails to sample nearby lines that are far from the
 * origin of the Hough space, thus resulting in inequal spatial sampling of
 * edgels in Cartesian space.
 */
std::vector<cv::Vec4i> detectLineSegmentsHough(cv::Mat grayImg,
    int sobelThreshHigh, int sobelThreshLow, int sobelBlurWidth,
    double houghRhoRes, double houghThetaRes,
    double houghEdgelThetaMargin,
    double houghRhoBlurRange, double houghThetaBlurRange,
    double houghRhoNMSRange, double houghThetaNMSRange,
    double houghMinAccumValue,
    double houghMaxDistToLine,
    double houghMinSegmentLength, double houghMaxSegmentGap);


/**
 * Detect quadrilaterals formed by 4 (near-)intersecting line segments, where
 * the intersections do not necessarily need to be near the endpoints of the
 * segments.
 *
 * The resulting quad corners are returned in clockwise order (in image space)
 */
std::list<Quad> scanQuadsExhaustive(const std::vector<cv::Vec4i>& segments,
    double intSegMinAngle = 30.0*vc_math::degree,
    double maxTIntDistRatio = 0.25,
    double maxEndptDistRatio = 0.1,
    double maxCornerGapEndptDistRatio = 0.2,
    double maxEdgeGapDistRatio = 0.5,
    double maxEdgeGapAlignAngle = 15.0*vc_math::degree,
    double minQuadWidth = 15.0);


/**
 * Detect quadrilaterals formed by 4 (near-)intersecting line segments, where
 * the intersections must be near the endpoints of individual segments.
 *
 * This function is much faster than scanQuadsExhaustive, mainly because it uses
 * spatial hashing to drastically cut down the number of pairs of segments to
 * check for proximity. In particular, all segment endpoints are binned into
 * a hash map, whose resolution is (imWidth/hashMapWidth, imHeight/hashMapWidth).
 * Only segments that have endpoints in 8-connected neighbouring bins are
 * considered as neighbours.
 *
 * The resulting quad corners are returned in clockwise order (in image space)
 */
std::list<Quad> scanQuadsSpatialHash(const std::vector<cv::Vec4i>& segments,
    unsigned int imWidth, unsigned int imHeight,
    double intSegMinAngle = 30.0*vc_math::degree,
    unsigned int hashMapWidth = 10,
    double maxTIntDistRatio = 0.25,
    double maxEndptDistRatio = 0.1,
    double maxCornerGapEndptDistRatio = 0.2,
    double maxEdgeGapDistRatio = 0.5,
    double maxEdgeGapAlignAngle = 15.0*vc_math::degree,
    double minQuadWidth = 15.0);


/**
 * Detect quadrilaterals in grayscale image by using adaptive thresholding,
 * contour detection, and approximate polygonal reduction of contours.
 *
 * This approach is heavily inspired by Aruco's quad detector, and is noticeably
 * faster (2-4x) than (detectLineSegments + scanQuadsExhaustive /
 * scanQuadsSpatialHash). On the other hand, it cannot tolerate edge-border
 * occlusions; it can return false quads based on the insides of adaptive
 * thresholded contours; and it also will return skewed quad corner positions
 * if occluded.
 *
 * DEFAULT value for quadMinWidth is chosen with the assumption that
 * there are up to 5Hz information in each FTag2 horizontal slice, which require
 * a minimum of 10 horizontal pixels to represent. Appending on the 2/8 border
 * slices, and rounding up, results in a minimum of 15 pixels. As for vertical
 * pixels, there must be at least 8 (rounding up to 10) pixels available,
 * in order to recover the 6 horizontal slices within a tag.
 *
 * @params adaptiveThreshBlockSize: see cv::adaptiveThreshold() [DEFAULT: thick borders -> more reliable contours]
 * @params adaptiveThreshMeanWeight: see cv::adaptiveThreshold() [>= 3, ODD] [DEFAULT: heuristically chosen ~ adaptiveThreshBlockSize]
 * @params quadMinWidth: minimum pixel width for accepted quad [DEFAULT: see comment above]
 * @params quadMinPerimeter: minimum pixel perimeter count for accepted quad [DEFAULT: 4*quadMinWidth]
 * @params approxPolyEpsSizeRatio: epsilon parameter of cv::approxPolyDP() is set to contourSize * approxPolyEpsSizeRatio [DEFAULT: heuristic]
*/
std::list<Quad> detectQuadsViaContour(cv::Mat grayImg,
  unsigned int adaptiveThreshBlockSize = 9,
  double adaptiveThreshMeanWeight = 9.0,
  unsigned int quadMinWidth = 10,
  unsigned int quadMinPerimeter = 40,
  double approxPolyEpsSizeRatio = 0.05);


/**
 * Thin wrapper for cv::cornerSubPix(); the quad corners are modified in-place.
 *
 * NOTE: this step is considerably faster than detectQuads, for typical SD/HD
 * images with natural scenes. Hence, convergence parameters are heuristically
 * chosen to yield accurate results, with little relative increase in run time.
 *
 * WARNING: there is no guarantee that the refined quads are not overlapping
 *
 * Default parameter values are chosen heuristically.
 */
inline void refineQuadCorners(cv::Mat grayImg, std::list<Quad>& quads,
    unsigned int winSize = 4, unsigned int maxIters = 10, double epsilon = 0.05) {
  if (winSize <= 1) winSize = 1;
  for (Quad& currQuad: quads) {
    cv::cornerSubPix(grayImg, currQuad.corners,
        cv::Size(winSize, winSize), cv::Size(-1, -1),
        cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, maxIters, epsilon));
  }
};


inline void drawQuad(cv::Mat img, const std::vector<cv::Point2f>& corners,
    CvScalar edgeColor = CV_RGB(0, 255, 0), CvScalar fillColor = CV_RGB(255, 0, 255)) {
  cv::line(img, corners[0], corners[1], edgeColor, 3);
  cv::line(img, corners[0], corners[1], fillColor, 1);
  cv::line(img, corners[1], corners[2], edgeColor, 3);
  cv::line(img, corners[1], corners[2], fillColor, 1);
  cv::line(img, corners[2], corners[3], edgeColor, 3);
  cv::line(img, corners[2], corners[3], fillColor, 1);
  cv::line(img, corners[3], corners[0], edgeColor, 3);
  cv::line(img, corners[3], corners[0], fillColor, 1);
};


inline void drawQuadWithCorner(cv::Mat img, const std::vector<cv::Point2f>& corners,
    CvScalar lineEdgeColor = CV_RGB(64, 64, 255), CvScalar lineFillColor = CV_RGB(255, 255, 64),
    CvScalar cornerEdgeColor = CV_RGB(255, 0, 0), CvScalar cornerFillColor = CV_RGB(0, 255, 255)) {
  cv::line(img, corners[0], corners[1], cornerEdgeColor, 3);
  cv::line(img, corners[0], corners[1], cornerFillColor, 1);
  cv::line(img, corners[1], corners[2], cornerEdgeColor, 3);
  cv::line(img, corners[1], corners[2], cornerFillColor, 1);
  cv::line(img, corners[2], corners[3], cornerEdgeColor, 3);
  cv::line(img, corners[2], corners[3], cornerFillColor, 1);
  cv::line(img, corners[3], corners[0], cornerEdgeColor, 3);
  cv::line(img, corners[3], corners[0], cornerFillColor, 1);
  cv::circle(img, corners[0], 5, cornerEdgeColor);
  cv::circle(img, corners[0], 3, cornerFillColor);
};


inline void drawMarkerLabel(cv::Mat img, const std::vector<cv::Point2f>& corners, std::string str,
    int fontFace = cv::FONT_HERSHEY_SIMPLEX, int fontThickness = 1, double fontScale = 0.4,
    CvScalar textBoxColor = CV_RGB(0, 0, 255), CvScalar textColor = CV_RGB(0, 255, 255)) {
  // Compute text size and position
  double mx = 0, my = 0;
  for (const cv::Point2f& pt: corners) {
    mx += pt.x;
    my += pt.y;
  }
  mx /= corners.size();
  my /= corners.size();

  int baseline = 0;
  cv::Size textSize = cv::getTextSize(str, fontFace, fontScale, fontThickness, &baseline);
  cv::Point textCenter(mx - textSize.width/2, my);

  // Draw filled text box and then text
  cv::rectangle(img, textCenter + cv::Point(0, baseline),
      textCenter + cv::Point(textSize.width, -textSize.height),
      textBoxColor, CV_FILLED);
  cv::putText(img, str, textCenter, fontFace, fontScale,
      textColor, fontThickness, 8);
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


cv::Mat extractHorzRays(cv::Mat croppedTag, unsigned int numSamples,
    unsigned int numRays = 6, bool markRays = false);


inline cv::Mat extractVertRays(cv::Mat croppedTag, unsigned int numSamples,
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


/**
 * quadSizeM in meters
 * cameraIntrinsic should be a 3x3 matrix of doubles
 * cameraDistortion should be a 5x1 matrix of doubles
 */
std::vector<cv::Point2f> backProjectQuad(double cam_pose_in_tag_frame_x, double cam_pose_in_tag_frame_y,
		double cam_pose_in_tag_frame_z, double cam_rot_in_tag_frame_w, double cam_rot_in_tag_frame_x,
		double cam_rot_in_tag_frame_y, double cam_rot_in_tag_frame_z, double quadSizeM,
		cv::Mat cameraIntrinsic, cv::Mat cameraDistortion);



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
  PhaseVariancePredictor() : weight_r(-0.433233403141656), weight_z(1.178509836433552), weight_angle(0.225729455615220),
      weight_freq(3.364693352246631), weight_bias(-4.412137643426274) {};

  void updateParams(double w_r, double w_z, double w_a, double w_f, double w_b) {
    paramsMutex.lock();
//    weight_r = w_r;
//    weight_z = w_z;
//    weight_angle = w_a;
//    weight_freq = w_f;
//    weight_bias = w_b;
    paramsMutex.unlock();
  };

  void predict(FTag2Marker* tag) {
    double r = sqrt(tag->pose.position_x*tag->pose.position_x + tag->pose.position_y*tag->pose.position_y);
    double z = tag->pose.position_z;
    double angle = tag->pose.getAngleFromCamera()*vc_math::radian;
    paramsMutex.lock();
    for (unsigned int freq = 1; freq <= tag->payload.phaseVariances.size(); freq++) {
      tag->payload.phaseVariances[freq-1]= pow(weight_bias + weight_r*r + weight_z*z +
          weight_angle*angle + weight_freq*freq,2);
    }
    paramsMutex.unlock();
  };
};


#endif /* FTAG2DETECTOR_HPP_ */
