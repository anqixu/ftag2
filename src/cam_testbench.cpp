#include "detector/FTag2Detector.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "common/Profiler.hpp"
#include <chrono>
#include <algorithm>
#include <cassert>
#include "common/VectorAndCircularMath.hpp"
#include <list>
#include <vector>

#include <ros/ros.h>
#include <dynamic_reconfigure/server.h>
#include <ftag2/CamTestbenchConfig.h>


//#define SAVE_IMAGES_FROM overlaidImg
//#define ENABLE_PROFILER


using namespace std;
using namespace std::chrono;
using namespace cv;
using namespace vc_math;


typedef dynamic_reconfigure::Server<ftag2::CamTestbenchConfig> ReconfigureServer;


struct Edgel {
  double x;
  double y;
  double dir;
  int idx;

  Edgel() : x(0), y(0), dir(vc_math::INVALID_ANGLE), idx(-1) {};
  Edgel(const cv::Point2d& pt, double d, int i) : x(pt.x), y(pt.y), dir(d), idx(i) {};
};
typedef std::list<Edgel> Edgels;
typedef Edgels::iterator EdgelsIt;
typedef std::pair<EdgelsIt, EdgelsIt> EdgelsItPair;


struct Segment {
  std::vector<int> edgelsIdx;
  double minDir, maxDir;

  Segment() : minDir(vc_math::INVALID_ANGLE), maxDir(vc_math::INVALID_ANGLE) {};
};


bool lessThanInY(const Edgel& a, const Edgel& b) { return (a.y < b.y); }


/**
 * Displays double-typed matrix after rescaling from [0, max] to [0, 255]
 *
 * NOTE: negative values in the original matrix will be saturated to 0
 */
void imshow64F(const std::string& winName, const cv::Mat& im) {
  if (im.type() == CV_64FC1) {
    double maxValue;
    cv::Mat im8U;
    cv::minMaxLoc(im, NULL, &maxValue);
    im.convertTo(im8U, CV_8UC1, 255.0/maxValue, 0);
    cv::imshow(winName, im8U);
  } else {
    cv::imshow(winName, im);
  }
};


/**
 * Constructs Hough accumulator space for lines using rho/theta parameterization
 *
 * NOTE: the accumulator accounts for theta values in the range of [0, pi) only,
 *       since line(rho = r, theta = t) == line(rho = -r, theta = t + pi)
 *
 * @param edgels: image mask (of type 8UC1) where edgels have non-zero values
 * @param dx: horizontal gradient response (of type 16SC1); see cv::Canny
 * @param dy: vertical gradient response (of type 16SC1); see cv::Canny
 * @param rhoRes: accumulator bin size for rho axis; in pixels
 * @param thetaRes: accumulator bin size for theta axis; in radians
 * @param thetaMargin: half-width of range of orientation angles centered
 *   around each edgel's (orthogonal) orientation to consider for accumulator;
 *   in radians
 */
cv::Mat houghAccum(const cv::Mat& edgels, const cv::Mat& dx, const cv::Mat& dy,
    double rhoRes, double thetaRes, double thetaMargin,
    unsigned int* edgelCount = NULL, cv::Mat* dir = NULL) {
  // Ensure that input matrices have same size
  assert(edgels.rows == dx.rows);
  assert(edgels.rows == dy.rows);
  assert(edgels.cols == dx.cols);
  assert(edgels.cols == dy.cols);

  // Ensure optimized pixel access capabilities
  assert(edgels.type() == CV_8UC1);
  assert(edgels.isContinuous());
  assert(dx.type() == CV_16SC1);
  assert(dx.isContinuous());
  assert(dy.type() == CV_16SC1);
  assert(dy.isContinuous());

  const unsigned int imWidth = edgels.cols;
  const unsigned int imHeight = edgels.rows;
  const unsigned int rhoHalfRange = imWidth + imHeight - 2;
  const int numRho = std::ceil(double(rhoHalfRange*2 + 1)/rhoRes);
  const int numTheta = std::ceil(vc_math::pi/thetaRes);
  const int numThetaInRange = std::ceil(thetaMargin*2/thetaRes);

  if (dir != NULL) {
    dir->create(imHeight, imWidth, CV_64FC1);
  }

  unsigned int x, y;
  int thetaI, rhoI, thetaCount;
  double thetaEdgel, theta, rho;
  cv::Mat thetaRhoAccum = cv::Mat::zeros(numTheta, numRho, CV_64FC1);
  unsigned int numEdgels = 0;
  for (y = 0; y < imHeight; y++) {
    for (x = 0; x < imWidth; x++) {
      if (edgels.at<unsigned char>(y, x) > 0) {
        numEdgels += 1;
        thetaEdgel = std::atan2(dy.at<signed short>(y, x), dx.at<signed short>(y, x));
        if (dir != NULL) {
          dir->at<double>(y, x) = thetaEdgel;
        }

        // Sweep over theta values in range
        theta = vc_math::wrapAngle(thetaEdgel - thetaMargin, vc_math::pi);
        thetaI = std::floor(theta/thetaRes);
        for (thetaCount = 0; thetaCount < numThetaInRange;
            thetaCount++, thetaI++, theta += thetaRes) {
          if (thetaI >= numTheta) {
            // Since we are only accumulating theta within [0, pi) (rather
            // than [0, 2*pi), we cannot use modulo arithmetics directly
            // to wrap index
            thetaI -= numTheta;
            theta -= vc_math::pi;
          }

          // Determine corresponding rho value and index, and update accumulator
          rho = x*cos(theta) + y*sin(theta);
          rhoI = std::floor(double(rho + rhoHalfRange)/rhoRes);
          thetaRhoAccum.at<double>(thetaI, rhoI) += 1 -
              vc_math::angularDist(theta, thetaEdgel, pi)/thetaMargin;
          // TODO: 0 test if decayed range really does help reduce false positive local maxima or not [if not, might even consider switching to integer accumulator!]

          /*
          // TODO: 0 remove super-sampling code after commit
          // Super-sampling
          for (int ss = 1; ss < 5; ss++) {
            rho = x*cos(theta+houghThetaRes/5*(ss)) + y*sin(theta+houghThetaRes/5*(ss));
            rhoI = round(double(rho + rhoHalfRange)/houghRhoRes);
            thetaRhoAccum.at<double>(thetaI, rhoI) += 1;
          }
          */
        }
      }
    }
  }

  if (edgelCount != NULL) { *edgelCount = numEdgels; }
  return thetaRhoAccum;
};


/**
 * Applies Gaussian blur to rho/theta Hough accumulator space (while accounting
 * for circular+mirrored nature of theta half-space)
 *
 * Blur window parameters are chosen to represent range spanning +/- 3 sigma
 * of Gaussian mask
 *
 * @param accum: numRho x numTheta matrix output from houghAccum; blurred in-place
 * @param rhoRes: accumulator bin size for rho axis; in pixels
 * @param thetaRes: accumulator bin size for theta axis; in radians
 * @param rhoBlurRange: range in rho space for Gaussian window; in pixels
 * @param thetaBlurRange: range in theta space for Gaussian window; in radians
 */
void houghBlur(cv::Mat& accum,
    double rhoRes, double thetaRes,
    double rhoBlurRange, double thetaBlurRange) {
  if (rhoBlurRange <= 0 || thetaBlurRange <= 0) { return; }

  const int numRho = accum.cols;
  const int numTheta = accum.rows;
  const unsigned int rhoBlurHalfWidth = std::min(std::max(
      int(std::floor(std::ceil(rhoBlurRange/rhoRes)/2)),
      1), numRho/2);
  const unsigned int thetaBlurHalfWidth = std::min(std::max(
      int(std::floor(std::ceil(thetaBlurRange/thetaRes)/2)),
      1), numTheta/2);
  const double rhoBlurSigma = double(rhoBlurHalfWidth*2+1)/6;
  const double thetaBlurSigma = double(thetaBlurHalfWidth*2+1)/6;

  // Expand Hough accumulator space by adding zero-borders for rho axis
  // and mirrored-wrapped-around-borders for theta axis
  cv::Mat accumBordered;
  cv::copyMakeBorder(accum, accumBordered,
      thetaBlurHalfWidth, thetaBlurHalfWidth, rhoBlurHalfWidth, rhoBlurHalfWidth,
      cv::BORDER_CONSTANT, cv::Scalar(0));
  cv::Mat accumBorderedTop = accumBordered(
      cv::Range(0, thetaBlurHalfWidth),
      cv::Range(rhoBlurHalfWidth, rhoBlurHalfWidth + numRho));
  cv::Mat accumBottom = accum.rowRange(accum.rows - thetaBlurHalfWidth, accum.rows);
  cv::flip(accumBottom, accumBorderedTop, 1);
  cv::Mat accumBorderedBottom = accumBordered(
      cv::Range(numTheta + thetaBlurHalfWidth, numTheta + thetaBlurHalfWidth*2),
      cv::Range(rhoBlurHalfWidth, rhoBlurHalfWidth + numRho));
  cv::Mat accumTop = accum.rowRange(0, thetaBlurHalfWidth);
  cv::flip(accumTop, accumBorderedBottom, 1);

  // Apply Gaussian blur
  cv::GaussianBlur(accumBordered, accumBordered,
      cv::Size(rhoBlurHalfWidth*2+1, thetaBlurHalfWidth*2+1),
      rhoBlurSigma, thetaBlurSigma, cv::BORDER_DEFAULT); // TODO: 1 find out what is fastest border blur option (since we don't care about border)
  accumBordered(cv::Range(thetaBlurHalfWidth, thetaBlurHalfWidth + numTheta),
      cv::Range(rhoBlurHalfWidth, rhoBlurHalfWidth + numRho)).copyTo(accum);
};


/**
 * Identifies local maxima within the Hough accumulator space using
 * 2-D block-based Non-Maxima Suppression (NMS)
 *
 * @param accum: numRho x numTheta matrix output from houghAccum; blurred in-place
 * @param rhoRes: accumulator bin size for rho axis; in pixels
 * @param thetaRes: accumulator bin size for theta axis; in radians
 * @param rhoNMSRange: range in rho space corresponding to NMS block dimension;
 *  in pixels
 * @param thetaNMSRange: range in theta space corresponding to NMS block
 *  dimension; in radians
 * @param houghMinAccumValue: minimum accepted accumulator value for local maximum
 *  (NOTE: value scale may differ from expectations following houghBlur)
 */
std::vector<cv::Point> houghNMS(const cv::Mat& accum,
    double rhoRes, double thetaRes,
    double rhoNMSRange, double thetaNMSRange, double houghMinAccumValue) {
  const int numRho = accum.cols;
  const int numTheta = accum.rows;
  const int rhoNMSHalfWidth = std::min(std::max(
      int(std::floor(std::ceil(rhoNMSRange/rhoRes)/2)),
      1), numRho/2);
  const int thetaNMSHalfWidth = std::min(std::max(
      int(std::floor(std::ceil(thetaNMSRange/thetaRes)/2)),
      1), numTheta/2);
  const unsigned int rhoNMSWidth = rhoNMSHalfWidth*2 + 1;
  const unsigned int thetaNMSWidth = thetaNMSHalfWidth*2 + 1;
  const unsigned int numRhoBlocks = std::ceil(double(numRho)/rhoNMSWidth);
  const unsigned int numThetaBlocks = std::ceil(double(numTheta)/thetaNMSWidth);
  const unsigned int extraThetaWidth = numThetaBlocks*thetaNMSWidth - numTheta;
  const unsigned int extraRhoWidth = numRhoBlocks*rhoNMSWidth - numRho;
  // NOTE: we need to add extra rows and columns to make the expanded
  //       matrix have width and height that are integer multiples of the
  //       corresponding block width and height

  // Expand the Hough accumulator space by adding zero-borders for rho
  // (horiz axis) and by adding mirrored-wrapped-around-borders for theta
  cv::Mat accumBordered;
  cv::copyMakeBorder(accum, accumBordered,
      thetaNMSHalfWidth, thetaNMSHalfWidth + extraThetaWidth,
      rhoNMSHalfWidth, rhoNMSHalfWidth + extraRhoWidth,
      cv::BORDER_CONSTANT, cv::Scalar(-1));
  cv::Mat accumBorderedTop = accumBordered(
      cv::Range(0, thetaNMSHalfWidth),
      cv::Range(rhoNMSHalfWidth, rhoNMSHalfWidth + numRho));
  cv::Mat accumBottom = accum.rowRange(accum.rows - thetaNMSHalfWidth, accum.rows);
  cv::flip(accumBottom, accumBorderedTop, 1);
  cv::Mat accumBorderedBottom = accumBordered(
      cv::Range(numTheta + thetaNMSHalfWidth, numTheta + thetaNMSHalfWidth*2 + extraThetaWidth),
      cv::Range(rhoNMSHalfWidth, rhoNMSHalfWidth + numRho));
  cv::Mat accumTop = accum.rowRange(0, thetaNMSHalfWidth + extraThetaWidth);
  cv::flip(accumTop, accumBorderedBottom, 1);

  // Iterately find local maxima within non-overlapping blocks
  std::vector<cv::Point> rhoThetaMaxPts;
  int thetaI, rhoI;
  double blockMaxVal;
  cv::Point2i blockMaxPtLocal, blockMaxPt, neighMaxPtLocal;
  cv::Range thetaBlockRange, rhoBlockRange, thetaNeighRange, rhoNeighRange;
  for (thetaI = 0; thetaI < numTheta; thetaI += thetaNMSWidth) {
    thetaBlockRange = cv::Range(thetaI + thetaNMSHalfWidth,
        thetaI + thetaNMSHalfWidth + thetaNMSWidth);

    for (rhoI = 0; rhoI < numRho; rhoI += rhoNMSWidth) {
      // Compute local maximum within block
      rhoBlockRange = cv::Range(rhoI + rhoNMSHalfWidth,
          rhoI + rhoNMSHalfWidth + rhoNMSWidth);
      cv::minMaxLoc(accumBordered(thetaBlockRange, rhoBlockRange),
          NULL, &blockMaxVal, NULL, &blockMaxPtLocal);
      blockMaxPt = blockMaxPtLocal +
          cv::Point2i(rhoI + rhoNMSHalfWidth, thetaI + thetaNMSHalfWidth);

      // Compute local maximum within neighbour centered around block max
      thetaNeighRange = cv::Range(blockMaxPt.y - thetaNMSHalfWidth,
          blockMaxPt.y - thetaNMSHalfWidth + thetaNMSWidth);
      rhoNeighRange = cv::Range(blockMaxPt.x - rhoNMSHalfWidth,
          blockMaxPt.x - rhoNMSHalfWidth + rhoNMSWidth);
      cv::minMaxLoc(accumBordered(thetaNeighRange, rhoNeighRange),
          NULL, NULL, NULL, &neighMaxPtLocal);

      // Accept maximum candidate if it is also its neighbouring window's
      // maximum, and its value is above the requested threshold
      if (neighMaxPtLocal.x == rhoNMSHalfWidth &&
          neighMaxPtLocal.y == thetaNMSHalfWidth &&
          accumBordered.at<double>(blockMaxPt.y, blockMaxPt.x) >= houghMinAccumValue) {
        // Subtract away border widths
        blockMaxPt -= cv::Point(rhoNMSHalfWidth, thetaNMSHalfWidth);
        if (blockMaxPt.x >= 0 && blockMaxPt.x < accum.cols &&
            blockMaxPt.y >= 0 && blockMaxPt.y < accum.rows) { // Sanity check
          rhoThetaMaxPts.push_back(blockMaxPt);
        }
      }
    }
  }

  return rhoThetaMaxPts;
};


/**
 * Matches edgels with line corresponding to (rho, theta) representation,
 * group into segments, filter out short segments, and de-gap short gaps
 *
 * NOTE: consecutive edgels whose gradient directions differ by > 90'
 *       (e.g. white-on-black edge near black-on-white edge) will not be
 *       grouped together into a single segment.
 *
 * @param currRho: maximal position in rho space, in pixels
 * @param currTheta: maximal position in theta value, in radians
 * @param edgelsXY: cv::Mat(CV_64F) containing edgel coordinates (consecutive pairs of doubles)
 * @param edgelDir: image-sized cv::Mat(CV_64F) containing gradient directions, in radians
 * @param maxDistToLine: maximum orthogonal distance to line for accepting edgels, in pixels
 * @param thetaMargin: half-width of range of orientation angles centered
 *   around each edgel's (orthogonal) orientation to consider for accumulator;
 *   in radians
 * @param minLength: minimum accepted length, for filtering short segments; in pixels
 * @param minGap: minimum distance to NOT de-gap 2 consecutive segments
 */
std::vector<Segment> houghExtractSegments(
    double currRho, double currTheta,
    cv::Mat edgelsXY, cv::Mat edgelDir,
    double maxDistToLine, double thetaMargin,
    double minLength, double minGap) {
  std::vector<Segment> segments;
  Edgels edgelsRot;
  std::vector<EdgelsItPair> segmentsRot;
  cv::Mat edgelsXYRot;
  unsigned int edgelI;
  double currEdgelDir;

  cv::Mat rotT(2, 2, CV_64FC1);
  double* rotTPtr = (double*) rotT.data;
  *rotTPtr = cos(currTheta); rotTPtr++;
  *rotTPtr = sin(currTheta); rotTPtr++;
  *rotTPtr = -sin(currTheta); rotTPtr++;
  *rotTPtr = cos(currTheta);

  const double xLine = currRho * cos(currTheta); // TODO: 1 use cos/sine table
  const double yLine = currRho * sin(currTheta);
  const unsigned int numEdgels = edgelsXY.cols * edgelsXY.rows * edgelsXY.channels() / 2;
  const double MAX_PX_GAP = sqrt(5); // account for worst-case projections of diagonal line (e.g. (1, 1), (3, 2), (5, 3), ... projected)
  // TODO: 0 promote MAX_PX_GAP to global var

  // Translate and rotate edgels so that line segments are aligned with y axis
  edgelsXY = edgelsXY.reshape(2, numEdgels);
  cv::add(edgelsXY, cv::Scalar(-xLine, -yLine), edgelsXYRot, cv::noArray(), CV_64FC2);
  edgelsXYRot = edgelsXYRot.reshape(1, numEdgels).t();
  edgelsXYRot = (rotT * edgelsXYRot).t();

  // Identify and sort all edgels that are within (maxDistToLine) pixels to line
  for (edgelI = 0; edgelI < numEdgels; edgelI++) {
    const cv::Point2d& currPt = edgelsXY.at<cv::Point2d>(edgelI, 0);
    const cv::Point2d& currPtRot = edgelsXYRot.at<cv::Point2d>(edgelI, 0);
    currEdgelDir = edgelDir.at<double>(currPt.y, currPt.x); // TODO: 0 this can be cached via edgelI, and passed as input arg
    if (fabs(currPtRot.x) <= maxDistToLine &&
        (vc_math::angularDist(currEdgelDir, currTheta, vc_math::pi) <= thetaMargin)) {
      edgelsRot.push_back(Edgel(currPtRot, currEdgelDir, edgelI));
    }
  }
  if (edgelsRot.size() <= 0) { return segments; }
  edgelsRot.sort(lessThanInY);

  // Extract line segments, filter short segments, and de-gap short gaps
  edgelsRot.push_back(Edgel()); // Insert stub last entry to facilitate loop
  EdgelsIt prevPt = edgelsRot.begin();
  EdgelsIt currPt = prevPt; currPt++;
  EdgelsIt segStart = edgelsRot.begin();
  EdgelsIt segEnd = segStart;
  bool needToAppend = false;
  for (; currPt != edgelsRot.end(); prevPt = currPt, currPt++) {
    if ((currPt->idx < 0) ||
        (vc_math::dist(currPt->x, currPt->y, prevPt->x, prevPt->y) > MAX_PX_GAP) ||
        (vc_math::angularDist(currPt->dir, prevPt->dir, vc_math::two_pi) > vc_math::half_pi)) { // new line segment
      // Check if latest-seen segment is sufficiently long
      segEnd = prevPt;
      if (vc_math::dist(segStart->x, segStart->y, segEnd->x, segEnd->y) >= minLength) {
        needToAppend = true;

        // Check if latest-seen 2 segments can be de-gapped
        if (segmentsRot.size() > 0) {
          const EdgelsIt& lastSegEnd = segmentsRot.back().second;
          if ((vc_math::dist(lastSegEnd->x, lastSegEnd->y, segStart->x, segStart->y) <= minGap) &&
              (vc_math::angularDist(lastSegEnd->dir, segStart->dir, vc_math::two_pi) <= vc_math::half_pi)) {
            segmentsRot.back().second = segEnd;
            needToAppend = false;
          }
        }

        if (needToAppend) {
          segmentsRot.push_back(EdgelsItPair(segStart, segEnd));
          needToAppend = false;
        }
      } else { // Prune last segment due to insufficient length
        edgelsRot.erase(segStart, ++segEnd);
      }

      // Track new segment
      segStart = currPt;
      segEnd = currPt;
    } // else still in current line segment
  } // no need to take care of last segment, since segStart == segEnd == inserted stub point

  // Identify all edgels belong to each segment, and compute the range of their
  // gradient directions
  //
  // NOTE: to account for angular wrap-around, keep both a min/max in the
  //       [0, 2*pi) range and a min/max in the [pi, 3*pi) shifted range
  EdgelsIt currEdgel;
  double minDir, maxDir, minDirShifted, maxDirShifted, currDir;
  for (const EdgelsItPair& segment: segmentsRot) {
    currEdgel = segment.first;
    minDir = vc_math::two_pi;
    maxDir = -1;
    minDirShifted = vc_math::pi*3;
    maxDirShifted = vc_math::pi - 1;
    Segment currSegment;

    while (true) {
      currSegment.edgelsIdx.push_back(currEdgel->idx);

      currDir = vc_math::wrapAngle(currEdgel->dir, vc_math::two_pi);
      if (currDir < minDir) { minDir = currDir; }
      if (currDir > maxDir) { maxDir = currDir; }

      if (currDir < vc_math::pi) { currDir += vc_math::two_pi; }
      if (currDir < minDirShifted) { minDirShifted = currDir; }
      if (currDir > maxDirShifted) { maxDirShifted = currDir; }

      if (currEdgel == segment.second) { break; }
      currEdgel++;
    }

    if ((maxDir - minDir) <= (maxDirShifted - minDirShifted)) {
      currSegment.minDir = minDir;
      currSegment.maxDir = maxDir;
    } else {
      currSegment.minDir = minDirShifted;
      currSegment.maxDir = maxDirShifted;
    }

    segments.push_back(currSegment);
  }

  return segments;
};


// TODO: 0.0 test this fn
bool appendToIdxA(std::vector<int>& a, std::vector<int>& b) {
  std::vector<int>::iterator frontOfB = std::find(a.begin(), a.end(), b.front());
  if (frontOfB != a.end()) {
    std::vector<int>::iterator backOfA = std::find(b.begin(), b.end(), a.back());
    if (backOfA != b.end()) { // partial overlap
      b.erase(b.begin(), ++backOfA);
      if (!b.empty()) {
        a.insert(a.end(), b.begin(), b.end());
      }
      return true;
    } else { // complete overlap
      b.clear();
      return true;
    }
  }

  return false;
};


// TODO: 0.1 test this fn
bool mergeOverlappingSegments(Segment& a, std::vector<Segment>::iterator b) {
  if ((a.minDir <= b->minDir && b->minDir <= a.maxDir) ||
      (a.minDir <= b->maxDir && b->maxDir <= a.maxDir) ||
      (b->minDir <= a.minDir && a.minDir <= b->maxDir) ||
      (b->minDir <= a.maxDir && a.maxDir <= b->maxDir)) {
    if (appendToIdxA(a.edgelsIdx, b->edgelsIdx)) {
      if (b->minDir < a.minDir) { a.minDir = b->minDir; }
      if (b->maxDir > a.maxDir) { a.maxDir = b->maxDir; }
      return true;
    } else if (appendToIdxA(b->edgelsIdx, a.edgelsIdx)) {
      a.edgelsIdx.swap(b->edgelsIdx);
      if (b->minDir < a.minDir) { a.minDir = b->minDir; }
      if (b->maxDir > a.maxDir) { a.maxDir = b->maxDir; }
      return true;
    }
  }

  return false;
};


/**
 * Finds the union of all (partially) overlapping segments, and returns
 * lists of each merged segment represented by their two end-points
 */
std::vector<cv::Point2d> houghMergeSegments(std::vector<Segment> segments,
    const cv::Mat& edgelsXY) {
  // TODO: 0.3 there are some visual evidence that this algo doesn't fully remove duplicates yet... need a case breakdown analysis

  std::vector<cv::Point2d> result;
  Segment currSegment;
  std::vector<Segment>::iterator segIt;

  while (!segments.empty()) {
    currSegment = segments.back();
    segments.pop_back();
    for (segIt = segments.begin(); segIt != segments.end();) {
      if (mergeOverlappingSegments(currSegment, segIt)) {
        segIt = segments.erase(segIt);
      } else {
        segIt++;
      }
    }

    // TODO: 0.2 test this logic!!!
    result.push_back(edgelsXY.at<cv::Point2d>(currSegment.edgelsIdx.front(), 0));
    result.push_back(edgelsXY.at<cv::Point2d>(currSegment.edgelsIdx.back(), 0));
  }

  return result;
};


void drawLines(cv::Mat img, const std::vector<cv::Point2d> rhoThetas) {
  cv::Point pa, pb;
  double angleHemi;
  for (const cv::Point2d& rhoTheta: rhoThetas) {
    angleHemi = vc_math::wrapAngle(rhoTheta.y, vc_math::pi);
    if (angleHemi > vc_math::half_pi) { angleHemi = vc_math::pi - angleHemi; }
    if (angleHemi > vc_math::half_pi/2) {
      pa.x = -img.cols;
      pa.y = round((rhoTheta.x - cos(rhoTheta.y)*pa.x)/sin(rhoTheta.y));
      pb.x = 2*img.cols;
      pb.y = round((rhoTheta.x - cos(rhoTheta.y)*pb.x)/sin(rhoTheta.y));
    } else {
      pa.y = -img.rows;
      pa.x = round((rhoTheta.x - sin(rhoTheta.y)*pa.y)/cos(rhoTheta.y));
      pb.y = 2*img.rows;
      pb.x = round((rhoTheta.x - sin(rhoTheta.y)*pb.y)/cos(rhoTheta.y));
    }
    cv::line(img, pa, pb, CV_RGB(0, 255, 255), 3);
    cv::line(img, pa, pb, CV_RGB(0, 0, 255), 1);
  }
};


/**
 * lineSegments: pairs of start pt and end pt
 */
void drawLineSegments(cv::Mat img, const std::vector<cv::Point2d> lineSegments) {
  unsigned int segmentI;
  for (segmentI = 0; segmentI < lineSegments.size(); segmentI += 2) {
    cv::line(img, lineSegments[segmentI], lineSegments[segmentI + 1], CV_RGB(255, 255, 0), 3);
  }
  for (segmentI = 0; segmentI < lineSegments.size(); segmentI += 2) {
    cv::line(img, lineSegments[segmentI], lineSegments[segmentI + 1], CV_RGB(255, 0, 0), 1);
  }
  for (segmentI = 0; segmentI < lineSegments.size(); segmentI += 2) {
    cv::circle(img, lineSegments[segmentI], 2, CV_RGB(0, 0, 255));
    cv::circle(img, lineSegments[segmentI + 1], 2, CV_RGB(0, 255, 255));
  }
};


/**
 * Identifies potential lines in image using Hough Transform, then use result
 * to identify grouped line segments
 */
std::vector<cv::Point2d> detectLineSegments(cv::Mat grayImg,
    int sobelThreshHigh, int sobelThreshLow, int sobelBlurWidth,
    double houghRhoRes, double houghThetaRes,
    double houghEdgelThetaMargin,
    double houghRhoBlurRange, double houghThetaBlurRange,
    double houghRhoNMSRange, double houghThetaNMSRange,
    double houghMinAccumValue,
    double houghMaxDistToLine,
    double houghMinSegmentLength, double houghMaxSegmentGap) {
  cv::Mat edgelImg, dxImg, dyImg;

  const unsigned int imWidth = grayImg.cols;
  const unsigned int imHeight = grayImg.rows;
  const unsigned int rhoHalfRange = imWidth + imHeight - 2;

  // Identify edgels
  blur(grayImg, edgelImg, Size(sobelBlurWidth, sobelBlurWidth));
  Canny(edgelImg, edgelImg, sobelThreshLow, sobelThreshHigh, sobelBlurWidth); // args: in, out, low_thresh, high_thresh, gauss_blur
  // TODO: 1 tune params: sobelBlurWidth==3 removes most unwanted edges but fails when tag is moving; sobelBlurWidth==5 gets more spurious edges in general but detects tag boundary when moving
  imshow("edgels", edgelImg); // TODO: 0 remove

  // Compute derivative components along x and y axes (needed to compute orientation of edgels)
  cv::Sobel(grayImg, dxImg, CV_16S, 1, 0, sobelBlurWidth, 1, 0, cv::BORDER_REPLICATE);
  cv::Sobel(grayImg, dyImg, CV_16S, 0, 1, sobelBlurWidth, 1, 0, cv::BORDER_REPLICATE);

  // Fill in Hough accumulator space
  unsigned int numEdgels = 0;
  cv::Mat edgelDir;
  cv::Mat thetaRhoAccum = houghAccum(edgelImg, dxImg, dyImg,
    houghRhoRes, houghThetaRes, houghEdgelThetaMargin, &numEdgels, &edgelDir);

  // Blur Hough accumulator space to smooth out spurious local peaks
  houghBlur(thetaRhoAccum,
    houghRhoRes, houghThetaRes, houghRhoBlurRange, houghThetaBlurRange);

  // Identify local maxima within Hough accumulator space
  std::vector<cv::Point2i> localMaxima = houghNMS(thetaRhoAccum,
    houghRhoRes, houghThetaRes, houghRhoNMSRange, houghThetaNMSRange,
    houghMinAccumValue);

  // DEBUG: draw local max in Hough space
  if (true) {
    cv::Mat accum;
    double maxValue;
    cv::minMaxLoc(thetaRhoAccum, NULL, &maxValue);
    thetaRhoAccum.convertTo(accum, CV_8UC1, 255.0/maxValue, 0);
    cv::cvtColor(accum, accum, CV_GRAY2RGB);
    for (const cv::Point2i& currMax: localMaxima) {
      cv::circle(accum, currMax, 5, CV_RGB(0, 255, 0), 5);
      cv::circle(accum, currMax, 2, CV_RGB(255, 0, 0), 2);
    }
    imshow("accum", accum);
  }

  // DEBUG: draw lines detected by Hough Transform
  if (true) {
    cv::Mat overlaidImg;
    cv::cvtColor(grayImg, overlaidImg, CV_GRAY2RGB);
    std::vector<cv::Point2d> rhoThetas;
    for (const cv::Point2i& currMax: localMaxima) {
      rhoThetas.push_back(cv::Point2d(houghRhoRes * currMax.x - rhoHalfRange, houghThetaRes * currMax.y));
    }

    if (false) {
      cout << "HT detected " << localMaxima.size() << " local rho/theta max:" << endl;
      for (const cv::Point2d& rhoTheta: rhoThetas) {
        cout << "- " << rhoTheta.x << " px | " << rhoTheta.y*radian << " deg" << endl;
      }
    }

    drawLines(overlaidImg, rhoThetas);
    cv::imshow("lines", overlaidImg);
  }

  // Accumulate all edgel positions into column-stacked matrix
  assert(edgelImg.type() == CV_8UC1);
  assert(edgelImg.isContinuous());
  cv::Mat edgelsXY(numEdgels, 1, CV_64FC2);
  double* edgelsXYPtr = (double*) edgelsXY.data;
  unsigned char* edgelImgPtr = (unsigned char*) edgelImg.data;
  unsigned int x, y;
  for (y = 0; y < imHeight; y++) {
    for (x = 0; x < imWidth; x++, edgelImgPtr++) {
      if (*edgelImgPtr > 0) {
        *edgelsXYPtr = x; edgelsXYPtr++;
        *edgelsXYPtr = y; edgelsXYPtr++;
      }
    }
  }

  // Identify edgels that match with each (rho, theta) local maxima,
  // and group into line segments
  std::vector<Segment> partialSegments;
  double currTheta, currRho;
  for (const cv::Point2d& currMax: localMaxima) {
    currRho = houghRhoRes * currMax.x - rhoHalfRange;
    currTheta = houghThetaRes * currMax.y;
    std::vector<Segment> currLineSegments =
      houghExtractSegments(currRho, currTheta, edgelsXY, edgelDir,
      houghMaxDistToLine, houghEdgelThetaMargin,
      houghMinSegmentLength, houghMaxSegmentGap);
    partialSegments.insert(partialSegments.end(),
        currLineSegments.begin(), currLineSegments.end());
  }

  // Merge partially overlapping segments into final list of segments
  std::vector<cv::Point2d> lineSegments =
      houghMergeSegments(partialSegments, edgelsXY);

  return lineSegments;
};


// DEBUG fn
cv::Mat genSquareImg(double deg) {
  int width = 300;
  int height = 200;
  cv::Mat testImg = cv::Mat::zeros(height, width, CV_8UC1);
  cv::Point p0(width/2, height/2);
  cv::Point p1, p2;
  cv::Point poly[1][4];
  const cv::Point* polyPtr[1] = {poly[0]};
  poly[0][2] = cv::Point(0, height);
  poly[0][3] = cv::Point(0, 0);
  int numPtsInPoly[] = {4};

  if (false) { // Draw edge
    p1 = cv::Point(double(width + tan(deg*degree)*height)/2.0, 0);
    p2 = p0 - (p1 - p0);
    poly[0][0] = p1;
    poly[0][1] = p2;
    //cv::line(testImg, p1, p2, cv::Scalar(255), width, CV_AA);
    cv::fillPoly(testImg, polyPtr, numPtsInPoly, 1, Scalar(255), 8);
  }

  if (true) { // Draw square
    int bw = width/6;
    int bh = height/6;
    int hw = width/2;
    int hh = height/2;
    poly[0][0] = cv::Point(hw-bw, hh-bh);
    poly[0][1] = cv::Point(hw-bw, hh+bh);
    poly[0][2] = cv::Point(hw+bw, hh+bh);
    poly[0][3] = cv::Point(hw+bw, hh-bh);
    cv::fillPoly(testImg, polyPtr, numPtsInPoly, 1, Scalar(255), 8);
    cv::Mat R = cv::getRotationMatrix2D(cv::Point(hw, hh), deg, 1.0);
    cv::warpAffine(testImg, testImg, R, testImg.size());
  }

  return testImg;
};


class FTag2Testbench {
public:
  FTag2Testbench() :
      local_nh("~"),
      dynCfgSyncReq(false),
      alive(false),
      dstID(0),
      dstFilename((char*) calloc(1000, sizeof(char))),
      latestProfTime(ros::Time::now()) {
    // Low-value params tuned for marginal acceptable results on synthetic images
    params.sobelThreshHigh = 100;
    params.sobelThreshLow = 30;
    params.sobelBlurWidth = 3;
    params.houghRhoRes = 1.0;
    params.houghThetaRes = 1.0; // *degree
    params.houghEdgelThetaMargin = 10.0; // *degree
    params.houghBlurRhoWidth = 5.0; // *houghThetaRes
    params.houghBlurThetaWidth = 7.0; // *houghThetaRes
    params.houghNMSRhoWidth = 9.0; // *houghThetaRes
    params.houghNMSThetaWidth = 13.0; // *houghThetaRes
    params.houghMinAccumValue = 4;
    params.houghMaxDistToLine = 5;
    params.houghMinSegmentLength = 5;
    params.houghMaxSegmentGap = 5;
    params.imRotateDeg = 0;

    /*
    #define GET_PARAM(v) \
      local_nh.param(std::string(#v), params.v, params.v)
    GET_PARAM(sobelThreshHigh);
    GET_PARAM(sobelThreshLow);
    GET_PARAM(sobelBlurWidth);
    GET_PARAM(houghRhoRes);
    GET_PARAM(houghThetaRes);
    GET_PARAM(houghEdgelThetaMargin);
    GET_PARAM(houghBlurRhoWidth);
    GET_PARAM(houghBlurThetaWidth);
    GET_PARAM(houghNMSRhoWidth);
    GET_PARAM(houghNMSThetaWidth);
    GET_PARAM(houghMinAccumValue);
    GET_PARAM(houghMaxDistToLine);
    GET_PARAM(houghMinSegmentLength);
    GET_PARAM(houghMaxSegmentGap);
    GET_PARAM(imRotateDeg);
    #undef GET_PARAM
    dynCfgSyncReq = true;
    */

    std::string source = "";
    local_nh.param("source", source, source);
    if (source.length() <= 0) {
      cam.open(0);
      if (!cam.isOpened()) {
        throw std::string("OpenCV did not detect any cameras.");
      }
    } else {
      sourceImg = cv::imread(source);
      if (sourceImg.empty()) {
        std::ostringstream oss;
        oss << "Failed to load " << source;
        throw oss.str();
      }
    }

    // Setup dynamic reconfigure server
    dynCfgServer = new ReconfigureServer(dynCfgMutex, local_nh);
    dynCfgServer->setCallback(bind(&FTag2Testbench::configCallback, this, _1, _2));

    namedWindow("source", CV_GUI_EXPANDED);
    namedWindow("edgels", CV_GUI_EXPANDED);
    namedWindow("accum", CV_GUI_EXPANDED);
    namedWindow("lines", CV_GUI_EXPANDED);
    namedWindow("segments", CV_GUI_EXPANDED);

    alive = true;
  };


  ~FTag2Testbench() {
    alive = false;
    free(dstFilename);
    dstFilename = NULL;
  };


  void configCallback(ftag2::CamTestbenchConfig& config, uint32_t level) {
    if (!alive) return;
    params = config;
  };


  void spin() {
    double rho_res = params.houghRhoRes;
    double theta_res = params.houghThetaRes*degree;
    cv::Point2d sourceCenter;
    cv::Mat rotMat;
    char c;

    while (ros::ok() && alive) {
#ifdef ENABLE_PROFILER
      rateProf.try_toc();
      rateProf.tic();
      durationProf.tic();
#endif

      // Update params back to dyncfg
      if (dynCfgSyncReq) {
        if (dynCfgMutex.try_lock()) { // Make sure that dynamic reconfigure server or config callback is not active
          dynCfgMutex.unlock();
          dynCfgServer->updateConfig(params);
          ROS_INFO_STREAM("Updated params");
          dynCfgSyncReq = false;
        }
      }

      // Fetch image
      if (cam.isOpened()) {
        cam >> sourceImg;
      }

      // Rotate image and convert to grayscale
      sourceCenter = cv::Point2d(sourceImg.cols/2.0, sourceImg.rows/2.0);
      rotMat = cv::getRotationMatrix2D(sourceCenter, params.imRotateDeg, 1.0);
      cv::warpAffine(sourceImg, sourceImgRot, rotMat, sourceImg.size());
      cv::cvtColor(sourceImgRot, grayImg, CV_RGB2GRAY);

      // Process through Hough transform
//#define OLD_CODE
#ifdef OLD_CODE
      int threshold = 50, linbinratio = 100, angbinratio = 100;
      cv::Mat edgeImg, linesImg;
      std::vector<cv::Vec4i> lines;

      // Detect lines
      cv::blur(grayImg, edgelImg, cv::Size(params.sobelBlurWidth, params.sobelBlurWidth));
      cv::Canny(edgelImg, edgelImg, params.sobelThreshLow, params.sobelThreshHigh, params.sobelBlurWidth); // args: in, out, low_thresh, high_thresh, gauss_blur
      cv::HoughLinesP(edgelImg, lines, max(linbinratio/10.0, 1.0), max(angbinratio/100.0*CV_PI/180, 1/100.0*CV_PI/180), threshold, 10*1.5, 10*1.5/3);
      //cv::HoughLines(edgelImg, lines, max(linbinratio/10.0, 1.0), max(angbinratio/100.0*CV_PI/180, 1/100.0*CV_PI/180), threshold);

      overlaidImg = sourceImg.clone();
      for (unsigned int i = 0; i < lines.size(); i++) {
        cv::line(overlaidImg, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(0,255,0), 2, 8);
      }
      imshow("lines", overlaidImg);
#else
      rho_res = params.houghRhoRes;
      theta_res = params.houghThetaRes*degree;
      lineSegments = detectLineSegments(grayImg,
          params.sobelThreshHigh, params.sobelThreshLow, params.sobelBlurWidth,
          rho_res, theta_res,
          params.houghEdgelThetaMargin*degree,
          params.houghBlurRhoWidth*rho_res, params.houghBlurThetaWidth*theta_res,
          params.houghNMSRhoWidth*rho_res, params.houghNMSThetaWidth*theta_res,
          params.houghMinAccumValue,
          params.houghMaxDistToLine,
          params.houghMinSegmentLength, params.houghMaxSegmentGap);

      overlaidImg = sourceImgRot.clone();
      drawLineSegments(overlaidImg, lineSegments);
      imshow("segments", overlaidImg);
#endif

#ifdef SAVE_IMAGES_FROM
      sprintf(dstFilename, "img%05d.jpg", dstID++);
      imwrite(dstFilename, SAVE_IMAGES_FROM);
      ROS_INFO_STREAM("Wrote to " << dstFilename);
#endif

#ifdef ENABLE_PROFILER
      durationProf.toc();

      ros::Time currTime = ros::Time::now();
      ros::Duration td = currTime - latestProfTime;
      if (td.toSecs() > 1.0) {
        cout << "Pipeline Duration: " << durationProf.getStatsString() << endl;
        cout << "Pipeline Rate: " << rateProf.getStatsString() << endl;
        latestProfTime = currTime;
      }
#endif

      // Spin ROS and HighGui
      ros::spinOnce();
      c = waitKey(30);
      if ((c & 0x0FF) == 'x' || (c & 0x0FF) == 'X') {
        alive = false;
      }
    }
  };


protected:
  ros::NodeHandle local_nh;

  ReconfigureServer* dynCfgServer;
  boost::recursive_mutex dynCfgMutex;
  bool dynCfgSyncReq;

  cv::VideoCapture cam;
  cv::Mat sourceImg, sourceImgRot, grayImg, overlaidImg;
  std::vector<cv::Point2d> lineSegments;

  ftag2::CamTestbenchConfig params;

  bool alive;

  int dstID;
  char* dstFilename;

  Profiler durationProf, rateProf;
  ros::Time latestProfTime;
};


int main(int argc, char** argv) {
  ros::init(argc, argv, "cam_testbench");

  try {
    FTag2Testbench testbench;
    testbench.spin();
  } catch (const cv::Exception& err) {
    cout << "CV EXCEPTION: " << err.what() << endl;
  } catch (const std::string& err) {
    cout << "ERROR: " << err << endl;
  }

  return EXIT_SUCCESS;
};
