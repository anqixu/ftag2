#include "detector/FTag2Detector.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cassert>
#include <list>
#include <vector>

#include <ros/ros.h>
#include <dynamic_reconfigure/server.h>
#include <ftag2/CamTestbenchConfig.h>

#include "common/Profiler.hpp"
#include "common/VectorAndCircularMath.hpp"


//#define SAVE_IMAGES_FROM overlaidImg
#define ENABLE_PROFILER


using namespace std;
using namespace std::chrono;
using namespace cv;
using namespace vc_math;


typedef dynamic_reconfigure::Server<ftag2::CamTestbenchConfig> ReconfigureServer;


const double MAX_CONN_GAP_PX = sqrt(5) * 1.1; // account for worst-case projections of diagonal line (e.g. (1, 1), (3, 2), (5, 3), ... projected + margin of tolerance)


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
typedef std::pair<EdgelsIt, EdgelsIt> EdgelsItPair; // TODO: 0.5 remove this unused typedef


struct Segment {
  std::list<int> ids;
  std::list<double> dirs;

  double minDir, maxDir;
  double startX, startY, endX, endY;

  Segment(double x, double y) :
    minDir(vc_math::INVALID_ANGLE), maxDir(vc_math::INVALID_ANGLE),
    startX(x), startY(y), endX(x), endY(y) {};

  ~Segment() {
    ids.clear();
    dirs.clear();
  };

  void addEdgel(const Edgel& e) {
    ids.push_back(e.idx);
    dirs.push_back(vc_math::wrapAngle(e.dir, vc_math::two_pi));
    endX = e.x;
    endY = e.y;
  };

  void append(Segment& tail) {
    ids.splice(ids.end(), tail.ids);
    dirs.splice(dirs.end(), tail.dirs);
    endX = tail.endX;
    endY = tail.endY;
  };

  void computeDirRange() {
    double minDirShifted, maxDirShifted;
    minDir = vc_math::two_pi;
    maxDir = -1;
    minDirShifted = vc_math::pi*3;
    maxDirShifted = vc_math::pi - 1;

    for (double currDir: dirs) {
      if (currDir < minDir) { minDir = currDir; }
      if (currDir > maxDir) { maxDir = currDir; }

      if (currDir < vc_math::pi) { currDir += vc_math::two_pi; }
      if (currDir < minDirShifted) { minDirShifted = currDir; }
      if (currDir > maxDirShifted) { maxDirShifted = currDir; }
    }

    if ((maxDir - minDir) > (maxDirShifted - minDirShifted) + vc_math::half_pi) {
      minDir = minDirShifted;
      maxDir = maxDirShifted;
    }

    if (minDir >= vc_math::two_pi && maxDir >= vc_math::two_pi) {
      minDir -= vc_math::two_pi;
      maxDir -= vc_math::two_pi;
    }
  };
};
typedef std::list<Segment> Segments;
typedef Segments::iterator SegmentsIt;


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
Segments houghExtractSegments(
    double currRho, double currTheta,
    cv::Mat edgelsXY, cv::Mat edgelDir,
    double maxDistToLine, double thetaMargin,
    double minLength, double minGap) {
  Segments segments;
  Edgels edgelsRot;
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

  // Translate and rotate edgels so that line segments are aligned with y axis
  edgelsXY = edgelsXY.reshape(2, numEdgels);
  cv::add(edgelsXY, cv::Scalar(-xLine, -yLine), edgelsXYRot, cv::noArray(), CV_64FC2);
  edgelsXYRot = edgelsXYRot.reshape(1, numEdgels).t();
  edgelsXYRot = (rotT * edgelsXYRot).t();

  // Identify and sort all edgels that are within (maxDistToLine) pixels to line
  double dirDist;
  for (edgelI = 0; edgelI < numEdgels; edgelI++) {
    const cv::Point2d& currPt = edgelsXY.at<cv::Point2d>(edgelI, 0);
    const cv::Point2d& currPtRot = edgelsXYRot.at<cv::Point2d>(edgelI, 0);
    currEdgelDir = edgelDir.at<double>(currPt.y, currPt.x); // TODO: 0 this can be cached via edgelI, and passed as input arg
    dirDist = vc_math::angularDist(currEdgelDir, currTheta, vc_math::two_pi);
    if (fabs(currPtRot.x) <= maxDistToLine) {
      if (dirDist <= thetaMargin) {
        edgelsRot.push_back(Edgel(currPtRot, currEdgelDir, edgelI));
      } else if (vc_math::pi - dirDist <= thetaMargin) {
        edgelsRot.push_back(Edgel(currPtRot, currEdgelDir + vc_math::pi, edgelI));
      }
    }
  }
  if (edgelsRot.empty()) { return segments; }
  edgelsRot.sort(lessThanInY);

  // Scan for connected components
  EdgelsIt currEdgel, nextEdgel;
  double gapDist;
  for (currEdgel = edgelsRot.begin(); currEdgel != edgelsRot.end(); currEdgel++) {
    Segment currSegment(currEdgel->x, currEdgel->y);
    currSegment.addEdgel(*currEdgel);

    nextEdgel = currEdgel; nextEdgel++;
    while (nextEdgel != edgelsRot.end()) {
      gapDist = vc_math::dist(currSegment.endX, currSegment.endY, nextEdgel->x, nextEdgel->y);
      if (gapDist <= MAX_CONN_GAP_PX) { // found connected edgel
        currSegment.addEdgel(*nextEdgel);
        nextEdgel = edgelsRot.erase(nextEdgel);
      } else if (gapDist > 2*maxDistToLine) {
        break;
      } else { // else current edgel not connected, but next one(s) might still be
        nextEdgel++;
      }
    }

    segments.push_back(currSegment);
  }

  // De-gap segments
  SegmentsIt currSeg, nextSeg;
  for (currSeg = segments.begin(); currSeg != segments.end(); currSeg++) {
    nextSeg = currSeg;
    nextSeg++;
    while (nextSeg != segments.end()) {
      // TODO: 000 something might be wrong with this access pattern: we've seen segfaults trying to access nextSeg->startX and startY in certain cases; suspect occurs following delete (look at which iterators are invalidated when removing entries from list; also double-check validity of currSeg; finally ensure that other similar style access of code does not segfault either)
      gapDist = vc_math::dist(currSeg->endX, currSeg->endY, nextSeg->startX, nextSeg->startY);
      if (gapDist <= minGap) {
        currSeg->append(*nextSeg);
        nextSeg = segments.erase(nextSeg);
      } else if (gapDist > minGap && gapDist > maxDistToLine) {
        break;
      } { // else still possible to merge current segment with further segments
        nextSeg++;
      }
    }
  }

  // Remove short segments and compute dir range
  for (currSeg = segments.begin(); currSeg != segments.end();) {
    if (vc_math::dist(currSeg->startX, currSeg->startY, currSeg->endX, currSeg->endY) < minLength) {
      currSeg = segments.erase(currSeg);
    } else {
      currSeg->computeDirRange();
      currSeg++;
    }
  }

  return segments;
};


bool appendToIdxA(std::list<int>& a, std::list<int>& b) {
  if (b.empty()) { return false; }
  std::list<int>::iterator frontOfB = std::find(a.begin(), a.end(), b.front());
  if (frontOfB != a.end()) {
    std::list<int>::iterator backOfA = std::find(b.begin(), b.end(), a.back());
    if (backOfA != b.end()) { // partial overlap
      b.erase(b.begin(), ++backOfA);
      a.splice(a.end(), b);
      return true;
    } else { // complete overlap
      b.clear();
      return true;
    }
  }

  return false;
};


bool in_range(double a, double b, double c) {
  if (a <= b && b <= c) { return true; }
  else if (a < vc_math::two_pi && c >= vc_math::two_pi) {
    b += vc_math::two_pi;
    if (a <= b && b <= c) { return true; }
  }
  return false;
};


bool mergeOverlappingSegments(Segment& a, SegmentsIt b) {
  if (in_range(a.minDir, b->minDir, a.maxDir) ||
      in_range(a.minDir, b->maxDir, a.maxDir) ||
      in_range(b->minDir, a.minDir, b->maxDir) ||
      in_range(b->minDir, a.maxDir, b->maxDir)) {
    if (appendToIdxA(a.ids, b->ids)) {
      // NOTE: account for shifted ranges
      if (a.minDir < vc_math::two_pi && a.maxDir >= vc_math::two_pi) {
        if (b->minDir < vc_math::two_pi && b->maxDir >= vc_math::two_pi) {
          if (b->minDir < a.minDir) { a.minDir = b->minDir; }
          if (b->maxDir > a.maxDir) { a.maxDir = b->maxDir; }
        } else {
          b->minDir += vc_math::two_pi;
          b->maxDir += vc_math::two_pi;
          if (b->minDir < a.minDir) { a.minDir = b->minDir; }
          if (b->maxDir > a.maxDir) { a.maxDir = b->maxDir; }
        }
      } else {
        if (b->minDir < vc_math::two_pi && b->maxDir >= vc_math::two_pi) {
          a.minDir += vc_math::two_pi;
          a.maxDir += vc_math::two_pi;
          if (b->minDir < a.minDir) { a.minDir = b->minDir; }
          if (b->maxDir > a.maxDir) { a.maxDir = b->maxDir; }
        } else {
          if (b->minDir < a.minDir) { a.minDir = b->minDir; }
          if (b->maxDir > a.maxDir) { a.maxDir = b->maxDir; }
        }
      }
      return true;
    } else if (appendToIdxA(b->ids, a.ids)) {
      a.ids.swap(b->ids);
      // NOTE: account for shifted ranges
      if (a.minDir < vc_math::two_pi && a.maxDir >= vc_math::two_pi) {
        if (b->minDir < vc_math::two_pi && b->maxDir >= vc_math::two_pi) {
          if (b->minDir < a.minDir) { a.minDir = b->minDir; }
          if (b->maxDir > a.maxDir) { a.maxDir = b->maxDir; }
        } else {
          b->minDir += vc_math::two_pi;
          b->maxDir += vc_math::two_pi;
          if (b->minDir < a.minDir) { a.minDir = b->minDir; }
          if (b->maxDir > a.maxDir) { a.maxDir = b->maxDir; }
        }
      } else {
        if (b->minDir < vc_math::two_pi && b->maxDir >= vc_math::two_pi) {
          a.minDir += vc_math::two_pi;
          a.maxDir += vc_math::two_pi;
          if (b->minDir < a.minDir) { a.minDir = b->minDir; }
          if (b->maxDir > a.maxDir) { a.maxDir = b->maxDir; }
        } else {
          if (b->minDir < a.minDir) { a.minDir = b->minDir; }
          if (b->maxDir > a.maxDir) { a.maxDir = b->maxDir; }
        }
      }
      return true;
    }
  }

  return false;
};


/**
 * Finds the union of all (partially) overlapping segments, and returns
 * lists of each merged segment represented by their two end-points
 */
std::vector<cv::Point2d> houghMergeSegments(Segments segments,
    const cv::Mat& edgelsXY) {
  std::vector<cv::Point2d> result;
  SegmentsIt segIt;

  while (!segments.empty()) {
    Segment currSegment = segments.front();
    segments.pop_front();
    for (segIt = segments.begin(); segIt != segments.end();) {
      if (mergeOverlappingSegments(currSegment, segIt)) {
        segIt = segments.erase(segIt);
      } else {
        segIt++;
      }
    }
    result.push_back(edgelsXY.at<cv::Point2d>(currSegment.ids.front(), 0));
    result.push_back(edgelsXY.at<cv::Point2d>(currSegment.ids.back(), 0));
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
std::vector<cv::Point2d> detectLineSegmentsHough(cv::Mat grayImg,
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
  imshow("edgels", edgelImg);

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
    cv::resize(accum, accum, cv::Size(800, 600));
    cv::imshow("accum", accum);
  }

  // DEBUG: draw lines detected by Hough Transform
  if (true) {
    cv::Mat overlaidImg;
    cv::cvtColor(grayImg, overlaidImg, CV_GRAY2RGB);
    std::vector<cv::Point2d> rhoThetas;
    for (const cv::Point2i& currMax: localMaxima) {
      rhoThetas.push_back(cv::Point2d(houghRhoRes * currMax.x - rhoHalfRange, houghThetaRes * currMax.y));
    }

    // TODO: 0 remove after debug
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
  Segments partialSegments;
  double currTheta, currRho;
  for (const cv::Point2d& currMax: localMaxima) {
    currRho = houghRhoRes * currMax.x - rhoHalfRange;
    currTheta = houghThetaRes * currMax.y;
    Segments currLineSegments =
      houghExtractSegments(currRho, currTheta, edgelsXY, edgelDir,
      houghMaxDistToLine, houghEdgelThetaMargin,
      houghMinSegmentLength, houghMaxSegmentGap);
    partialSegments.splice(partialSegments.end(), currLineSegments);
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
    params.houghMinAccumValue = 10;
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
      //cv::resize(sourceImg, sourceImg, cv::Size(), 1/4.0, 1/4.0); // TODO: 0 remove after debugging
    }

    // Setup dynamic reconfigure server
    dynCfgServer = new ReconfigureServer(dynCfgMutex, local_nh);
    dynCfgServer->setCallback(bind(&FTag2Testbench::configCallback, this, _1, _2));

    //namedWindow("source", CV_GUI_EXPANDED);
    namedWindow("debug", CV_GUI_EXPANDED);
    namedWindow("edgels", CV_GUI_EXPANDED);
    //namedWindow("accum", CV_GUI_EXPANDED);
    //namedWindow("lines", CV_GUI_EXPANDED);
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
    //double rho_res = params.houghRhoRes;
    //double theta_res = params.houghThetaRes*degree;
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

      // Extract line segments
      /*
      // OpenCV probabilistic Hough Transform implementation
      // WARNING: often misses obvious segments in image; not exactly sure which params are responsible
      int threshold = 50, linbinratio = 100, angbinratio = 100;
      cv::Mat edgeImg, linesImg;
      std::vector<cv::Vec4i> lines;

      // Detect lines
      cv::blur(grayImg, edgelImg, cv::Size(params.sobelBlurWidth, params.sobelBlurWidth));
      cv::Canny(edgelImg, edgelImg, params.sobelThreshLow, params.sobelThreshHigh, params.sobelBlurWidth); // args: in, out, low_thresh, high_thresh, gauss_blur
      cv::HoughLinesP(edgelImg, lines, max(linbinratio/10.0, 1.0), max(angbinratio/100.0*CV_PI/180, 1/100.0*CV_PI/180), threshold, 10*1.5, 10*1.5/3);

      overlaidImg = sourceImg.clone();
      for (unsigned int i = 0; i < lines.size(); i++) {
        cv::line(overlaidImg, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(0,255,0), 2, 8);
      }
      imshow("segments", overlaidImg);
      */

      /*
      // Our improved Standard Hough Transform implementation
      // WARNING: since the Hough space unevenly samples lines with different angles at larger radii from
      //          the offset point, this implementation often obvious clear segments in image; also it is quite slow
      rho_res = params.houghRhoRes;
      theta_res = params.houghThetaRes*degree;
      lineSegments = detectLineSegmentsHough(grayImg,
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
      */

      // Optimized segment detector using angle-bounded connected edgel components
      std::list<cv::Vec4i> segments = detectLineSegments(grayImg,
          params.sobelThreshHigh, params.sobelThreshLow, params.sobelBlurWidth,
          (unsigned int) params.houghMinAccumValue, params.houghEdgelThetaMargin*degree,
          params.houghMinSegmentLength);
      cv::Mat overlayImg = sourceImgRot.clone();
      //drawLineSegments(overlaidImg, segments); // TODO: 000 debug why passing overlaidImg as value (or ref) ends up with a 0x0 sized image
      for (const cv::Vec4i& endpts: segments) {
        cv::line(overlayImg, cv::Point2i(endpts[0], endpts[1]), cv::Point2i(endpts[2], endpts[3]), CV_RGB(255, 255, 0), 3);
      }
      for (const cv::Vec4i& endpts: segments) {
        cv::line(overlayImg, cv::Point2i(endpts[0], endpts[1]), cv::Point2i(endpts[2], endpts[3]), CV_RGB(255, 0, 0), 1);
      }
      for (const cv::Vec4i& endpts: segments) {
        cv::circle(overlayImg, cv::Point2i(endpts[0], endpts[1]), 2, CV_RGB(0, 0, 255));
        cv::circle(overlayImg, cv::Point2i(endpts[2], endpts[3]), 2, CV_RGB(0, 255, 255));
      }
      cv::imshow("segments", overlayImg);
      cout << "Detected " << segments.size() << " line segments" << endl;

#ifdef SAVE_IMAGES_FROM
      sprintf(dstFilename, "img%05d.jpg", dstID++);
      imwrite(dstFilename, SAVE_IMAGES_FROM);
      ROS_INFO_STREAM("Wrote to " << dstFilename);
#endif

#ifdef ENABLE_PROFILER
      durationProf.toc();

      ros::Time currTime = ros::Time::now();
      ros::Duration td = currTime - latestProfTime;
      if (td.toSec() > 1.0) {
        cout << "Pipeline Duration: " << durationProf.getStatsString() << endl;
        cout << "Pipeline Rate: " << rateProf.getStatsString() << endl;
        latestProfTime = currTime;
      }
#endif

      // Spin ROS and HighGui
      if (!alive) { break; }
      ros::spinOnce();
      c = waitKey(30);
      //c = waitKey(); // TODO: 0 revert back
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
