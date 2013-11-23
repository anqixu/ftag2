#include "detector/FTag2Detector.hpp"
#include "common/BaseCV.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <vector>


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


struct SegmentHough {
  std::list<int> ids;
  std::list<double> dirs;

  double minDir, maxDir;
  double startX, startY, endX, endY;

  SegmentHough(double x, double y) :
    minDir(vc_math::INVALID_ANGLE), maxDir(vc_math::INVALID_ANGLE),
    startX(x), startY(y), endX(x), endY(y) {};

  ~SegmentHough() {
    ids.clear();
    dirs.clear();
  };

  void addEdgel(const Edgel& e) {
    ids.push_back(e.idx);
    dirs.push_back(vc_math::wrapAngle(e.dir, vc_math::two_pi));
    endX = e.x;
    endY = e.y;
  };

  void append(SegmentHough& tail) {
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
typedef std::list<SegmentHough> SegmentHoughs;
typedef SegmentHoughs::iterator SegmentHoughsIt;


bool lessThanInY(const Edgel& a, const Edgel& b) { return (a.y < b.y); }


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
              vc_math::angularDist(theta, thetaEdgel, vc_math::pi)/thetaMargin;
          // TODO: 9 test if decayed range really does help reduce false positive local maxima or not [if not, might even consider switching to integer accumulator!]
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
      rhoBlurSigma, thetaBlurSigma, cv::BORDER_DEFAULT);
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
SegmentHoughs houghExtractSegments(
    double currRho, double currTheta,
    cv::Mat edgelsXY, cv::Mat edgelDir,
    double maxDistToLine, double thetaMargin,
    double minLength, double minGap) {
  SegmentHoughs segments;
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

  const double xLine = currRho * cos(currTheta); // TODO: 9 use cos/sine lookup table to optimize computation
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
    currEdgelDir = edgelDir.at<double>(currPt.y, currPt.x); // TODO: 9 this can be cached via edgelI, and passed as input arg
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
    SegmentHough currSegment(currEdgel->x, currEdgel->y);
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
  SegmentHoughsIt currSeg, nextSeg;
  for (currSeg = segments.begin(); currSeg != segments.end(); currSeg++) {
    nextSeg = currSeg;
    nextSeg++;
    while (nextSeg != segments.end()) {
      // TODO: 9 something might be wrong with this access pattern: we've seen segfaults trying to access nextSeg->startX and startY in certain cases; suspect occurs following delete (look at which iterators are invalidated when removing entries from list; also double-check validity of currSeg; finally ensure that other similar style access of code does not segfault either)
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


bool inRange(double a, double b, double c) {
  if (a <= b && b <= c) { return true; }
  else if (a < vc_math::two_pi && c >= vc_math::two_pi) {
    b += vc_math::two_pi;
    if (a <= b && b <= c) { return true; }
  }
  return false;
};


bool mergeOverlappingSegments(SegmentHough& a, SegmentHoughsIt b) {
  if (inRange(a.minDir, b->minDir, a.maxDir) ||
      inRange(a.minDir, b->maxDir, a.maxDir) ||
      inRange(b->minDir, a.minDir, b->maxDir) ||
      inRange(b->minDir, a.maxDir, b->maxDir)) {
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
std::vector<cv::Vec4i> houghMergeSegments(SegmentHoughs segments,
    const cv::Mat& edgelsXY) {
  std::vector<cv::Vec4i> result;
  SegmentHoughsIt segIt;

  while (!segments.empty()) {
    SegmentHough currSegment = segments.front();
    segments.pop_front();
    for (segIt = segments.begin(); segIt != segments.end();) {
      if (mergeOverlappingSegments(currSegment, segIt)) {
        segIt = segments.erase(segIt);
      } else {
        segIt++;
      }
    }
    const cv::Point2d& endA = edgelsXY.at<cv::Point2d>(currSegment.ids.front(), 0);
    const cv::Point2d& endB = edgelsXY.at<cv::Point2d>(currSegment.ids.back(), 0);
    result.push_back(cv::Vec4i(endA.x, endA.y, endB.x, endB.y));
  }

  return result;
};


/**
 * Identifies potential lines in image using Hough Transform, then use result
 * to identify grouped line segments
 */
std::vector<cv::Vec4i> detectLineSegmentsHough(cv::Mat grayImg,
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
  cv::blur(grayImg, edgelImg, cv::Size(sobelBlurWidth, sobelBlurWidth));
  cv::Canny(edgelImg, edgelImg, sobelThreshLow, sobelThreshHigh, sobelBlurWidth); // args: in, out, low_thresh, high_thresh, gauss_blur
  cv::imshow("edgels", edgelImg);

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
  /*
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
  */

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
  SegmentHoughs partialSegments;
  double currTheta, currRho;
  for (const cv::Point2d& currMax: localMaxima) {
    currRho = houghRhoRes * currMax.x - rhoHalfRange;
    currTheta = houghThetaRes * currMax.y;
    SegmentHoughs currLineSegments =
      houghExtractSegments(currRho, currTheta, edgelsXY, edgelDir,
      houghMaxDistToLine, houghEdgelThetaMargin,
      houghMinSegmentLength, houghMaxSegmentGap);
    partialSegments.splice(partialSegments.end(), currLineSegments);
  }

  // Merge partially overlapping segments into final list of segments
  std::vector<cv::Vec4i> lineSegments = houghMergeSegments(partialSegments, edgelsXY);

  return lineSegments;
};
