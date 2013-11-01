#include "detector/FTag2Detector.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "common/Profiler.hpp"
#include <chrono>
#include <algorithm>
#include <cassert>
#include "common/VectorAndCircularMath.hpp"


//#define SAVE_IMAGES_FROM overlaidImg
//#define ENABLE_PROFILER


using namespace std;
using namespace std::chrono;
using namespace cv;
using namespace vc_math;


bool lessThanInY(const cv::Point2d& a, const cv::Point2d& b) { return (a.y < b.y); }

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
    unsigned int* edgelCount = NULL) {
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
 * houghRhoRes: in px
 * houghTheatRes: in radians
 * houghEdgelThetaMargin: in radians; half-width of range of orientation angles centered around edgel's (orthogonal) orientation
 * houghBlurRhoWidth: in px; width of neighbor in rho axis to apply Gaussian blurring
 * houghBlurThetaWidth: in radians; width of neighbor in theta axis to apply Gaussian blurring
 * houghNMSRhoWidth: in px; width of neighbor in rho axis to search for local maxima
 * houghNMSThetaWidth: in radians; width of neighbor in theta axis to search for local maxima
 * houghMaxDistToLine: in px, maximum acceptable point-line distance to consider point belonging to line
 * houghMaxLineGap: in px, maximum de-gap distance between 2 line segments
 */
cv::Mat detectLines(cv::Mat grayImg,
    int sobelThreshHigh, int sobelThreshLow, int sobelBlurWidth,
    double houghRhoRes, double houghThetaRes,
    double houghEdgelThetaMargin,
    double houghBlurRhoWidth, double houghBlurThetaWidth,
    double houghNMSRhoWidth, double houghNMSThetaWidth,
    double houghMaxDistToLine,
    double houghMinSegmentLength, double houghMaxSegmentGap) {
  cv::Mat edgeImg, dxImg, dyImg;

  const unsigned int imWidth = grayImg.cols;
  const unsigned int imHeight = grayImg.rows;

  // Identify edgels
  // TODO: 1 tune params: blur==3 removes most unwanted edges but fails when tag is moving; blur==5 gets more spurious edges in general but detects tag boundary when moving
  blur(grayImg, edgeImg, Size(sobelBlurWidth, sobelBlurWidth));
  Canny(edgeImg, edgeImg, sobelThreshLow, sobelThreshHigh, sobelBlurWidth); // args: in, out, low_thresh, high_thresh, gauss_blur

  imshow("edgels", edgeImg); // TODO: 0 remove

  // Compute derivative components along x and y axes (needed to compute orientation of edgels)
  cv::Sobel(grayImg, dxImg, CV_16S, 1, 0, sobelBlurWidth, 1, 0, cv::BORDER_REPLICATE);
  cv::Sobel(grayImg, dyImg, CV_16S, 0, 1, sobelBlurWidth, 1, 0, cv::BORDER_REPLICATE);

  // Trace potential lines in polar coordinates (Duda & Hart)
  cv::Mat overlaidImg;
  cvtColor(grayImg, overlaidImg, CV_GRAY2RGB);
  {
    // Construct theta/rho Hough accumulator matrix
    // NOTE: given mapping rho = x*cos(theta) + y*sin(theta),
    //       and knowing that cos(theta), sin(theta) \in [-1, 1],
    //       x \in [0, W-1], y \in [0, H-1], therefore the range of rho is
    //       [-W+1, W-1] (+) [-H+1, H-1] = [-W-H+2, W+H-2]
    const unsigned int rhoHalfRange = imWidth + imHeight - 2;
    const unsigned int numRho = ceil(double(2*rhoHalfRange + 1)/houghRhoRes);
    const unsigned int numTheta = ceil(pi/houghThetaRes);
    const int halfNumRho = int(numRho/2);
    const int halfNumTheta = int(numTheta/2);

    const unsigned int houghBlurRhoWindowHalfSize =
        std::min(std::max(int(std::floor(std::ceil(houghBlurRhoWidth/houghRhoRes)/2)), 1), halfNumRho);
    const unsigned int houghBlurThetaWindowHalfSize =
        std::min(std::max(int(std::floor(std::ceil(houghBlurThetaWidth/houghThetaRes)/2)), 1), halfNumTheta);
    const double houghBlurRhoStd = double(houghBlurRhoWindowHalfSize*2+1)/6; // Assume that filter contains [-3*s, 3*s] of Gaussian
    const double houghBlurThetaStd = double(houghBlurThetaWindowHalfSize*2+1)/6;

    const unsigned int houghNMSRhoWindowHalfSize =
        std::min(std::max(int(std::floor(std::ceil(houghNMSRhoWidth/houghRhoRes)/2)), 1), halfNumRho);
    const unsigned int houghNMSThetaWindowHalfSize =
        std::min(std::max(int(std::floor(std::ceil(houghNMSThetaWidth/houghThetaRes)/2)), 1), halfNumTheta);
    const unsigned int houghNMSRhoWindowSize = houghNMSRhoWindowHalfSize*2 + 1;
    const unsigned int houghNMSThetaWindowSize = houghNMSThetaWindowHalfSize*2 + 1;
    const unsigned int houghNMSRhoNumWindows = ceil(double(numRho)/houghNMSRhoWindowSize);
    const unsigned int houghNMSThetaNumWindows = ceil(double(numTheta)/houghNMSThetaWindowSize);

    const double MAX_PX_GAP = 1.0/cos(45*degree); // account for worst-case projections of diagonal line (e.g. (1, 1), (2, 2), (3, 3), ... projected)
    // TODO: 0 come back to MAX_PX_GAP param and explain why we need to increase tolerance (suspect have to do with Canny edge detector + something...


    // Fill in Hough accumulator
    unsigned int numEdgels = 0;
    cv::Mat thetaRhoAccum = houghAccum(edgeImg, dxImg, dyImg,
        houghRhoRes, houghThetaRes, houghEdgelThetaMargin, &numEdgels);
    imshow64F("accum", thetaRhoAccum);


    cv::Mat thetaRhoAccumBlurred;
    if (houghBlurRhoWidth > 0 && houghBlurThetaWidth > 0) { // BLUR
      // Expand the Hough accumulator space by adding zero-borders for rho
      // (horiz axis) and by adding mirrored-wrapped-around-borders for theta
      cv::Mat thetaRhoAccumWithBlurBorder, thetaRhoAccumBottomFlipped, thetaRhoAccumTopFlipped;
      cv::Mat thetaRhoAccumBottom =
          thetaRhoAccum.rowRange(thetaRhoAccum.rows - houghBlurThetaWindowHalfSize, thetaRhoAccum.rows);
      cv::flip(thetaRhoAccumBottom, thetaRhoAccumBottomFlipped, 1);
      cv::Mat thetaRhoAccumTop =
          thetaRhoAccum.rowRange(0, houghBlurThetaWindowHalfSize);
      cv::flip(thetaRhoAccumTop, thetaRhoAccumTopFlipped, 1);
      cv::vconcat(thetaRhoAccumBottomFlipped, thetaRhoAccum, thetaRhoAccumWithBlurBorder);
      cv::vconcat(thetaRhoAccumWithBlurBorder, thetaRhoAccumTopFlipped, thetaRhoAccumWithBlurBorder);
      cv::copyMakeBorder(thetaRhoAccumWithBlurBorder, thetaRhoAccumWithBlurBorder,
          0, 0,
          houghBlurRhoWindowHalfSize, houghBlurRhoWindowHalfSize,
          cv::BORDER_CONSTANT, cv::Scalar(0));

      // TODO: 000 remove
      if (false) {
        cv::Mat accum;
        double maxAccum;
        minMaxLoc(thetaRhoAccumWithBlurBorder, NULL, &maxAccum);
        thetaRhoAccumWithBlurBorder.convertTo(accum, CV_8UC1, 255.0/maxAccum, 0);
        imshow("accum_preblur", accum);
        cout << "thetaRhoAccumWithBlurBorder: " << thetaRhoAccumWithBlurBorder.rows << " x " << thetaRhoAccumWithBlurBorder.cols << endl;
        cv::waitKey(10);
      }

      // Apply Gaussian blur to Hough accumulator space
      cv::GaussianBlur(thetaRhoAccumWithBlurBorder, thetaRhoAccumWithBlurBorder,
          cv::Size(houghBlurRhoWindowHalfSize*2+1, houghBlurThetaWindowHalfSize*2+1),
          houghBlurRhoStd, houghBlurThetaStd,
          cv::BORDER_DEFAULT);
      thetaRhoAccumBlurred = thetaRhoAccumWithBlurBorder(
          cv::Range(houghBlurThetaWindowHalfSize, houghBlurThetaWindowHalfSize + numTheta),
          cv::Range(houghBlurRhoWindowHalfSize, houghBlurRhoWindowHalfSize + numRho));
    } else {
      thetaRhoAccumBlurred = thetaRhoAccum;
    }

    if (false) {
      cv::Mat accum;
      double maxAccum;
      minMaxLoc(thetaRhoAccumBlurred, NULL, &maxAccum);
      thetaRhoAccumBlurred.convertTo(accum, CV_8UC1, 255.0/maxAccum, 0);
      imshow("accum_postblur", accum);
      cout << "thetaRhoAccumBlurred: " << thetaRhoAccumBlurred.rows << " x " << thetaRhoAccumBlurred.cols << endl;
      cv::waitKey(10);
    }

    // Expand the Hough accumulator space by adding zero-borders for rho
    // (horiz axis) and by adding mirrored-wrapped-around-borders for theta
    //
    // NOTE: need to add additional rows and columns to make the expanded matrix
    //       have width and height that are integer multiples of the
    //       corresponding block width and height
    cv::Mat thetaRhoAccumBlurredWithNMSBorder, thetaRhoAccumBlurredBottomFlipped, thetaRhoAccumBlurredTopFlipped;
    cv::Mat thetaRhoAccumBlurredBottom =
        thetaRhoAccumBlurred.rowRange(thetaRhoAccum.rows - houghNMSThetaWindowHalfSize, thetaRhoAccum.rows);
    cv::flip(thetaRhoAccumBlurredBottom, thetaRhoAccumBlurredBottomFlipped, 1);
    cv::Mat thetaRhoAccumBlurredTop =
        thetaRhoAccumBlurred.rowRange(0, houghNMSThetaWindowHalfSize + (houghNMSThetaNumWindows*houghNMSThetaWindowSize - numTheta));
    cv::flip(thetaRhoAccumBlurredTop, thetaRhoAccumBlurredTopFlipped, 1);
    cv::vconcat(thetaRhoAccumBlurredBottomFlipped, thetaRhoAccumBlurred, thetaRhoAccumBlurredWithNMSBorder);
    cv::vconcat(thetaRhoAccumBlurredWithNMSBorder, thetaRhoAccumBlurredTopFlipped, thetaRhoAccumBlurredWithNMSBorder);
    cv::copyMakeBorder(thetaRhoAccumBlurredWithNMSBorder, thetaRhoAccumBlurredWithNMSBorder,
        0, 0,
        houghNMSRhoWindowHalfSize,
        houghNMSRhoWindowHalfSize + (houghNMSRhoNumWindows*houghNMSRhoWindowSize - numRho),
        cv::BORDER_CONSTANT, cv::Scalar(-1.0));

    // TODO: 000 remove
    if (false) {
      cv::Mat accum;
      double maxAccum;
      minMaxLoc(thetaRhoAccumBlurredWithNMSBorder, NULL, &maxAccum);
      thetaRhoAccumBlurredWithNMSBorder.convertTo(accum, CV_8UC1, 255.0/maxAccum, 0);
      imshow("accum_prenms", accum);
      cout << "thetaRhoAccumBlurredWithNMSBorder: " << thetaRhoAccumBlurredWithNMSBorder.rows << " x " << thetaRhoAccumBlurredWithNMSBorder.cols << endl;
      cv::waitKey(10);
    }

    // TODO: 0 find local maxima using 2-D block-based non-maxima suppression (NMS)
    std::vector<cv::Point> localMaxima;
    if (true) {
      // Iterate over non-overlapping local block neighbourhoods
      int thetaI, rhoI;
      double blockMaxVal;
      cv::Point blockMaxPtLocal, blockMaxPt, neighMaxPtLocal;
      cv::Range thetaBlockRange, rhoBlockRange, thetaNeighRange, rhoNeighRange;
      for (thetaI = 0; thetaI < numTheta; thetaI += houghNMSThetaWindowSize) {
        thetaBlockRange = cv::Range(thetaI + houghNMSThetaWindowHalfSize,
            thetaI + houghNMSThetaWindowHalfSize + houghNMSThetaWindowSize);

        for (rhoI = 0; rhoI < numRho; rhoI += houghNMSRhoWindowSize) {
          // Compute local maximum within block
          rhoBlockRange = cv::Range(rhoI + houghNMSRhoWindowHalfSize,
              rhoI + houghNMSRhoWindowHalfSize + houghNMSRhoWindowSize);
          cv::minMaxLoc(thetaRhoAccumBlurredWithNMSBorder(thetaBlockRange, rhoBlockRange),
              NULL, &blockMaxVal,
              NULL, &blockMaxPtLocal);
          blockMaxPt = blockMaxPtLocal +
              cv::Point(rhoI + houghNMSRhoWindowHalfSize, thetaI + houghNMSThetaWindowHalfSize);

          // Compute local maximum within neighbour centered around block max
          thetaNeighRange = cv::Range(blockMaxPt.y - houghNMSThetaWindowHalfSize,
              blockMaxPt.y - houghNMSThetaWindowHalfSize + houghNMSThetaWindowSize);
          rhoNeighRange = cv::Range(blockMaxPt.x - houghNMSRhoWindowHalfSize,
              blockMaxPt.x - houghNMSRhoWindowHalfSize + houghNMSRhoWindowSize);
          cv::minMaxLoc(thetaRhoAccumBlurredWithNMSBorder(thetaNeighRange, rhoNeighRange),
              NULL, NULL,
              NULL, &neighMaxPtLocal);

          // If neighbour's maximum is its center, then found true local maximum within window
          if (neighMaxPtLocal.x == houghNMSRhoWindowHalfSize &&
              neighMaxPtLocal.y == houghNMSThetaWindowHalfSize) {
            // Subtract away border pixels
            blockMaxPt -= cv::Point(houghNMSRhoWindowHalfSize, houghNMSThetaWindowHalfSize);
            if (blockMaxPt.x >= 0 && blockMaxPt.x < thetaRhoAccum.cols &&
                blockMaxPt.y >= 0 && blockMaxPt.y < thetaRhoAccum.rows) {
              localMaxima.push_back(blockMaxPt);

              //cout << "Found new local maximum @ rho/theta (" << blockMaxPt.x << ", " << blockMaxPt.y << ")" << endl;
            }
          }
        }
      }
    }

    // TODO: 00000 remove
    if (true) {
      //cout << "LOCAL MAXIMA VIA NMS:" << endl;
      unsigned int falseCounts = 0;

      std::vector<cv::Point> filteredLocalMaxima;
      for (const cv::Point& localMax: localMaxima) {


        if (localMax.y < 0 || localMax.y >= thetaRhoAccumBlurred.rows ||
            localMax.x < 0 || localMax.x >= thetaRhoAccumBlurred.cols) {
          cout << "SEGFAULT INCOMING!" << endl;
          cout << "- localMax.x (rho): " << localMax.x << endl;
          cout << "- thetaRhoAccumBlurred.cols: " << thetaRhoAccumBlurred.cols << endl;
          cout << "- localMax.y (theta): " << localMax.y << endl;
          cout << "- thetaRhoAccumBlurred.rows (rho): " << thetaRhoAccumBlurred.rows << endl;
          cout << "- thetaRhoAccum: " << thetaRhoAccum.rows << " x " << thetaRhoAccum.cols << endl;
        }

        if (thetaRhoAccumBlurred.at<double>(localMax.y, localMax.x) <= 3) {
          falseCounts += 1;
        } else {
          filteredLocalMaxima.push_back(localMax);

          if (false) {
            cout << "- rho/theta (" << localMax.x << ", " << localMax.y << ")";
            cout << ": " << thetaRhoAccumBlurred.at<double>(localMax.y, localMax.x) << endl;
          }
        }
      }

      cout << "Found: " << localMaxima.size() - falseCounts << " local maxima + " << falseCounts << " sub-thresholded FP";

      localMaxima.swap(filteredLocalMaxima);
    }

    // HACK: scan through the accumulator to find the maximum and its location
    double maxTheta = 0, maxRho = 0;
    if (false) {
      cv::Point maxAccumLoc;
      minMaxLoc(thetaRhoAccum, NULL, NULL, NULL, &maxAccumLoc);
      maxTheta = houghThetaRes * maxAccumLoc.y;
      maxRho = houghRhoRes * maxAccumLoc.x - rhoHalfRange;
    }


    // Accumulate all edgel positions into column-stacked 2-channel vector
    cv::Mat edgelsXY(numEdgels, 1, CV_64FC2);
    double* edgelsXYPtr = (double*) edgelsXY.data;
    int y, x;
    for (y = 0; y < grayImg.rows; y++) {
      for (x = 0; x < grayImg.cols; x++) {
        if (edgeImg.at<unsigned char>(y, x) > 0) {
          *edgelsXYPtr = x;
          edgelsXYPtr++;
          *edgelsXYPtr = y;
          edgelsXYPtr++;
        }
      }
    }

    // TODO: 0 Compute connected line segments based on location of maxima
    std::vector<cv::Point2d> lineSegments; // stored as (segmentStart, segmentEnd) pairs of points
    //if (true) {
    for (const cv::Point& localMax: localMaxima) {
      maxTheta = houghThetaRes * localMax.y;
      maxRho = houghRhoRes * localMax.x - rhoHalfRange;

      // Translate+rotate edgels into space where line is aligned with y axis
      double xLine = maxRho * cos(maxTheta);
      double yLine = maxRho * sin(maxTheta);
      cv::Mat edgelsXYRot;
      cv::add(edgelsXY, cv::Scalar(-xLine, -yLine), edgelsXYRot, cv::noArray(), CV_64FC2);
      cv::Mat R = (cv::Mat_<double>(2, 2) << cos(maxTheta), sin(maxTheta), -sin(maxTheta), cos(maxTheta));
      edgelsXYRot = edgelsXYRot.reshape(1, numEdgels).t();
      edgelsXYRot = (R * edgelsXYRot).t();
      std::vector<cv::Point2d> edgelsPosInLine;
      int edgelI;

      if (false) {
        cout << endl << "For line @ rho,theta: (" << maxRho << ", " << maxTheta << "):" << endl;
        cout << "  - added to " << -xLine << ", " << -yLine << endl;
        cout << "  - rotated by " << maxTheta*radian << " (" << cos(maxTheta) << ", " << sin(maxTheta) << ", " << -sin(maxTheta) << ", " << cos(maxTheta) << ")" << endl;
      }

      for (edgelI = 0; edgelI < numEdgels; edgelI++) {
        const cv::Point2d currPt = edgelsXY.at<cv::Point2d>(edgelI, 0);
        const cv::Point2d currPtRot = edgelsXYRot.at<cv::Point2d>(edgelI, 0);
        if (fabs(currPtRot.x) <= houghMaxDistToLine &&
            angularDist(atan2(dyImg.at<signed short>(currPt.y, currPt.x), dxImg.at<signed short>(currPt.y, currPt.x)),
            maxTheta, pi) <= houghEdgelThetaMargin) {
          //edgelsPosInLine.push_back(edgelsXYRot.at<double>(edgelI, 1));
          edgelsPosInLine.push_back(currPtRot);
        }

        if (false) {
          if (fabs(currPtRot.x) <= houghMaxDistToLine &&
              angularDist(atan2(dyImg.at<signed short>(currPt.y, currPt.x), dxImg.at<signed short>(currPt.y, currPt.x)),
              maxTheta, pi) <= houghEdgelThetaMargin) {
            cout << "OK point (" << currPt.x << ", " << currPt.y << ") -> T+R (" <<
                currPtRot.x << ", " << currPtRot.y << ")" << endl;
          } else {
            if (true) {
              cout << "   point (" << currPt.x << ", " << currPt.y << ") -> T+R (" <<
                  currPtRot.x << ", " << currPtRot.y << ")" << endl;
              cout << "   - gradiant theta: " << atan2(dyImg.at<signed short>(currPt.y, currPt.x), dxImg.at<signed short>(currPt.y, currPt.x)) <<
                  ", maxTheta: " << maxTheta << ", angDist: " <<
                  angularDist(atan2(dyImg.at<signed short>(currPt.y, currPt.x), dxImg.at<signed short>(currPt.y, currPt.x)), maxTheta, pi) <<
                  " < " << houghEdgelThetaMargin << endl;
              cout << "   - first condition: " << (fabs(currPtRot.x) <= houghMaxDistToLine) << endl;
              cout << "     - houghMaxDistToLine: " << houghMaxDistToLine << endl;
              cout << "   - second condition: " << (angularDist(atan2(dyImg.at<signed short>(currPt.y, currPt.x), dxImg.at<signed short>(currPt.y, currPt.x)),
                  maxTheta, pi) <= houghEdgelThetaMargin) << endl;
            }
          }
        }

      }
      std::sort(edgelsPosInLine.begin(), edgelsPosInLine.end(), lessThanInY);



      if (false) {
        cout << "AFTER sorting, pts:" << endl;
        for (const cv::Point2d& xy: edgelsPosInLine) {
          cout << "-- point (" << xy.x << ", " << xy.y << ")" << endl;
        }
      }


      // Determine line segments based on minimum gap distance between projected points
      std::vector<cv::Point2d> currLineSegments;
      bool segmentStart = true;
      std::vector<cv::Point2d>::iterator currPt = edgelsPosInLine.begin();
      std::vector<cv::Point2d>::iterator lastPt = edgelsPosInLine.end();
      for (; currPt < edgelsPosInLine.end(); lastPt = currPt, currPt++) {
        if (segmentStart) {
          currLineSegments.push_back(*currPt);
          segmentStart = false;
        } else {
          if (currPt->y - lastPt->y > MAX_PX_GAP) { // New line segment
          //if ((currPt->x - lastPt->x)*(currPt->x - lastPt->x) + (currPt->y - lastPt->y)*(currPt->y - lastPt->y) > 1.5) { // New line segment
            if (false) {
              cout << "Adding new line because:" << endl;
              cout << "  - currPt->y: " << currPt->y << endl;
              cout << "  - lastPt->y: " << lastPt->y << endl;
              cout << "  - dist: " << currPt->y - lastPt->y << " > " << MAX_PX_GAP << endl;
            }

            currLineSegments.push_back(*lastPt);
            currLineSegments.push_back(*currPt);
          } // else, still in current line segment
        }
      }
      if (!segmentStart) { // Terminate last line segment
        currLineSegments.push_back(*lastPt);
      }



      if (false) {
        cout << "line segments:" << endl;
        for (const cv::Point2d& xy: currLineSegments) {
          cout << "-- (" << xy.x << ", " << xy.y << ")" << endl;
        }
      }




      // Prune line segments that are too short
      if (currLineSegments.size() >= 2) {
        segmentStart = true;
        std::vector<cv::Point2d> longLineSegments;
        for (currPt = currLineSegments.begin(); currPt < currLineSegments.end();
            lastPt = currPt, currPt++, segmentStart = !segmentStart) {
          if (!segmentStart) {
            if (currPt->y - lastPt->y >= houghMinSegmentLength*MAX_PX_GAP) { // *MAX_PX_GAP to account for worst-case slanted line segment
              longLineSegments.push_back(*lastPt);
              longLineSegments.push_back(*currPt);
            }
          }
        }
        currLineSegments.swap(longLineSegments);
      }

      if (true) {
      //if (currLineSegments.size() >= 4) {
        // De-gap line segments
        std::vector<cv::Point2d> degappedLineSegments;
        segmentStart = true;
        for (currPt = currLineSegments.begin(); currPt < currLineSegments.end();
            lastPt = currPt, currPt++, segmentStart = !segmentStart) {
          if (!segmentStart) {
            if (degappedLineSegments.empty()) {
              degappedLineSegments.push_back(*lastPt);
              degappedLineSegments.push_back(*currPt);
            } else {
              if (lastPt->y - degappedLineSegments.back().y <= houghMaxSegmentGap*MAX_PX_GAP) { // *MAX_PX_GAP to account for worst-case slanted line segment
                degappedLineSegments.back() = *currPt;
              } else {
                degappedLineSegments.push_back(*lastPt);
                degappedLineSegments.push_back(*currPt);
              }
            }
          }
        }
        degappedLineSegments.swap(currLineSegments);
      }

      // For each line segment, perform line fit and store updated line
      if (currLineSegments.size() > 0) {
        if (false) {
          int segmentI, ptI;
          for (segmentI = 0; segmentI < currLineSegments.size(); segmentI += 2) {
            cv::Point2d& startPt = currLineSegments[segmentI];
            cv::Point2d& endPt = currLineSegments[segmentI + 1];

            std::vector<cv::Point2d> _segmentY;
            std::vector<double> _segmentX;
            for (ptI = 0; ptI < edgelsPosInLine.size(); ptI++) {
              if (edgelsPosInLine[ptI].y < startPt.y) {
                continue;
              } else if (edgelsPosInLine[ptI].y <= endPt.y) {
                _segmentY.push_back(cv::Point2d(1, edgelsPosInLine[ptI].y));
                _segmentX.push_back(edgelsPosInLine[ptI].x);
              }
              if (edgelsPosInLine[ptI].y >= endPt.y) {
                break;
              }
            }

            cv::Mat segmentY(_segmentY, false);
            cv::Mat segmentX(_segmentX, false);

            if (false) {
              cout << "For segment: (" << startPt.x << ", " << startPt.y << ") -> (" << endPt.x << ", " << endPt.y << "):" << endl;
              for (ptI = 0; ptI < segmentY.rows; ptI++) {
                //cout << "- " << segmentY.at<double>(ptI, 0) << " + (" << segmentX.at<double>(ptI, 0) << ", " << segmentY.at<double>(ptI, 1) << ")" << endl;
                cout << "  " << segmentX.at<double>(ptI, 0) << ", " << segmentY.at<double>(ptI, 1) << "," << endl;
              }
            }

            segmentY = segmentY.reshape(1, segmentY.rows);

            cv::Mat params = (segmentY.t()*segmentY).inv()*segmentY.t()*segmentX;

            double* paramsPtr = (double*) params.data;
            double b = *paramsPtr;
            paramsPtr++;
            double a = *paramsPtr;

            if (false) {
              cout << "+ Params: x = " << a << " * y + " << b << endl;
              for (ptI = 0; ptI < segmentY.rows; ptI++) {
                cout << "- (" << segmentX.at<double>(ptI, 0) << ", " << segmentY.at<double>(ptI, 1) << ") -> " << segmentY.at<double>(ptI, 1)*a + b << endl;
              }
            }

            startPt.x = a*startPt.y + b;
            endPt.x = a*endPt.y + b;
          }

          if (false) {
            cout << "revised line segments:" << endl;
            for (const cv::Point2d& xy: currLineSegments) {
              cout << "-- (" << xy.x << ", " << xy.y << ")" << endl;
            }
          }
        }


        cv::Mat segmentsXYRot(currLineSegments.size(), 2, CV_64FC1);
        double* segmentsXYRotPtr = (double*) segmentsXYRot.data;
        for (const cv::Point2d& xy: currLineSegments) {
          *segmentsXYRotPtr = xy.x;
          segmentsXYRotPtr++;
          *segmentsXYRotPtr = xy.y;
          segmentsXYRotPtr++;
        }
        cv::Mat segmentsXY = segmentsXYRot*R;
        int segmentXYI;
        double* segmentsXYPtr = (double*) segmentsXY.data;
        for (segmentXYI = 0; segmentXYI < segmentsXY.rows; segmentXYI += 1) {
          *segmentsXYPtr += xLine;
          segmentsXYPtr++;
          *segmentsXYPtr += yLine;
          segmentsXYPtr++;
          lineSegments.push_back(segmentsXY.at<cv::Point2d>(segmentXYI, 0));
        }




        if (false) {
          cout << "segmentsXY (final): (" << segmentsXY.rows << " x " << segmentsXY.cols << ")" << endl;
          int r, c;
          for (r = 0; r < segmentsXY.rows; r++) {
            for (c = 0; c < segmentsXY.cols; c++) {
              cout << segmentsXY.at<double>(r, c) << " ";
            }
            cout << endl;
          }
          cout << endl;
        }
      }
    }

    // Draw max line over image
    if (false) { // DRAW ENTIRE LINE
      int x0, y0, x1, y1;
      x0 = -grayImg.cols;
      x1 = 2*grayImg.cols;
      y0 = round((maxRho - cos(maxTheta)*x0)/sin(maxTheta));
      y1 = round((maxRho - cos(maxTheta)*x1)/sin(maxTheta));
      cv::Point p0(x0, y0);
      cv::Point p1(x1, y1);
      cv::line(overlaidImg, p0, p1, CV_RGB(0, 0, 255), 1);
    } else { // Draw line segments
      int segmentI;
      for (segmentI = 0; segmentI < (int) lineSegments.size(); segmentI += 2) {
        cv::line(overlaidImg, lineSegments[segmentI], lineSegments[segmentI + 1], CV_RGB(255, 255, 0), 3);
      }
      for (segmentI = 0; segmentI < (int) lineSegments.size(); segmentI += 2) {
        cv::line(overlaidImg, lineSegments[segmentI], lineSegments[segmentI + 1], CV_RGB(255, 0, 0), 1);
      }
      for (segmentI = 0; segmentI < (int) lineSegments.size(); segmentI += 2) {
        cv::circle(overlaidImg, lineSegments[segmentI], 2, CV_RGB(0, 0, 255));
        cv::circle(overlaidImg, lineSegments[segmentI + 1], 2, CV_RGB(0, 255, 255));
      }
    }

    cout << " -> " << lineSegments.size()/2 << " line segments" << endl;

    // TODO: 000 remove
    //imshow("overlaidImg", overlaidImg);
  }

  return overlaidImg;

  /*
  cv::AutoBuffer<int> _accum, _sort_buf;
  cv::AutoBuffer<float> _tabSin, _tabCos;

  const uchar* image;
  int step, width, height;
  int numangle, numrho;
  int total = 0;
  int i, j;
  float irho = 1 / rho;
  double scale;

  image = img->data.ptr;
  step = img->step;
  width = img->cols;
  height = img->rows;

  numangle = cvRound(CV_PI / theta);
  numrho = cvRound(((width + height) * 2 + 1) / rho);

  _accum.allocate((numangle+2) * (numrho+2));
  _sort_buf.allocate(numangle * numrho);
  _tabSin.allocate(numangle);
  _tabCos.allocate(numangle);
  int *accum = _accum, *sort_buf = _sort_buf;
  float *tabSin = _tabSin, *tabCos = _tabCos;

  memset( accum, 0, sizeof(accum[0]) * (numangle+2) * (numrho+2) );

  float ang = 0;
  for(int n = 0; n < numangle; ang += theta, n++ )
  {
      tabSin[n] = (float)(sin((double)ang) * irho);
      tabCos[n] = (float)(cos((double)ang) * irho);
  }

  // stage 1. fill accumulator
  for( i = 0; i < height; i++ )
      for( j = 0; j < width; j++ )
      {
          if( image[i * step + j] != 0 )
              for(int n = 0; n < numangle; n++ )
              {
                  int r = cvRound( j * tabCos[n] + i * tabSin[n] );
                  r += (numrho - 1) / 2;
                  accum[(n+1) * (numrho+2) + r+1]++;
              }
      }

  // stage 2. find local maximums
  for(int r = 0; r < numrho; r++ )
      for(int n = 0; n < numangle; n++ )
      {
          int base = (n+1) * (numrho+2) + r+1;
          if( accum[base] > threshold &&
              accum[base] > accum[base - 1] && accum[base] >= accum[base + 1] &&
              accum[base] > accum[base - numrho - 2] && accum[base] >= accum[base + numrho + 2] )
              sort_buf[total++] = base;
      }

  // stage 3. sort the detected lines by accumulator value
  icvHoughSortDescent32s( sort_buf, total, accum );

  // stage 4. store the first min(total,linesMax) lines to the output buffer
  linesMax = MIN(linesMax, total);
  scale = 1./(numrho+2);
  for( i = 0; i < linesMax; i++ )
  {
      CvLinePolar line;
      int idx = sort_buf[i];
      int n = cvFloor(idx*scale) - 1;
      int r = idx - (n+1)*(numrho+2) - 1;
      line.rho = (r - (numrho - 1)*0.5f) * rho;
      line.angle = n * theta;
      cvSeqPush( lines, &line );
  }
  */
};


int main(int argc, char** argv) {
  try {
    // Initialize parameters
    int sobelThreshHigh = 100;
    int sobelThreshLow = 30;
    int sobelBlurWidth = 3;

    double houghRhoRes = 1.0;
    double houghThetaRes = two_pi/360;
    double houghEdgelThetaMargin = (10.0)*degree;
    double houghBlurRhoWidth = 5.0;
    double houghBlurThetaWidth = 7*degree;
    double houghNMSRhoWidth = houghRhoRes*9;
    double houghNMSThetaWidth = houghThetaRes*13;
    double houghMaxDistToLine = 5; // NOTE: should be dependent on image size
    double houghMinSegmentLength = 5;
    double houghMaxSegmentGap = 5;
    /*
    double houghRhoRes = 2.0;
    double houghThetaRes = two_pi/360*2;
    double houghEdgelThetaMargin = (15.0)*degree;
    double houghBlurRhoWidth = 5.0;
    double houghBlurThetaWidth = 7*degree;
    //houghBlurRhoWidth = 0; houghBlurThetaWidth = 0;
    double houghNMSRhoWidth = houghRhoRes*5;
    double houghNMSThetaWidth = houghThetaRes*9;
    //houghNMSRhoWidth = houghRhoRes*3; houghNMSThetaWidth = houghThetaRes*3;
    double houghMaxDistToLine = 3; // NOTE: should be dependent on image size
    double houghMinSegmentLength = 5;
    double houghMaxSegmentGap = 2;
    */

    // Initialize internal variables
    cv::Mat bgrImg, grayImg, blurredImg, edgeImg, linesImg;
    std::vector<cv::Vec4i> lines;
#ifdef SAVE_IMAGES_FROM
    int imgid = 0;
    char* filename = (char*) calloc(1000, sizeof(char));
#endif
#ifdef ENABLE_PROFILER
    Profiler durationProf, rateProf;
    //time_point<system_clock> lastProfTime = system_clock::now();
    time_point<system_clock> currTime;
    duration<double> profTD;
#endif

//#define USE_CAMERA
#ifdef USE_CAMERA
    // Open camera
    VideoCapture cam(0);
    if (!cam.isOpened()) {
      cerr << "OpenCV did not detect any cameras." << endl;
      return EXIT_FAILURE;
    }


    // Manage OpenCV windows
    namedWindow("edgels", CV_GUI_EXPANDED);
    namedWindow("overlaidImg", CV_GUI_EXPANDED);

    //namedWindow("edge", CV_GUI_EXPANDED);
    //createTrackbar("sobelThreshLow", "edge", &sobelThreshLow, 255);
    //createTrackbar("sobelThreshHigh", "edge", &sobelThreshHigh, 255);

    //namedWindow("lines", CV_GUI_EXPANDED);
    //createTrackbar("threshold", "lines", &threshold, 255);
    //createTrackbar("linbinratio", "lines", &linbinratio, 100);
    //createTrackbar("angbinratio", "lines", &angbinratio, 100);


    // Main camera pipeline
    bool alive = true;
    while (alive) {
#ifdef ENABLE_PROFILER
      rateProf.try_toc();
      rateProf.tic();
      durationProf.tic();
#endif

      // Obtain grayscale image
      cam >> bgrImg;
      cvtColor(bgrImg, grayImg, CV_BGR2GRAY);
      //imshow("source", grayImg);

//#define OLD_CODE
#ifdef OLD_CODE
      int threshold = 50, linbinratio = 100, angbinratio = 100;

      // Detect lines
      blur(grayImg, edgeImg, Size(sobelBlurWidth, sobelBlurWidth));
      Canny(edgeImg, edgeImg, sobelThreshLow, sobelThreshHigh, sobelBlurWidth); // args: in, out, low_thresh, high_thresh, gauss_blur
      cout << "pre detect line" << flush << endl;
      cv::HoughLinesP(edgeImg, lines, max(linbinratio/10.0, 1.0), max(angbinratio/100.0*CV_PI/180, 1/100.0*CV_PI/180), threshold, 10*1.5, 10*1.5/3);
      //cv::HoughLines(edgeImg, lines, max(linbinratio/10.0, 1.0), max(angbinratio/100.0*CV_PI/180, 1/100.0*CV_PI/180), threshold);
      cout << "post detect line" << flush << endl;
      cvtColor(edgeImg, linesImg, CV_GRAY2BGR);
      for (unsigned int i = 0; i < lines.size(); i++) {
        line(linesImg, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(0,255,0), 2, 8);
      }
      imshow("overlaidImg", linesImg);
#else
      cv::Mat overlaidImg = detectLines(grayImg,
          sobelThreshHigh, sobelThreshLow, sobelBlurWidth,
          houghRhoRes, houghThetaRes,
          houghEdgelThetaMargin,
          houghBlurRhoWidth, houghBlurThetaWidth,
          houghNMSRhoWidth, houghNMSThetaWidth,
          houghMaxDistToLine,
          houghMinSegmentLength, houghMaxSegmentGap);
      imshow("overlaidImg", overlaidImg);
#endif

#ifdef SAVE_IMAGES_FROM
      sprintf(filename, "img%05d.jpg", imgid++);
      imwrite(filename, SAVE_IMAGES_FROM);
      cout << "Wrote to " << filename << endl;
#endif

#ifdef ENABLE_PROFILER
      durationProf.toc();

      currTime = system_clock::now();
      profTD = currTime - lastProfTime;
      if (profTD.count() > 1) {
        cout << "Pipeline Duration: " << durationProf.getStatsString() << endl;
        cout << "Pipeline Rate: " << rateProf.getStatsString() << endl;
        lastProfTime = currTime;
      }
#endif

      // Process displays
      if (waitKey(30) >= 0) { break; }
    }
#else
    namedWindow("source", CV_GUI_EXPANDED);
    namedWindow("edgels", CV_GUI_EXPANDED);
    namedWindow("accum", CV_GUI_EXPANDED);
    //namedWindow("accum_postblur", CV_GUI_EXPANDED);
    //namedWindow("accum_prenms", CV_GUI_EXPANDED);
    namedWindow("overlaidImg", CV_GUI_EXPANDED);

    cv::Mat testImg;
    int width = 300;
    int height = 200;
    cv::Point p0(width/2, height/2);
    cv::Point p1, p2;
    cv::Point poly[1][4];
    const cv::Point* polyPtr[1] = {poly[0]};
    poly[0][2] = cv::Point(0, height);
    poly[0][3] = cv::Point(0, 0);
    int numPtsInPoly[] = {4};

    double deg;
    int c;
    //for (deg = -56; deg < 57; deg += 15) {
    for (deg = -11; deg < 180-11; deg += 15) {
    //for (deg = 80; deg < 100; deg += 1) {
      testImg = cv::Mat::zeros(height, width, CV_8UC1);

      if (false) { // Draw edge
        p1 = cv::Point(double(width + tan(deg*degree)*height)/2.0, 0);
        p2 = p0 - (p1 - p0);
        poly[0][0] = p1;
        poly[0][1] = p2;
        //cv::line(testImg, p1, p2, cv::Scalar(255), width, CV_AA);
        cv::fillPoly(testImg, polyPtr, numPtsInPoly, 1, Scalar(255), 8);
        cout << "=== Source img w/ line at " << deg << " degrees" << endl;
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

        cout << "=== Source img w/ box rotated at " << deg << " degrees" << endl;
      }

      imshow("source", testImg);

      cv::Mat overlaidImg = detectLines(testImg,
          sobelThreshHigh, sobelThreshLow, sobelBlurWidth,
          houghRhoRes, houghThetaRes,
          houghEdgelThetaMargin,
          houghBlurRhoWidth, houghBlurThetaWidth,
          houghNMSRhoWidth, houghNMSThetaWidth,
          houghMaxDistToLine,
          houghMinSegmentLength, houghMaxSegmentGap);
      imshow("overlaidImg", overlaidImg);
      cout << "recall source @ " << wrapAngle(deg, 180) << " degrees" << endl;

      c = waitKey();
      if ((c & 0x0FF) == 'x') {
        break;
      }
    }
#endif
  } catch (const Exception& err) {
    cout << "CV Exception: " << err.what() << endl;
  }

  return EXIT_SUCCESS;
};
