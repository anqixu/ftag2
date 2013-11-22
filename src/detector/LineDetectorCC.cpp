#include "detector/FTag2Detector.hpp"
#include "common/BaseCV.hpp"
#include <cmath>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>


struct Segment {
  // Elements populated by grouplearSegmentsInCC()
  std::vector<cv::Point2i> points; // NOTE: points are NOT assumed to be sorted in any way
  double minAngleSlack, maxAngleSlack, minAngle, maxAngle;

  // Elements populated by computeSegmentEndpoints()
  cv::Vec4f line;
  cv::Point2i endpointA, endpointB;
};


/**
 * Identifies 8-connected components of non-zero elements in edgel image
 */
std::vector< std::vector<cv::Point2i> > identifyEdgelCCs(cv::Mat edgelImg) {
  const unsigned int imWidth = edgelImg.cols;
  const unsigned int imHeight = edgelImg.rows;

  // Assign each pixel into their connected component ID
  cv::Mat imageIDs;
  std::vector<unsigned int> parentIDs;
  std::vector<unsigned char> setLabels;
  BaseCV::computeDisjointSets(edgelImg, imageIDs, parentIDs, setLabels, true); // true: 8-connected
  BaseCV::condenseDisjointSetLabels(imageIDs, parentIDs, setLabels, 1); // Enforce 0-step parent ID invariance

  // Extract non-zero-labelled connected components
  std::vector<int> edgelCCIDs(parentIDs.size(), -1);
  unsigned int numEdgelCCs = 0;
  unsigned int* imageIDRow = NULL;
  unsigned int currParentID;
  for (currParentID = 0; currParentID < parentIDs.size(); currParentID++) {
    if (setLabels[currParentID] != 0) { // condenseDisjointSetLabels() also enforced (i == parentIDs[i])
      edgelCCIDs[currParentID] = numEdgelCCs;
      numEdgelCCs += 1;
    }
  }

  // Identify coordinates of all edgels within each connected component
  unsigned int currY, currX;
  int currSetID;
  std::vector< std::vector<cv::Point2i> > edgelCCs(numEdgelCCs);
  for (currY = 0; currY < imHeight; currY++) {
    imageIDRow = imageIDs.ptr<unsigned int>(currY);
    for (currX = 0; currX < imWidth; currX++) {
      currSetID = edgelCCIDs[imageIDRow[currX]]; // No need to access parentIDs[] since enforced 0-step parent invariance above
      if (currSetID >= 0) {
        edgelCCs[currSetID].push_back(cv::Point2i(currX, currY));
      }
    }
  }

  return edgelCCs;
};


/**
 * Groups edgel elements in each connected component into linear segments, where each
 * grouping has a gradient direction range of at most 2*angleMarginRad.
 *
 * This is implemented using a depth-first iterative floodfill algorithm, over the given
 * edgel image. Connected edgels are included into an existing segment if their gradient
 * direction is within the current slack angle range. The slack angle range starts at
 * [firstDir - 2*angleMarginRad, firstDir + 2*angleMarginDir] and iteratively reduces
 * to a minimum of [minAngle, minAngle + 2*angleMarginRad] as more edgels are included.
 */
std::list<Segment> groupLinearSegmentsInCC(const std::vector< std::vector<cv::Point2i> > edgelCCs,
    const cv::Mat edgelImg, const cv::Mat dxImg, const cv::Mat dyImg, double angleMarginRad) {
  const int imWidth = edgelImg.cols;
  const int imHeight = edgelImg.rows;
  cv::Mat remEdgelImg = edgelImg.clone();
  std::list<Segment> segments;
  std::vector<cv::Point2i> floodfillStack;
  cv::Point2i currPt;
  double currAngle, slackAngle;
  bool includeNeighbour;
  int neighX, neighY;
  int offsetCount;
  for (const std::vector<cv::Point2i>& edgelCC: edgelCCs) { // for each connected component
    for (const cv::Point2i& initPt: edgelCC) { // for each edgel inside the CC
      // Floodfill only if edgel has not been filled yet
      if (remEdgelImg.at<unsigned char>(initPt.y, initPt.x) == 0) {
        continue;
      }

      // Start new segment of colinear edgels
      Segment currSegment;
      floodfillStack.push_back(initPt);
      currSegment.points.push_back(initPt);
      remEdgelImg.at<unsigned char>(initPt.y, initPt.x) = 0;

      // Identify [minAngleSlack, maxAngleSlack] range in [0, 4*pi) range (assuming margin is < pi)
      currAngle = std::atan2(dyImg.at<short>(initPt.y, initPt.x), dxImg.at<short>(initPt.y, initPt.x));
      currSegment.minAngleSlack = currAngle - 2*angleMarginRad;
      if (currSegment.minAngleSlack < 0) {
        currSegment.minAngleSlack = vc_math::wrapAngle(currSegment.minAngleSlack, vc_math::two_pi);
      }
      currAngle = currSegment.minAngleSlack + 2*angleMarginRad;
      currSegment.maxAngleSlack = currAngle + 2*angleMarginRad;
      currSegment.minAngle = currAngle;
      currSegment.maxAngle = currAngle;

      // Apply depth-first floodfill while reducing [minAngleSlack, maxAngleSlack] range
      while (!floodfillStack.empty()) {
        currPt = floodfillStack.back();
        floodfillStack.pop_back();

        // Iteratively include 8-connected edgels into segment
        for (neighY = std::max(currPt.y - 1, 0); neighY <= std::min(currPt.y + 1, imHeight - 1); neighY++) {
          for (neighX = std::max(currPt.x - 1, 0); neighX <= std::min(currPt.x + 1, imWidth - 1); neighX++) {
            if ((neighY == currPt.y && neighX == currPt.x) || (remEdgelImg.at<unsigned char>(neighY, neighX) == 0)) {
              continue;
            }

            // Determine whether current edgel's gradient direction is within slack angle range
            // NOTE: opposite gradient directions g and (g + pi) are grouped together
            // NOTE: since slack angle range may be beyond [0, 2*pi) range, need to check for gradient directions + 2*pi also
            includeNeighbour = false;
            currAngle = std::atan2(dyImg.at<short>(neighY, neighX), dxImg.at<short>(neighY, neighX));
            if (currAngle < 0) currAngle += vc_math::two_pi;
            if (currAngle >= vc_math::pi) currAngle -= vc_math::pi;
            for (offsetCount = 0; offsetCount < 4; offsetCount++, currAngle += vc_math::pi) {
              if (currSegment.minAngleSlack <= currAngle && currAngle <= currSegment.maxAngleSlack) {
                includeNeighbour = true;
                break;
              }
            }

            // Include neighbouring edgel and update both the angle range and the slack angle range
            if (includeNeighbour) {
              floodfillStack.push_back(cv::Point2i(neighX, neighY));
              currSegment.points.push_back(floodfillStack.back());
              remEdgelImg.at<unsigned char>(neighY, neighX) = 0;

              if (currAngle < currSegment.minAngle) {
                currSegment.minAngle = currAngle;
                slackAngle = 2*angleMarginRad - (currSegment.maxAngle - currSegment.minAngle);
                currSegment.minAngleSlack = currSegment.minAngle - slackAngle;
                currSegment.maxAngleSlack = currSegment.maxAngle + slackAngle;
              } else if (currAngle > currSegment.maxAngle) {
                currSegment.maxAngle = currAngle;
                slackAngle = 2*angleMarginRad - (currSegment.maxAngle - currSegment.minAngle);
                currSegment.minAngleSlack = currSegment.minAngle - slackAngle;
                currSegment.maxAngleSlack = currSegment.maxAngle + slackAngle;
              }
            }
          } // neighX
        } // neighY
      } // depth-first floodfill

      segments.push_back(currSegment);
    } // for each edgel inside the CC
  } // for each connected component

  return segments;
};


/**
 * Computes endpoints (a.x, a.y, b.x, b.y) for segment composed of (unordered) connected
 * edgels, by rotating edgels to align their line fit to a dominant axis (i.e. Y-axis),
 * and identifying minimum and maximum range of their projected values on this
 * axis (i.e. projected y-values).
 */
cv::Vec4i computeSegmentEndpoints(Segment& currSegment,
    double lineFitRadiusAcc = 0.01, double lineFitAngleAcc = 0.01) {
  double lineAngle, sinAngle, cosAngle;
  double minRotValue, maxRotValue, currRotValue;

  cv::fitLine(currSegment.points, currSegment.line, CV_DIST_L2, 0, lineFitRadiusAcc, lineFitAngleAcc);

  lineAngle = std::atan2(currSegment.line[1], currSegment.line[0]);
  sinAngle = std::sin(lineAngle);
  cosAngle = std::cos(lineAngle);

  int minID = 0;
  int maxID = 0;
  int currID = 0;
  for (cv::Point2i& currPt: currSegment.points) {
    currRotValue = currPt.x * cosAngle + currPt.y * sinAngle;
    if (currID == 0) {
      minRotValue = currRotValue;
      maxRotValue = currRotValue;
    } else {
      if (currRotValue < minRotValue) {
        minRotValue = currRotValue;
        minID = currID;
      }
      if (currRotValue > maxRotValue) {
        maxRotValue = currRotValue;
        maxID = currID;
      }
    }
    currID += 1;
  }
  currSegment.endpointA = currSegment.points[minID];
  currSegment.endpointB = currSegment.points[maxID];

  return cv::Vec4i(currSegment.endpointA.x, currSegment.endpointA.y, currSegment.endpointB.x, currSegment.endpointB.y);
};


std::list<cv::Vec4i> detectLineSegments(cv::Mat grayImg,
    int sobelThreshHigh, int sobelThreshLow, int sobelBlurWidth,
    unsigned int ccMinNumEdgels, double angleMarginRad,
    unsigned int segmentMinNumEdgels) {
  // Validate inputs
  assert(grayImg.type() == CV_8UC1);
  assert(angleMarginRad < vc_math::pi);

  // Identify edgels
  cv::Mat edgelImg, dxImg, dyImg;
  blur(grayImg, edgelImg, cv::Size(sobelBlurWidth, sobelBlurWidth));
  Canny(edgelImg, edgelImg, sobelThreshLow, sobelThreshHigh, sobelBlurWidth); // args: in, out, low_thresh, high_thresh, gauss_blur

  // Compute derivative components along x and y axes (needed to compute orientation of edgels)
  cv::Sobel(grayImg, dxImg, CV_16S, 1, 0, sobelBlurWidth, 1, 0, cv::BORDER_REPLICATE);
  cv::Sobel(grayImg, dyImg, CV_16S, 0, 1, sobelBlurWidth, 1, 0, cv::BORDER_REPLICATE);

  // Identify all connected edgel components above minimum count threshold
  std::vector< std::vector<cv::Point2i> > edgelCCs = identifyEdgelCCs(edgelImg);
  std::vector< std::vector<cv::Point2i> >::iterator ccIt = edgelCCs.begin();
  while (ccIt != edgelCCs.end()) {
    if (ccIt->size() < ccMinNumEdgels) {
      ccIt = edgelCCs.erase(ccIt);
    } else {
      ccIt++;
    }
  }

  // Identify connected components within angular range, and above minimum count threshold
  std::list<Segment> segments = groupLinearSegmentsInCC(edgelCCs, edgelImg, dxImg, dyImg, angleMarginRad);
  std::list<Segment>::iterator segIt = segments.begin();
  while (segIt != segments.end()) {
    if (segIt->points.size() < segmentMinNumEdgels) {
      segIt = segments.erase(segIt);
    } else {
      segIt++;
    }
  }

  // Sort edgels in each segment in continguous order, as defined by their closest fitted line
  std::list<cv::Vec4i> segmentEndpoints;
  for (Segment& currSegment: segments) {
    segmentEndpoints.push_back(computeSegmentEndpoints(currSegment));
  }

  /*
  // Display edgels
  cv::Mat overlayImg = edgelImg * 0.5;
  std::cout << "Found " << segments.size() << " segments [from a total of " << edgelCCs.size() << " edgel sets]" << std::endl;
  for (Segment& currSegment: segments) {
    assert(!currSegment.points.empty());
    for (cv::Point2i& currPt: currSegment.points) {
      overlayImg.at<unsigned char>(currPt.y, currPt.x) = 200;
    }
  }
  for (cv::Vec4i& endpts: segmentEndpoints) {
    cv::circle(overlayImg, cv::Point2i(endpts[0], endpts[1]), 2, cv::Scalar(255), 1);
    cv::circle(overlayImg, cv::Point2i(endpts[2], endpts[3]), 2, cv::Scalar(255), 2);
  }
  cv::imshow("segments", overlayImg);
  */

  return segmentEndpoints;
};
