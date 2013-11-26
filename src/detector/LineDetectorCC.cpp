#include "detector/FTag2Detector.hpp"
#include "common/BaseCV.hpp"
#include <cmath>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>


struct Segment {
  // Elements populated by grouplearSegmentsInCC()
  std::vector<cv::Point2i> points; // NOTE: points are NOT assumed to be sorted in any way
  double minAngleSlack, maxAngleSlack, minAngle, maxAngle; // WARNING: may be corrupted by tryMergeColinear()

  // Elements populated by populateSegmentEndpoints()
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
 * Computes endpoints (a.x, a.y, b.x, b.y) for segment composed of (unordered) connected
 * edgels, by rotating edgels to align their line fit to a dominant axis (i.e. Y-axis),
 * and identifying minimum and maximum range of their projected values on this
 * axis (i.e. projected y-values).
 */
void populateSegmentEndpoints(Segment& currSegment,
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
};


bool tryMergeColinear(Segment& a, Segment& b, double endpointDistThresh = 2.0,
    double angleThreshRad = 10.0*vc_math::degree) {
  if (b.points.empty()) return false;
  if ((vc_math::dist(a.endpointA, b.endpointA) > endpointDistThresh) &&
      (vc_math::dist(a.endpointA, b.endpointB) > endpointDistThresh) &&
      (vc_math::dist(a.endpointB, b.endpointA) > endpointDistThresh) &&
      (vc_math::dist(a.endpointA, b.endpointB) > endpointDistThresh)) return false;
  if (vc_math::angularDist(
          std::atan2(a.line[1], a.line[0]),
          std::atan2(b.line[1], b.line[0]), vc_math::pi) > angleThreshRad) return false;

  a.points.reserve(a.points.size() + b.points.size());
  a.points.insert(a.points.end(), b.points.begin(), b.points.end());
  b.points.clear();
  a.minAngle = -1; a.maxAngle = -1; // indicate no longer possible to keep track of angles

  // NOTE: this could MAYBE be more efficiently implemented by finding
  //       adjacent endpoints, which would apply only for non-overlapping
  //       colinear segments. Instead, the following approach would also work
  //       even in cases of semi-overlapping segments.
  populateSegmentEndpoints(a);
  return true;
};


/**
 * Groups edgel elements in each connected component into linear segments, where each
 * grouping has a gradient direction range of at most 2*angleMarginRad.
 *
 * This is implemented using a breadth-first iterative floodfill algorithm, over the given
 * edgel image. Connected edgels are included into an existing segment if their gradient
 * direction is within the current slack angle range. The slack angle range starts at
 * [firstDir - 2*angleMarginRad, firstDir + 2*angleMarginDir] and iteratively reduces
 * to a minimum of [minAngle, minAngle + 2*angleMarginRad] as more edgels are included.
 *
 * NOTE: breadth-first search is preferred over depth-first search because
 *       a depth-first floodfill starting at a corner edgel (where the gradient
 *       direction is between two orthogonal adjascent line segments) may
 *       include only a subset of one of the segments. In contrast, a breadth-
 *       first floodfill starting at a corner edgel will at most remove 1 edgel
 *       from each of the two adjascent segments.
 */
std::list<Segment> groupLinearSegmentsInCC(const std::vector< std::vector<cv::Point2i> > edgelCCs,
    const cv::Mat edgelImg, const cv::Mat dxImg, const cv::Mat dyImg, double angleMarginRad) {
  const int imWidth = edgelImg.cols;
  const int imHeight = edgelImg.rows;
  cv::Mat remEdgelImg = edgelImg.clone();
  std::list<Segment> segments, segmentsCC;
  std::list<cv::Point2i> floodfillQueue;
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
      floodfillQueue.push_back(initPt);
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

      // Apply breadth-first floodfill while reducing [minAngleSlack, maxAngleSlack] range
      while (!floodfillQueue.empty()) {
        currPt = floodfillQueue.front();
        floodfillQueue.pop_front();

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
              floodfillQueue.push_back(cv::Point2i(neighX, neighY));
              currSegment.points.push_back(floodfillQueue.back());
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
      } // breadth-first floodfill

      // Only register line segment if it contains 2 or more edgels
      if (currSegment.points.size() >= 2) {
        populateSegmentEndpoints(currSegment);
        segmentsCC.push_back(currSegment);
      }
    } // for each edgel inside the CC

    // Merge nearby co-linear segments
    if (segmentsCC.size() > 1) {
      std::list<Segment>::iterator segA = segmentsCC.begin(), segB;
      for (; segA != segmentsCC.end(); segA++) {
        if (segA->points.empty()) continue;

        segB = segA; segB++;
        for (; segB != segmentsCC.end(); segB++) {
          if (segB->points.empty()) continue;
          tryMergeColinear(*segA, *segB);
        }
      }

      // Erase merged segments
      segA = segmentsCC.begin();
      while (segA != segmentsCC.end()) {
        if (segA->points.empty()) {
          segA = segmentsCC.erase(segA);
        } else {
          segA++;
        }
      }
    }
    segments.splice(segments.end(), segmentsCC);
  } // for each connected component

  return segments;
};


std::vector<cv::Vec4i> detectLineSegments(cv::Mat grayImg,
    int sobelThreshHigh, int sobelThreshLow, int sobelBlurWidth,
    unsigned int ccMinNumEdgels, double angleMarginRad,
    unsigned int segmentMinNumEdgels) {
  // Validate inputs
  assert(grayImg.type() == CV_8UC1);
  assert(angleMarginRad < vc_math::pi);

  // Identify edgels and compute derivate components along x and y axes (needed to compute orientation of edgels)
  cv::Mat edgelImg, dxImg, dyImg;
  blur(grayImg, edgelImg, cv::Size(sobelBlurWidth, sobelBlurWidth));

  // NOTE: The commented implementation using OpenCV's Canny + Sobel functions
  //       is wasteful since Canny calls Sobel internally. We therefore copied
  //       Canny's source code, and asked it to also return dx and dy.
  //Canny(edgelImg, edgelImg, sobelThreshLow, sobelThreshHigh, sobelBlurWidth); // args: in, out, low_thresh, high_thresh, gauss_blur
  //cv::Sobel(grayImg, dxImg, CV_16S, 1, 0, sobelBlurWidth, 1, 0, cv::BORDER_REPLICATE);
  //cv::Sobel(grayImg, dyImg, CV_16S, 0, 1, sobelBlurWidth, 1, 0, cv::BORDER_REPLICATE);
  OpenCVCanny(edgelImg, edgelImg, sobelThreshLow, sobelThreshHigh, sobelBlurWidth, dxImg, dyImg);
  cv::imshow("edgels", edgelImg); // TODO: 0 remove after debug

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

  std::vector<cv::Vec4i> segmentEndpoints;
  for (Segment& currSegment: segments) {
    segmentEndpoints.push_back(cv::Vec4i(
        currSegment.endpointA.x,
        currSegment.endpointA.y,
        currSegment.endpointB.x,
        currSegment.endpointB.y));
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
  cv::imshow("debug", overlayImg);
  */

  return segmentEndpoints;
};
