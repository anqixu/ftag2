#include "detector/FTag2Detector.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <list>


struct LineSegment {
  double xa, ya, xb, yb;
  unsigned int ha, hb; // hash bucket indices

  double length;
  double orientation;

  std::list<unsigned int> neighIDs;
};


struct HashBucket {
  std::list<unsigned int> segmentIDs; // sorted due to in-order insertion
  int hx, hy; // hashmap row and column indices

  HashBucket() : hx(-1), hy(-1) {};
  HashBucket(int x, int y, int idx) : hx(x), hy(y) { segmentIDs.push_back(idx); };
};


/**
 * Returns 1 if the line segments intersect, 0 if their lines intersect
 * beyond one or both segments, or -1 if they are co-linear.
 * In addition, the intersection point is stored in intPt if available.
 * Furthermore, minimum distance between endpoints and intersecting point
 * for each segment are returned as well.
 */
char getSegmentIntersection(const cv::Vec4i& segA, const cv::Vec4i& segB,
    cv::Point2d& intPt, double* distSegAInt = NULL, double* distSegBInt = NULL,
    double* segAIntRatio = NULL, double* segBIntRatio = NULL,
    double* endptsDist = NULL) {
  // Solution deduced from following equations:
  // - xi = xa1 + ra (xa2 - xa1) = xb1 + rb (xb2 - xb1)
  // - yi = ya1 + ra (ya2 - ya1) = yb1 + rb (yb2 - yb1)
  //
  // For convenience, define dxa = xa2 - xa1, dya = ya2 - ya1, etc.
  double dxa = segA[2] - segA[0];
  double dya = segA[3] - segA[1];
  double dxb = segB[2] - segB[0];
  double dyb = segB[3] - segB[1];
  double det = dya*dxb - dxa*dyb;
  if (fabs(det) <= 10*std::numeric_limits<double>::epsilon()) {
    intPt.x = 0; intPt.y = 0;
    if (distSegAInt != NULL) *distSegAInt = std::numeric_limits<double>::infinity();
    if (distSegBInt != NULL) *distSegBInt = std::numeric_limits<double>::infinity();
    if (segAIntRatio != NULL) *segAIntRatio = std::numeric_limits<double>::infinity();
    if (segBIntRatio != NULL) *segBIntRatio = std::numeric_limits<double>::infinity();
    if (endptsDist != NULL) *endptsDist = std::numeric_limits<double>::infinity();
    return -1;
  }

  double dxba = segB[0] - segA[0];
  double dyba = segB[1] - segA[1];
  double ra = (-dyb*dxba + dxb*dyba) / det;
  double rb = (-dya*dxba + dxa*dyba) / det;
  intPt.x = segA[0] + ra * dxa;
  intPt.y = segA[1] + ra * dya;

  if (distSegAInt != NULL) {
    double raDist = (ra <= 0.5) ? ra : 1.0 - ra;
    *distSegAInt = std::fabs(raDist) * std::sqrt(dxa*dxa + dya*dya);
  }
  if (distSegBInt != NULL) {
    double rbDist = (rb <= 0.5) ? rb : 1.0 - rb;
    *distSegBInt = std::fabs(rbDist) * std::sqrt(dxb*dxb + dyb*dyb);
  }
  if (segAIntRatio != NULL) {
    *segAIntRatio = ra;
  }
  if (segBIntRatio != NULL) {
    *segBIntRatio = rb;
  }
  if (endptsDist != NULL) {
    if (ra < 0.5) {
      if (rb < 0.5) {
        *endptsDist = vc_math::dist(segA[0], segA[1], segB[0], segB[1]);
      } else {
        *endptsDist = vc_math::dist(segA[0], segA[1], segB[2], segB[3]);
      }
    } else {
      if (rb < 0.5) {
        *endptsDist = vc_math::dist(segA[2], segA[3], segB[0], segB[1]);
      } else {
        *endptsDist = vc_math::dist(segA[2], segA[3], segB[2], segB[3]);
      }
    }
  }

  if (ra >= 0 && ra <= 1 && rb >= 0 && rb <= 1) {
    return 1;
  }
  return 0;
};


char getSegmentIntersection(const LineSegment& segA, const LineSegment& segB,
    cv::Point2d& intPt, double* distSegAInt = NULL, double* distSegBInt = NULL,
    double* segAIntRatio = NULL, double* segBIntRatio = NULL,
    double* endptsDist = NULL) {
  // Solution deduced from following equations:
  // - xi = xa1 + ra (xa2 - xa1) = xb1 + rb (xb2 - xb1)
  // - yi = ya1 + ra (ya2 - ya1) = yb1 + rb (yb2 - yb1)
  //
  // For convenience, define dxa = xa2 - xa1, dya = ya2 - ya1, etc.
  double dxa = segA.xb - segA.xa;
  double dya = segA.yb - segA.ya;
  double dxb = segB.xb - segB.xa;
  double dyb = segB.yb - segB.ya;
  double det = dya*dxb - dxa*dyb;
  if (fabs(det) <= 10*std::numeric_limits<double>::epsilon()) {
    intPt.x = 0; intPt.y = 0;
    if (distSegAInt != NULL) *distSegAInt = std::numeric_limits<double>::infinity();
    if (distSegBInt != NULL) *distSegBInt = std::numeric_limits<double>::infinity();
    if (segAIntRatio != NULL) *segAIntRatio = std::numeric_limits<double>::infinity();
    if (segBIntRatio != NULL) *segBIntRatio = std::numeric_limits<double>::infinity();
    if (endptsDist != NULL) *endptsDist = std::numeric_limits<double>::infinity();
    return -1;
  }

  double dxba = segB.xa - segA.xa;
  double dyba = segB.ya - segA.ya;
  double ra = (-dyb*dxba + dxb*dyba) / det;
  double rb = (-dya*dxba + dxa*dyba) / det;
  intPt.x = segA.xa + ra * dxa;
  intPt.y = segA.ya + ra * dya;

  if (distSegAInt != NULL) {
    double raDist = (ra <= 0.5) ? ra : 1.0 - ra;
    *distSegAInt = std::fabs(raDist) * segA.length;
  }
  if (distSegBInt != NULL) {
    double rbDist = (rb <= 0.5) ? rb : 1.0 - rb;
    *distSegBInt = std::fabs(rbDist) * segB.length;
  }
  if (segAIntRatio != NULL) {
    *segAIntRatio = ra;
  }
  if (segBIntRatio != NULL) {
    *segBIntRatio = rb;
  }
  if (endptsDist != NULL) {
    if (ra < 0.5) {
      if (rb < 0.5) {
        *endptsDist = vc_math::dist(segA.xa, segA.ya, segB.xa, segB.ya);
      } else {
        *endptsDist = vc_math::dist(segA.xa, segA.ya, segB.xb, segB.yb);
      }
    } else {
      if (rb < 0.5) {
        *endptsDist = vc_math::dist(segA.xb, segA.yb, segB.xa, segB.ya);
      } else {
        *endptsDist = vc_math::dist(segA.xb, segA.yb, segB.xb, segB.yb);
      }
    }
  }

  if (ra >= 0 && ra <= 1 && rb >= 0 && rb <= 1) {
    return 1;
  }
  return 0;
};


bool isClockwiseOrder(const cv::Vec4i& segA, const cv::Vec4i& segB, const cv::Point2d& intPt) {
  cv::Point2d endA1(segA[0], segA[1]);
  cv::Point2d endA2(segA[2], segA[3]);
  cv::Point2d endB1(segB[0], segB[1]);
  cv::Point2d endB2(segB[2], segB[3]);
  cv::Point2d vecA, vecB;
  double distA1 = vc_math::distSqrd(intPt, endA1);
  double distA2 = vc_math::distSqrd(intPt, endA2);
  double distB1 = vc_math::distSqrd(intPt, endB1);
  double distB2 = vc_math::distSqrd(intPt, endB2);

  if (distA1 >= distA2) {
    if (distB1 >= distB2) {
      vecA = endA1 - intPt;
      vecB = endB1 - intPt;
    } else { // distB2 > distB1
      vecA = endA1 - intPt;
      vecB = endB2 - intPt;
    }
  } else { // distA2 > distA1
    if (distB1 >= distB2) {
      vecA = endA2 - intPt;
      vecB = endB1 - intPt;
    } else { // distB2 > distB1
      vecA = endA2 - intPt;
      vecB = endB2 - intPt;
    }
  }

  return ((vecA.y*vecB.x - vecA.x*vecB.y) >= 0);
};


// Scan for 4-connected cycles
void completeQuadDFT(const std::vector< std::list<unsigned int> >& adjList,
    std::vector<cv::Vec4i>& quads,
    cv::Vec4i& currQuad,
    unsigned int startIdx, unsigned int currIdx, int depth) {
  if (depth > 3) return;

  currQuad[depth] = currIdx;

  if (depth == 3) {
    for (unsigned int adjIdx: adjList[currIdx]) {
      if (adjIdx == startIdx) {
        quads.push_back(vc_math::minCyclicOrder(currQuad));
        return;
      }
    }
  } else {
    for (unsigned int adjIdx: adjList[currIdx]) {
      if (adjIdx == startIdx) continue; // found a cycle shorter than a quad
      completeQuadDFT(adjList, quads, currQuad, startIdx, adjIdx, depth + 1);
    }
  }
};


struct PartialQuadDFTParams {
  const std::vector< std::list<unsigned int> >& redAdjList;
  const std::vector<cv::Vec4i>& segments;
  const std::vector<double>& segmentLengths;
  const std::vector<int>& toOrigSegIDs;
  double fourConnMaxEndptDistRatio;
  double fiveConnMaxGapDistRatio;
  double fiveConnMaxAlignAngle; // in radians
  PartialQuadDFTParams(
      const std::vector< std::list<unsigned int> >& _redAdjList,
      const std::vector<cv::Vec4i>& _segments,
      const std::vector<double>& _segmentLengths,
      const std::vector<int>& _toOrigSegIDs,
      double _fourConnMaxEndptDistRatio,
      double _fiveConnMaxGapDistRatio,
      double _fiveConnMaxAlignAngle) :
        redAdjList(_redAdjList),
        segments(_segments),
        segmentLengths(_segmentLengths),
        toOrigSegIDs(_toOrigSegIDs),
        fourConnMaxEndptDistRatio(_fourConnMaxEndptDistRatio),
        fiveConnMaxGapDistRatio(_fiveConnMaxGapDistRatio),
        fiveConnMaxAlignAngle(_fiveConnMaxAlignAngle) {};
};


// Scan for 4-connected segments (either cycles, or where 1 quad corner is
// gapped), and potential 5-connected segments (quads where middle of 1 edge is
// gapped, to be later verified by isFiveConnQuad())
void partialQuadDFT(const PartialQuadDFTParams& data,
    std::vector<cv::Vec4i>& quads,
    std::vector< std::array<int, 5> >& fiveConnSegments,
    std::array<int, 5>& currQuadIDs,
    unsigned int currIdx, int depth) {
  if (depth > 3) return;

  currQuadIDs[depth] = currIdx;

  if (depth == 3) {
    cv::Point2d intPt;
    double distSegAInt, distSegBInt;
    int i = data.toOrigSegIDs[currQuadIDs[0]];
    int j = data.toOrigSegIDs[currIdx];

    // Check for 4-connected partial quad
    char intersect = getSegmentIntersection(data.segments[i], data.segments[j],
        intPt, &distSegAInt, &distSegBInt);
    if (intersect == 0 &&
        (distSegAInt / (distSegAInt + data.segmentLengths[i]) <= data.fourConnMaxEndptDistRatio) &&
        (distSegBInt / (distSegBInt + data.segmentLengths[j]) <= data.fourConnMaxEndptDistRatio)) {
      intersect = 1;
    }
    if (intersect > 0) {
      quads.push_back(vc_math::minCyclicOrder(cv::Vec4i(currQuadIDs[0], currQuadIDs[1], currQuadIDs[2], currQuadIDs[3])));
      return;
    }

    // If not a partial 4-conn quad, then store 5-conn quads, to be bridged later
    for (unsigned int adjIdx: data.redAdjList[currIdx]) {
      // Check for cyclic quads
      // NOTE: triggered only if fourConnMaxEndptDistRatio is lower than quadMaxEndptDistRatio
      if ((int) adjIdx == currQuadIDs[0]) {
        quads.push_back(vc_math::minCyclicOrder(cv::Vec4i(currQuadIDs[0], currQuadIDs[1], currQuadIDs[2], currQuadIDs[3])));
        continue;
      } else {
        currQuadIDs[4] = adjIdx;
        fiveConnSegments.push_back(currQuadIDs);
      }
    }
  } else {
    bool foundCycle;
    for (unsigned int adjIdx: data.redAdjList[currIdx]) {
      // Check for less-than-4 cycles
      foundCycle = false;
      for (int prevDepth = 0; prevDepth < depth - 1; prevDepth++) {
        if ((int) adjIdx == currQuadIDs[prevDepth]) {
          foundCycle = true;
          continue;
        }
      }
      if (foundCycle) continue;

      // Recurse
      partialQuadDFT(data, quads, fiveConnSegments, currQuadIDs, adjIdx, depth + 1);
    }
  }
};


// Verify if 5-connected segment corresponds to a quad
bool isFiveConnQuad(const PartialQuadDFTParams& data,
    const std::array<int, 5>& fiveConnSegIDs,
    Quad& resultBuffer, double minQuadWidth) {
  int firstIdx = data.toOrigSegIDs[fiveConnSegIDs[0]];
  int fifthIdx = data.toOrigSegIDs[fiveConnSegIDs[4]];
  const cv::Vec4i& firstSegment = data.segments[firstIdx];
  const cv::Vec4i& fifthSegment = data.segments[fifthIdx];

  // Find intersection points between 1st/2nd segments, and 4th/5th segments
  cv::Point2d intPtFirstSecond, intPtFourthFifth;
  int secondIdx = data.toOrigSegIDs[fiveConnSegIDs[1]];
  int fourthIdx = data.toOrigSegIDs[fiveConnSegIDs[3]];
  const cv::Vec4i& secondSegment = data.segments[secondIdx];
  const cv::Vec4i& fourthSegment = data.segments[fourthIdx];
  getSegmentIntersection(firstSegment, secondSegment, intPtFirstSecond);
  getSegmentIntersection(fourthSegment, fifthSegment, intPtFourthFifth);
  double superSegmentDist = vc_math::dist(intPtFirstSecond, intPtFourthFifth);
  if (superSegmentDist < minQuadWidth) { return false; }

  // Ensure that first and last segments each align sufficiently to the super segment
  double superSegmentOrientation = vc_math::orientation(intPtFirstSecond, intPtFourthFifth);
  double firstSegmentAlignAngle = vc_math::angularDist(vc_math::orientation(firstSegment),
      superSegmentOrientation, vc_math::pi);
  double fifthSegmentAlignAngle = vc_math::angularDist(vc_math::orientation(fifthSegment),
      superSegmentOrientation, vc_math::pi);
  if (firstSegmentAlignAngle > data.fiveConnMaxAlignAngle &&
      firstSegmentAlignAngle < vc_math::pi - data.fiveConnMaxAlignAngle) { return false; }
  if (fifthSegmentAlignAngle > data.fiveConnMaxAlignAngle &&
      fifthSegmentAlignAngle < vc_math::pi - data.fiveConnMaxAlignAngle) { return false; }

  // Start with a "0 to 1" range corresponding to the super-segment bridging
  // between the aforementioned 2 intersection points, and reduce this range
  // based on overlaps from the projected 1st segment and projected 5th segment
  //
  // norm_proj(Pt onto intA->intB) = (Pt-intA) (dot) (intB-intA) / norm(intB-intA)^2
  //
  // Also note that we only care about the range "right" of the projected first
  // segment and "left" of the projected fifth segment, on this super-segment
  double superSegmentDistSqrd = superSegmentDist*superSegmentDist;
  double intBAx = intPtFourthFifth.x - intPtFirstSecond.x;
  double intBAy = intPtFourthFifth.y - intPtFirstSecond.y;
  double projFirstLeft = ((firstSegment[0] - intPtFirstSecond.x)*intBAx +
      (firstSegment[1] - intPtFirstSecond.y)*intBAy)/superSegmentDistSqrd;
  double projFirstRight = ((firstSegment[2] - intPtFirstSecond.x)*intBAx +
      (firstSegment[3] - intPtFirstSecond.y)*intBAy)/superSegmentDistSqrd;
  double projFifthLeft = ((fifthSegment[0] - intPtFirstSecond.x)*intBAx +
      (fifthSegment[1] - intPtFirstSecond.y)*intBAy)/superSegmentDistSqrd;
  double projFifthRight = ((fifthSegment[2] - intPtFirstSecond.x)*intBAx +
      (fifthSegment[3] - intPtFirstSecond.y)*intBAy)/superSegmentDistSqrd;
  if (projFirstLeft > projFirstRight) { std::swap(projFirstLeft, projFirstRight); }
  if (projFifthLeft > projFifthRight) { std::swap(projFifthLeft, projFifthRight); }
  if (projFifthLeft > 1) projFifthLeft = 1;
  if (projFirstRight < 0) projFirstRight = 0;
  double normGapDist = projFifthLeft - projFirstRight;
  if (normGapDist > data.fiveConnMaxGapDistRatio) { return false; }

  // If the 1st/5th segments are sufficiently aligned, and if their gap is
  // sufficiently short, then construct quad
  int thirdIdx = data.toOrigSegIDs[fiveConnSegIDs[2]];
  const cv::Vec4i& thirdSegment = data.segments[thirdIdx];
  cv::Point2d corner;
  resultBuffer.corners[0].x = intPtFirstSecond.x; resultBuffer.corners[0].y = intPtFirstSecond.y;
  getSegmentIntersection(secondSegment, thirdSegment, corner);
  resultBuffer.corners[1].x = corner.x; resultBuffer.corners[1].y = corner.y;
  getSegmentIntersection(thirdSegment, fourthSegment, corner);
  resultBuffer.corners[2].x = corner.x; resultBuffer.corners[2].y = corner.y;
  resultBuffer.corners[3].x = intPtFourthFifth.x; resultBuffer.corners[3].y = intPtFourthFifth.y;
  resultBuffer.updateArea();
  return (resultBuffer.area > 0);
};


std::list<Quad> scanQuadsExhaustive(const std::vector<cv::Vec4i>& segments,
    double intSegMinAngle, double maxTIntDistRatio, double maxEndptDistRatio,
    double maxCornerGapEndptDistRatio,
    double maxEdgeGapDistRatio, double maxEdgeGapAlignAngle,
    double minQuadWidth) {
  std::list<Quad> quads;

  // Compute lengths and orientations of each segment
  std::vector<double> segmentLengths;
  std::vector<double> segmentOrientations;
  for (const cv::Vec4i& seg: segments) {
    segmentLengths.push_back(vc_math::dist(seg));
    segmentOrientations.push_back(vc_math::orientation(seg));
  }

  // Identify connected segments
  std::vector< std::list<unsigned int> > adjList(segments.size());
  std::vector<bool> incomingAdj(segments.size(), false); // whether segment has incoming adjacent segment(s)
  unsigned int i, j;
  cv::Point2d intPt;
  char intersect;
  double distSegAInt, distSegBInt, segAIntRatio, segBIntRatio;
  for (i = 0; i < segments.size(); i++) {
    for (j = i+1; j < segments.size(); j++) {
      // Do not connect nearby segments with sharp (or near-180') angles in between them
      double intAngle = vc_math::angularDist(segmentOrientations[i],
          segmentOrientations[j], vc_math::pi);
      if (intAngle < intSegMinAngle || intAngle > vc_math::pi - intSegMinAngle) { continue; }

      intersect = getSegmentIntersection(segments[i], segments[j], intPt,
          &distSegAInt, &distSegBInt, &segAIntRatio, &segBIntRatio);

      // Do not connect T-shaped segments
      if ((segAIntRatio >= maxTIntDistRatio && segAIntRatio < 1.0 - maxTIntDistRatio) ||
          (segBIntRatio >= maxTIntDistRatio && segBIntRatio < 1.0 - maxTIntDistRatio)) {
        continue;
      }

      // Connect segments whose endpoints are nearby, and also whose
      // intersecting point is also near each of the segments' endpoints
      // (specifically where the triangle between the 2 endpoints and the
      // intersecting point is at most as sharp as a triangle with sides
      // endptThresh, 2*endptThresh, 2*endptThresh)
      if (intersect == 0 &&
          (distSegAInt / (distSegAInt + segmentLengths[i]) <= maxEndptDistRatio) &&
          (distSegBInt / (distSegBInt + segmentLengths[j]) <= maxEndptDistRatio)) {
        intersect = 1;
      }

      // Determine adjacency order between the two segments
      if (intersect > 0) {
        if (isClockwiseOrder(segments[i], segments[j], intPt)) {
          adjList[i].push_back(j);
          incomingAdj[j] = true;
        } else {
          adjList[j].push_back(i);
          incomingAdj[i] = true;
        }
      }
    }
  }

  // Keep only intersecting edgels and create reduced adjacency matrix + list
  std::vector<int> toIntSegIDs(segments.size(), -1);
  std::vector<int> toOrigSegIDs;
  for (i = 0; i < segments.size(); i++) {
    if (adjList[i].size() > 0 || incomingAdj[i]) {
      j = toOrigSegIDs.size();
      toIntSegIDs[i] = j;
      toOrigSegIDs.push_back(i);
    }
  }
  std::vector< std::list<unsigned int> > redAdjList(toOrigSegIDs.size());
  for (j = 0; j < toOrigSegIDs.size(); j++) {
    std::list<unsigned int>& currAdj = redAdjList[j];
    i = toOrigSegIDs[j];
    for (unsigned int neighI: adjList[i]) {
      unsigned int neighJ = toIntSegIDs[neighI];
      currAdj.push_back(neighJ);
    }
  }

  // Traverse through adjascency list and search for complete (cyclic) and
  // partial (single-corner-obstructed and single-edge-obstructed) quads
  PartialQuadDFTParams partialQuadData(redAdjList, segments, segmentLengths,
      toOrigSegIDs, maxCornerGapEndptDistRatio, maxEdgeGapDistRatio,
      maxEdgeGapAlignAngle);
  std::vector<cv::Vec4i> segQuads;
  std::vector< std::array<int, 5> > segFiveConns;
  std::array<int, 5> currQuadIDs;
  for (unsigned int currCandIdx = 0; currCandIdx < toOrigSegIDs.size(); currCandIdx++) {
    partialQuadDFT(partialQuadData, segQuads, segFiveConns, currQuadIDs, currCandIdx, 0);
  }
  vc_math::unique(segQuads);
  vc_math::unique(segFiveConns);

  // Compute corners of 4-connected quads
  cv::Point2d corner;
  Quad quad;
  for (cv::Vec4i& segQuad: segQuads) {
    const cv::Vec4i& segA = segments[toOrigSegIDs[segQuad[0]]];
    const cv::Vec4i& segB = segments[toOrigSegIDs[segQuad[1]]];
    const cv::Vec4i& segC = segments[toOrigSegIDs[segQuad[2]]];
    const cv::Vec4i& segD = segments[toOrigSegIDs[segQuad[3]]];
    getSegmentIntersection(segA, segB, corner);
    quad.corners[0].x = corner.x; quad.corners[0].y = corner.y;
    getSegmentIntersection(segB, segC, corner);
    quad.corners[1].x = corner.x; quad.corners[1].y = corner.y;
    getSegmentIntersection(segC, segD, corner);
    quad.corners[2].x = corner.x; quad.corners[2].y = corner.y;
    getSegmentIntersection(segD, segA, corner);
    quad.corners[3].x = corner.x; quad.corners[3].y = corner.y;
    quad.updateArea();
    if (quad.area >= minQuadWidth*minQuadWidth && quad.checkMinWidth(minQuadWidth)) { // checking min width to prevent triangle+edge 4-conn segments from being accepted as quads
      quads.push_back(quad);
    }
  }

  // Construct single-edge-obstructed quads from 5-connected segments
  for (const std::array<int, 5>& segFiveConn: segFiveConns) {
    if (isFiveConnQuad(partialQuadData, segFiveConn, quad, minQuadWidth)) {
      if (quad.area >= minQuadWidth*minQuadWidth && quad.checkMinWidth(minQuadWidth)) {
        quads.push_back(quad);
      }
    }
  }

  return quads;
};


std::list<Quad> scanQuadsSpatialHash(const std::vector<cv::Vec4i>& segments,
    unsigned int imWidth, unsigned int imHeight,
    double intSegMinAngle, unsigned int hashMapWidth,
    double maxTIntDistRatio, double maxEndptDistRatio,
    double maxCornerGapEndptDistRatio,
    double maxEdgeGapDistRatio, double maxEdgeGapAlignAngle,
    double minQuadWidth) {
  std::list<Quad> quads;

  // Initialize augmented data structure for each segment
  std::vector<LineSegment> segmentStructs(segments.size());
  std::vector<double> segmentLengths;
  std::vector<LineSegment>::iterator segmentIt = segmentStructs.begin();
  for (const cv::Vec4i& seg: segments) {
    segmentIt->xa = seg[0];
    segmentIt->ya = seg[1];
    segmentIt->xb = seg[2];
    segmentIt->yb = seg[3];
    segmentIt->length = vc_math::dist(seg);
    segmentIt->orientation = vc_math::orientation(seg);
    segmentLengths.push_back(segmentIt->length);
    segmentIt++;
  }

  // Build spatial hashmap, to identify segment endpoints near each other
  std::vector<HashBucket> buckets; buckets.push_back(HashBucket()); // First element is invalid by design
  cv::Mat hashmap = cv::Mat::zeros(
      std::ceil(double(imHeight)/hashMapWidth),
      std::ceil(double(imWidth)/hashMapWidth),
      CV_32SC1); // Stores 0-indices to hash bucket entries
  int hx, hy; // Scratch vars
  int newBucketIdx = 1;
  unsigned int segmentID = 0;
  for (LineSegment& currSegment: segmentStructs) {
    // Process endpoint A
    hx = std::floor(double(currSegment.xa)/hashMapWidth);
    hy = std::floor(double(currSegment.ya)/hashMapWidth);
    int& bucketIdxA = hashmap.at<int>(hy, hx);
    if (bucketIdxA <= 0) { // Insert into new bucket
      buckets.push_back(HashBucket(hx, hy, segmentID));
      bucketIdxA = newBucketIdx;
      currSegment.ha = bucketIdxA;
      newBucketIdx += 1;
    } else { // Insert into existing bucket
      HashBucket& currBucket = buckets[bucketIdxA];
      currSegment.ha = bucketIdxA;
      currBucket.segmentIDs.push_back(segmentID);
    }

    // Process endpoint B
    hx = std::floor(double(currSegment.xb)/hashMapWidth);
    hy = std::floor(double(currSegment.yb)/hashMapWidth);
    int& bucketIdxB = hashmap.at<int>(hy, hx);
    if (bucketIdxB <= 0) { // Insert into new bucket
      buckets.push_back(HashBucket(hx, hy, segmentID));
      bucketIdxB = newBucketIdx;
      currSegment.hb = bucketIdxB;
      newBucketIdx += 1;
    } else { // Insert into existing bucket
      HashBucket& currBucket = buckets[bucketIdxB];
      currSegment.hb = bucketIdxB;
      if (currSegment.ha != currSegment.hb) { // Prevent registering segment at both endpoints in same bucket
        currBucket.segmentIDs.push_back(segmentID);
      }
    }

    segmentID += 1;
  }

  // Identify potential neighbours to each segment
  segmentID = 0;
  for (LineSegment& currSegment: segmentStructs) {
    // List potential neighbours of endpoint A
    hx = buckets[currSegment.ha].hx;
    hy = buckets[currSegment.ha].hy;
    for (int nhy = std::max(hy - 1, 0); nhy < std::min(hy + 2, hashmap.rows); nhy += 1) {
      for (int nhx = std::max(hx - 1, 0); nhx < std::min(hx + 2, hashmap.cols); nhx += 1) {
        unsigned int bucketIdx = hashmap.at<unsigned int>(nhy, nhx);
        if (bucketIdx <= 0) continue; // No neighbours in current hashmap cell
        for (unsigned int neighSegmentID: buckets[bucketIdx].segmentIDs) {
          if (neighSegmentID <= segmentID) continue; // Skip ID of current segment or previously checked neighbour
          currSegment.neighIDs.push_back(neighSegmentID);
        }
      }
    }

    // List potential neighbours of endpoint B
    if (currSegment.ha != currSegment.hb) {
      hx = buckets[currSegment.hb].hx;
      hy = buckets[currSegment.hb].hy;
      for (int nhy = std::max(hy - 1, 0); nhy < std::min(hy + 2, hashmap.rows); nhy += 1) {
        for (int nhx = std::max(hx - 1, 0); nhx < std::min(hx + 2, hashmap.cols); nhx += 1) {
          unsigned int bucketIdx = hashmap.at<unsigned int>(nhy, nhx);
          if (bucketIdx <= 0) continue; // No neighbours in current hashmap cell
          for (unsigned int neighSegmentID: buckets[bucketIdx].segmentIDs) {
            if (neighSegmentID <= segmentID) continue; // Skip ID of current segment or previously checked neighbour
            currSegment.neighIDs.push_back(neighSegmentID);
          }
        }
      }
    }

    // Maintain a sorted set of neighbours
    if (currSegment.neighIDs.empty()) continue;
    currSegment.neighIDs.sort();
    currSegment.neighIDs.erase(std::unique(currSegment.neighIDs.begin(),
      currSegment.neighIDs.end()), currSegment.neighIDs.end());

    segmentID += 1;
  }

  // Build adjacency list by checking nearby segments to each segment's endpoints
  std::vector< std::list<unsigned int> > adjList(segments.size());
  std::vector<bool> incomingAdj(segments.size(), false); // whether segment has incoming adjacent segment(s)
  segmentID = 0;
  int intersect;
  cv::Point2d intPt;
  double distSegAInt, distSegBInt, segAIntRatio, segBIntRatio, endptsDist;
  for (LineSegment& currSegment: segmentStructs) {
    for (unsigned int neighSegmentID: currSegment.neighIDs) {
      // Do not connect nearby segments with sharp (or near-180') angles in between them
      double intAngle = vc_math::angularDist(currSegment.orientation,
        segmentStructs[neighSegmentID].orientation, vc_math::pi);
      if (intAngle < intSegMinAngle || intAngle > vc_math::pi - intSegMinAngle) { continue; }

      intersect = getSegmentIntersection(currSegment, segmentStructs[neighSegmentID],
        intPt, &distSegAInt, &distSegBInt, &segAIntRatio, &segBIntRatio, &endptsDist);

      // Do not connect T-shaped segments
      if ((segAIntRatio >= maxTIntDistRatio && segAIntRatio < 1.0 - maxTIntDistRatio) ||
          (segBIntRatio >= maxTIntDistRatio && segBIntRatio < 1.0 - maxTIntDistRatio)) {
        continue;
      }

      // Connect segments whose endpoints are nearby, and also whose
      // intersecting point is also near each of the segments' endpoints
      // (specifically where the triangle between the 2 endpoints and the
      // intersecting point is at most as sharp as a triangle with sides
      // endptThresh, 2*endptThresh, 2*endptThresh)
      if (intersect == 0 &&
          (distSegAInt / (distSegAInt + currSegment.length) <= maxEndptDistRatio) &&
          (distSegBInt / (distSegBInt + segmentStructs[neighSegmentID].length) <= maxEndptDistRatio)) {
        intersect = 1;
      }

      // Determine adjacency order between the two segments
      if (intersect > 0) {
        if (isClockwiseOrder(segments[segmentID], segments[neighSegmentID], intPt)) {
          adjList[segmentID].push_back(neighSegmentID);
          incomingAdj[neighSegmentID] = true;
        } else {
          adjList[neighSegmentID].push_back(segmentID);
          incomingAdj[segmentID] = true;
        }
      }
    }

    segmentID += 1;
  }

  // Keep only intersecting edgels and create reduced adjacency matrix + list
  std::vector<int> toIntSegIDs(segments.size(), -1);
  std::vector<int> toOrigSegIDs;
  unsigned int i, j;
  for (i = 0; i < segments.size(); i++) {
    if (adjList[i].size() > 0 || incomingAdj[i]) {
      j = toOrigSegIDs.size();
      toIntSegIDs[i] = j;
      toOrigSegIDs.push_back(i);
    }
  }
  std::vector< std::list<unsigned int> > redAdjList(toOrigSegIDs.size());
  for (j = 0; j < toOrigSegIDs.size(); j++) {
    std::list<unsigned int>& currAdj = redAdjList[j];
    i = toOrigSegIDs[j];
    for (unsigned int neighI: adjList[i]) {
      unsigned int neighJ = toIntSegIDs[neighI];
      currAdj.push_back(neighJ);
    }
  }

  // Traverse through adjascency list and search for complete (cyclic) and
  // partial (single-corner-obstructed and single-edge-obstructed) quads
  PartialQuadDFTParams partialQuadData(redAdjList, segments, segmentLengths,
      toOrigSegIDs, maxCornerGapEndptDistRatio, maxEdgeGapDistRatio,
      maxEdgeGapAlignAngle);
  std::vector<cv::Vec4i> segQuads;
  std::vector< std::array<int, 5> > segFiveConns;
  std::array<int, 5> currQuadIDs;
  for (unsigned int currCandIdx = 0; currCandIdx < toOrigSegIDs.size(); currCandIdx++) {
    partialQuadDFT(partialQuadData, segQuads, segFiveConns, currQuadIDs, currCandIdx, 0);
  }
  vc_math::unique(segQuads);
  vc_math::unique(segFiveConns);

  // Compute corners of 4-connected quads
  cv::Point2d corner;
  Quad quad;
  for (cv::Vec4i& segQuad: segQuads) {
    const cv::Vec4i& segA = segments[toOrigSegIDs[segQuad[0]]];
    const cv::Vec4i& segB = segments[toOrigSegIDs[segQuad[1]]];
    const cv::Vec4i& segC = segments[toOrigSegIDs[segQuad[2]]];
    const cv::Vec4i& segD = segments[toOrigSegIDs[segQuad[3]]];
    getSegmentIntersection(segA, segB, corner);
    quad.corners[0].x = corner.x; quad.corners[0].y = corner.y;
    getSegmentIntersection(segB, segC, corner);
    quad.corners[1].x = corner.x; quad.corners[1].y = corner.y;
    getSegmentIntersection(segC, segD, corner);
    quad.corners[2].x = corner.x; quad.corners[2].y = corner.y;
    getSegmentIntersection(segD, segA, corner);
    quad.corners[3].x = corner.x; quad.corners[3].y = corner.y;
    quad.updateArea();
    if (quad.area >= minQuadWidth*minQuadWidth && quad.checkMinWidth(minQuadWidth)) { // checking min width to prevent triangle+edge 4-conn segments from being accepted as quads
      quads.push_back(quad);
    }
  }

  // Construct single-edge-obstructed quads from 5-connected segments
  for (const std::array<int, 5>& segFiveConn: segFiveConns) {
    if (isFiveConnQuad(partialQuadData, segFiveConn, quad, minQuadWidth)) {
      if (quad.area >= minQuadWidth*minQuadWidth && quad.checkMinWidth(minQuadWidth)) {
        quads.push_back(quad);
      }
    }
  }

  return quads;
};


std::list<Quad> detectQuadsViaContour(cv::Mat grayImg,
    unsigned int adaptiveThreshBlockSize, double adaptiveThreshMeanWeight,
    unsigned int quadMinWidth, unsigned int quadMinPerimeter,
    double approxPolyEpsSizeRatio) {
  // Validate input arguments
  if (adaptiveThreshBlockSize < 3) { adaptiveThreshBlockSize = 3; }
  else if (adaptiveThreshBlockSize % 2 == 0) { adaptiveThreshBlockSize += 1; }

  // Threshold image
  cv::Mat threshImg;
  cv::adaptiveThreshold(grayImg, threshImg, 255,
    cv::ADAPTIVE_THRESH_MEAN_C,
    cv::THRESH_BINARY_INV,
    adaptiveThreshBlockSize, adaptiveThreshMeanWeight);

  // Scan for rectangular contours in thresholded image
  unsigned int quadMaxPerimeter = std::max(grayImg.cols, grayImg.rows) * 4; // used as a rough sanity check
  double quadMinWidthSqrd = quadMinWidth*quadMinWidth;
  std::list<Quad> quads;
  std::vector< std::vector<cv::Point> > contours;
  std::vector<cv::Vec4i> contourHierarchy; // Throw-away var; unpopulated due to CV_RETR_LIST arg
  std::vector<cv::Point> currApproxPoly;
  Quad currQuad;
  cv::Point _v1, _v2;
  cv::findContours(threshImg, contours, contourHierarchy,
      CV_RETR_LIST, CV_CHAIN_APPROX_NONE); // NOTE: this fn changes threshImg!
  for (std::vector<cv::Point>& currContour: contours) {
    // Reject small or large contours
    if (currContour.size() < quadMinPerimeter ||
        currContour.size() > quadMaxPerimeter) continue;

    // Approximate to a polygon
    cv::approxPolyDP(currContour, currApproxPoly,
        approxPolyEpsSizeRatio * currContour.size(),
        true); // closed = true

    // Reject non-quadrilaterals
    if (currApproxPoly.size() != 4) continue;

    // Reject non-convex polygons
    if (!cv::isContourConvex(currApproxPoly)) continue;

    // Reject quads that have small widths
    bool lessThanMinWidth = false;
    for (int i = 0; i < 4; i++) {
      if (i == 3) {
        lessThanMinWidth = (vc_math::distSqrd(currApproxPoly[i], currApproxPoly[0]) < quadMinWidthSqrd);
      } else {
        lessThanMinWidth = (vc_math::distSqrd(currApproxPoly[i], currApproxPoly[i+1]) < quadMinWidthSqrd);
      }
      if (lessThanMinWidth) break;
    }
    if (lessThanMinWidth) continue;

    // Store as quad struct (where corners are stored in clockwise order)
    _v1 = currApproxPoly[1] - currApproxPoly[0];
    _v2 = currApproxPoly[2] - currApproxPoly[1];
    if (_v1.x*_v2.y - _v1.y*_v2.x < 0) { // Currently in counter-clockwise order; store in clockwise order
      for (int i = 0; i < 4; i++) {
        currQuad.corners[i].x = currApproxPoly[3-i].x;
        currQuad.corners[i].y = currApproxPoly[3-i].y;
      }
    } else { // Already in clockwise order
      for (int i = 0; i < 4; i++) {
        currQuad.corners[i].x = currApproxPoly[i].x;
        currQuad.corners[i].y = currApproxPoly[i].y;
      }
    }
    currQuad.updateArea();
    quads.push_back(currQuad);
  }

  return quads;
};


cv::Mat extractQuadImg(const cv::Mat img, const Quad& quad, unsigned int minWidth, bool oversample, bool grayscale) {
  int shortestEdgeWidth = floor(
      std::min(std::min(vc_math::dist(quad.corners[0], quad.corners[1]),
                        vc_math::dist(quad.corners[1], quad.corners[2])),
               std::min(vc_math::dist(quad.corners[2], quad.corners[3]),
                        vc_math::dist(quad.corners[3], quad.corners[0]))));

  if (shortestEdgeWidth < int(minWidth)) return cv::Mat();

  // Assumed corners stored in counter-clockwise order in image space
  // (where +x: right, +y: bottom), starting with top-right corner
  cv::Mat quadImg;
  std::vector<cv::Point2f> rectifiedCorners;
  if (oversample) {
    shortestEdgeWidth += 2;
    rectifiedCorners.push_back(cv::Point2f(shortestEdgeWidth-1, 1));
    rectifiedCorners.push_back(cv::Point2f(shortestEdgeWidth-1, shortestEdgeWidth-1));
    rectifiedCorners.push_back(cv::Point2f(1, shortestEdgeWidth-1));
    rectifiedCorners.push_back(cv::Point2f(1, 1));
  } else {
    rectifiedCorners.push_back(cv::Point2f(shortestEdgeWidth, 0));
    rectifiedCorners.push_back(cv::Point2f(shortestEdgeWidth, shortestEdgeWidth));
    rectifiedCorners.push_back(cv::Point2f(0, shortestEdgeWidth));
    rectifiedCorners.push_back(cv::Point2f(0, 0));
  }
  cv::Mat T = cv::getPerspectiveTransform(quad.corners, rectifiedCorners);
  cv::warpPerspective(img, quadImg, T, cv::Size(shortestEdgeWidth, shortestEdgeWidth),
      cv::INTER_LINEAR);

  if (grayscale && quadImg.channels() != 1) {
    cv::cvtColor(quadImg, quadImg, CV_RGB2GRAY);
  }

  return quadImg;
};


cv::Mat trimFTag2Quad(cv::Mat tag, double maxStripAvgDiff) {
  const unsigned int numRows = tag.rows;
  const unsigned int numCols = tag.cols;

  cv::Mat tagGray;
  if (tag.channels() != 1) {
    cv::cvtColor(tag, tagGray, CV_RGB2GRAY);
  } else {
    tagGray = tag;
  }

  double row1Avg = double(sum(tagGray.row(0))[0])/numCols;
  double row2Avg = double(sum(tagGray.row(1))[0])/numCols;
  double row3Avg = double(sum(tagGray.row(2))[0])/numCols;
  double rowRm1Avg = double(sum(tagGray.row(numRows-2))[0])/numCols;
  double rowRAvg = double(sum(tagGray.row(numRows-1))[0])/numCols;
  double col1Avg = double(sum(tagGray.col(0))[0])/numRows;
  double col2Avg = double(sum(tagGray.col(1))[0])/numRows;
  double col3Avg = double(sum(tagGray.col(2))[0])/numRows;
  double colCm1Avg = double(sum(tagGray.row(numCols-2))[0])/numRows;
  double colCAvg = double(sum(tagGray.row(numCols-1))[0])/numRows;
  int trimLeft = 0, trimRight = 0, trimTop = 0, trimBottom = 0;
  bool trim = false;

  if (row2Avg - row3Avg > maxStripAvgDiff) {
    trimTop = 2;
    trim = true;
  } else if (row1Avg - row2Avg > maxStripAvgDiff) {
    trimTop = 1;
    trim = true;
  }
  if (rowRAvg - rowRm1Avg > maxStripAvgDiff) {
    trimBottom = 1;
    trim = true;
  }
  if (col2Avg - col3Avg > maxStripAvgDiff) {
    trimLeft = 2;
    trim = true;
  } else if (col1Avg - col2Avg > maxStripAvgDiff) {
    trimLeft = 1;
    trim = true;
  }
  if (colCAvg - colCm1Avg > maxStripAvgDiff) {
    trimRight = 1;
    trim = true;
  }

  cv::Mat trimmedTag;
  if (trim) {
    trimmedTag = tag(cv::Range(trimTop, numRows - trimBottom),
        cv::Range(trimLeft, numCols - trimRight));
  } else {
    trimmedTag = tag;
  }

  return trimmedTag;
};


cv::Mat extractHorzRays(cv::Mat croppedTag, unsigned int numSamples,
    unsigned int numRays, bool markRays) {
  double rowHeight = double(croppedTag.rows)/numRays;
  cv::Mat rays = cv::Mat::zeros(numRays, croppedTag.cols, CV_64FC1);
  cv::Mat tagRayF;
  unsigned int i, j;
  int quadRow;
  for (i = 0; i < numRays; i++) {
    cv::Mat raysRow = rays.row(i);
    for (j = 0; j < numSamples; j++) {
      quadRow = rowHeight * (i + double(j + 1)/(numSamples + 1));
      cv::Mat tagRay = croppedTag.row(quadRow);
      tagRay.convertTo(tagRayF, CV_64FC1);
      raysRow += tagRayF;
      if (markRays) { tagRay.setTo(255); }
    }
  }
  if (numSamples > 1) {
    rays /= numSamples;
  }
  return rays;
};


void solvePose(const std::vector<cv::Point2f> cornersPx, double quadSizeM,
    cv::Mat cameraIntrinsic, cv::Mat cameraDistortion,
    double& tx, double &ty, double& tz,
    double& rw, double& rx, double& ry, double& rz) {
  std::vector<cv::Point3d> spatialPoints;
  cv::Mat transVec, rotVec, rotMat;

  double quadSizeHalved = quadSizeM / 2;
  // NOTE: recall that the tag's frame adheres to a right-handed coordinate
  //       system, where +x: right of tag, +y: bottom of tag, +z: into tag
  spatialPoints.push_back(cv::Point3d( quadSizeHalved, -quadSizeHalved, 0.0)); // tag's top-right corner
  spatialPoints.push_back(cv::Point3d( quadSizeHalved,  quadSizeHalved, 0.0)); // tag's bottom-right corner
  spatialPoints.push_back(cv::Point3d(-quadSizeHalved,  quadSizeHalved, 0.0)); // tag's bottom-left corner
  spatialPoints.push_back(cv::Point3d(-quadSizeHalved, -quadSizeHalved, 0.0)); // tag's top-left corner

  // NOTE: OpenCV's solvePnP doc states that rvec & tvec together "brings points
  //       from the model coordinate system to the camera coordinate system".
  //       Empirically, these results define a transformation matrix T
  //       such that point_in_camera_frame = T * point_in_tag_frame. This is
  //       consistent with FTag2Pose's internal structure, where the tag's
  //       position and orientation is defined with respect to the camera's
  //       static/world frame.
  cv::solvePnP(spatialPoints, cornersPx, cameraIntrinsic, cameraDistortion,
      rotVec, transVec, false, CV_ITERATIVE);
  // CV_P3P and especially CV_EPNP confirmed to produce worse pose estimate
  // than CV_ITERATIVE, when not using a pose prior
  cv::Rodrigues(rotVec, rotMat);
  vc_math::rotMat2quat(rotMat, rw, rx, ry, rz);

  tx = transVec.at<double>(0);
  ty = transVec.at<double>(1);
  tz = transVec.at<double>(2);
};


std::vector<cv::Point2f> backProjectQuad(double cam_pose_in_tag_frame_x, double cam_pose_in_tag_frame_y,
		double cam_pose_in_tag_frame_z, double cam_rot_in_tag_frame_w, double cam_rot_in_tag_frame_x,
		double cam_rot_in_tag_frame_y, double cam_rot_in_tag_frame_z, double quadSizeM,
		cv::Mat cameraIntrinsic, cv::Mat cameraDistortion) {

	std::vector<cv::Point3d> tag_corners_in_tag_frame;
	double quadSizeHalved = quadSizeM / 2;
	tag_corners_in_tag_frame.push_back(cv::Point3d(-quadSizeHalved, -quadSizeHalved, 0.0));
	tag_corners_in_tag_frame.push_back(cv::Point3d(-quadSizeHalved,  quadSizeHalved, 0.0));
	tag_corners_in_tag_frame.push_back(cv::Point3d( quadSizeHalved,  quadSizeHalved, 0.0));
	tag_corners_in_tag_frame.push_back(cv::Point3d( quadSizeHalved, -quadSizeHalved, 0.0));

	cv::Point3d cam_pose_in_tag_frame( cam_pose_in_tag_frame_x, cam_pose_in_tag_frame_y, cam_pose_in_tag_frame_z);
	cv::Mat cam_rot_in_tag_frame = vc_math::quat2RotMat(cam_rot_in_tag_frame_w, cam_rot_in_tag_frame_x, cam_rot_in_tag_frame_y, cam_rot_in_tag_frame_z);
//	cv::Mat tag_rot_in_cam_frame = cam_rot_in_tag_frame.t();
//	cv::Mat mat_tag_pose_in_cam_frame =  -1.0*tag_rot_in_cam_frame*cv::Mat(cam_pose_in_tag_frame);
//	cv::Point3d tag_pose_in_cam_frame(mat_tag_pose_in_cam_frame);

//	std::cout << "Center: ( " << cam_pose_in_tag_frame_x << ", " << cam_pose_in_tag_frame_y << ", " << cam_pose_in_tag_frame_z << " )\t" << std::endl;
//	std::cout << "tag_corners_in_tag_frame: ";
//	for ( cv::Point3d& cor: tag_corners_in_tag_frame )
//	{
//		std::cout << "( " << cor.x << ", " << cor.y << ", " << cor.z << " )\t" << std::endl;
//	}
//	std::cout << std::endl;
//
//	std::cout << "cam_rot_in_tag_frame': " << std::endl << cv::format(cam_rot_in_tag_frame, "matlab") << std::endl;

	cv::Mat rotVec = cv::Mat(1, 3, CV_64FC1);
	cv::Rodrigues(cam_rot_in_tag_frame, rotVec);

	cv::Mat transVec = cv::Mat(1, 3, CV_64FC1);
	transVec.at<double>(0) = cam_pose_in_tag_frame.x;
	transVec.at<double>(1) = cam_pose_in_tag_frame.y;
	transVec.at<double>(2) = cam_pose_in_tag_frame.z;

	cv::Mat cornersMat;
//	cv::projectPoints(tag_corners_in_tag_frame, rotVec, transVec, cameraIntrinsic, cameraDistortion, cornersMat);
	cv::projectPoints(tag_corners_in_tag_frame, rotVec, transVec, cameraIntrinsic, cameraDistortion, cornersMat);

//	std::cout << "cornersMat: " << std::endl << cv::format( cornersMat, "matlab") << std::endl;

	std::vector<cv::Point2f> cornersPx;
	for ( int row = 0; row < cornersMat.rows; row++ )
	{
		cv::Point2f pt( cornersMat.at<double>(row,0), cornersMat.at<double>(row,1) );
		cornersPx.push_back(pt);
	}

//	std::cout << "Corners: " << std::endl;
//	for ( cv::Point2f& cor: cornersPx )
//	{
//		std::cout << "( " << cor.x << ", " << cor.y << " )\t" << std::endl;
//	}
//	std::cout << std::endl;
	return cornersPx;
}
