#include "detector/FTag2Detector.hpp"
#include <cmath>


/**
 * Returns 1 if the line segments intersect, 0 if their lines intersect
 * beyond one or both segments, or -1 if they are co-linear.
 * In addition, the intersection point is stored in intPt if available.
 * Furthermore, minimum distance between endpoints and intersecting point
 * for each segment are returned as well.
 */
char getSegmentIntersection(const cv::Vec4i& segA, const cv::Vec4i& segB,
    cv::Point2d& intPt, double* distSegAInt = NULL, double* distSegBInt = NULL,
    double* segAIntRatio = NULL, double* segBIntRatio = NULL) {
  double s1_x, s1_y, s2_x, s2_y, det, dx, dy, s, t;
  s1_x = segA[2] - segA[0]; s1_y = segA[3] - segA[1];
  s2_x = segB[2] - segB[0]; s2_y = segB[3] - segB[1];
  det = (-s2_x * s1_y + s1_x * s2_y);
  if (fabs(det) <= 10*std::numeric_limits<double>::epsilon()) {
    intPt.x = 0; intPt.y = 0;
    return -1;
  }

  dx = segA[0] - segB[0];
  dy = segA[1] - segB[1];
  s = (-s1_y * dx + s1_x * dy) / det;
  t = ( s2_x * dy - s2_y * dx) / det;
  intPt.x = segA[0] + t*s1_x;
  intPt.y = segA[1] + t*s1_y;

  if (distSegAInt != NULL) {
    double tDist = (t <= 0.5) ? t : 1.0 - t;
    *distSegAInt = std::fabs(tDist) * std::sqrt(s1_x*s1_x+s1_y*s1_y);
  }
  if (distSegBInt != NULL) {
    double sDist = (s <= 0.5) ? s : 1.0 - s;
    *distSegBInt = std::fabs(sDist) * std::sqrt(s2_x*s2_x+s2_y*s2_y);
  }
  if (segAIntRatio != NULL) {
    *segAIntRatio = t;
  }
  if (segBIntRatio != NULL) {
    *segBIntRatio = s;
  }

  if (s >= 0 && s <= 1 && t >= 0 && t <= 1) {
    return 1;
  }
  return 0;
};


inline bool areEndpointsNear(const cv::Vec4i& segA, const cv::Vec4i& segB,
    double maxEndptDist) {
  return ((vc_math::dist(segA[0], segA[1], segB[0], segB[1]) <= maxEndptDist) ||
          (vc_math::dist(segA[0], segA[1], segB[2], segB[3]) <= maxEndptDist) ||
          (vc_math::dist(segA[2], segA[3], segB[0], segB[1]) <= maxEndptDist) ||
          (vc_math::dist(segA[2], segA[3], segB[2], segB[3]) <= maxEndptDist));
};


inline bool overlap(const cv::Vec4i& a, const cv::Vec4i& b) {
  return (
    (a[0] == b[0]) || (a[0] == b[1]) || (a[0] == b[2]) || (a[0] == b[3]) ||
    (a[1] == b[0]) || (a[1] == b[1]) || (a[1] == b[2]) || (a[1] == b[3]) ||
    (a[2] == b[0]) || (a[2] == b[1]) || (a[2] == b[2]) || (a[2] == b[3]) ||
    (a[3] == b[0]) || (a[3] == b[1]) || (a[3] == b[2]) || (a[3] == b[3]));
};


bool isClockwiseOrder(const cv::Vec4i& segA, const cv::Vec4i& segB, const cv::Point2d& intPt) {
  cv::Point2d endA1(segA[0], segA[1]);
  cv::Point2d endA2(segA[2], segA[3]);
  cv::Point2d endB1(segB[0], segB[1]);
  cv::Point2d endB2(segB[2], segB[3]);
  cv::Point2d vecA, vecB;
  double distA1 = vc_math::dist(intPt, endA1);
  double distA2 = vc_math::dist(intPt, endA2);
  double distB1 = vc_math::dist(intPt, endB1);
  double distB2 = vc_math::dist(intPt, endB2);

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


#ifdef DEPRECATED
std::list<Quad> detectQuadsOld(const std::vector<cv::Vec4i>& segments,
    double intSegMinAngle, double maxEndptDist) {
  std::list<Quad> quads;

  // Identify connected segments
  std::vector< std::list<unsigned int> > adjList(segments.size());
  std::vector<bool> incomingAdj(segments.size(), false); // whether segment has incoming adjacent segment(s)
  unsigned int i, j;
  cv::Point2d intPt;
  char intersect;
  double distSegAInt, distSegBInt;
  for (i = 0; i < segments.size(); i++) {
    for (j = i+1; j < segments.size(); j++) {
      // Do not connect nearby segments with sharp (or near-180') angles in between them
      double intAngle = vc_math::angularDist(vc_math::orientation(segments[i]),
          vc_math::orientation(segments[j]), vc_math::pi);
      if (intAngle < intSegMinAngle || intAngle > vc_math::pi - intSegMinAngle) { continue; }

      intersect = getSegmentIntersection(segments[i], segments[j], intPt,
          &distSegAInt, &distSegBInt);
      // Connect segments whose endpoints are nearby, and also whose
      // intersecting point is also near each of the segments' endpoints
      // (specifically where the triangle between the 2 endpoints and the
      // intersecting point is at most as sharp as a triangle with sides
      // endptThresh, 2*endptThresh, 2*endptThresh)
      if (intersect == 0 &&
          distSegAInt <= 2*maxEndptDist &&
          distSegBInt <= 2*maxEndptDist &&
          areEndpointsNear(segments[i], segments[j], maxEndptDist)) {
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
  //cv::Mat redAdj = cv::Mat::zeros(toOrigSegIDs.size(), toOrigSegIDs.size(), CV_32SC1);
  std::vector< std::list<unsigned int> > redAdjList(toOrigSegIDs.size());
  for (j = 0; j < toOrigSegIDs.size(); j++) {
    std::list<unsigned int>& currAdj = redAdjList[j];
    i = toOrigSegIDs[j];
    for (unsigned int neighI: adjList[i]) {
      unsigned int neighJ = toIntSegIDs[neighI];
      //redAdj.at<int>(j, neighJ) = 1;
      currAdj.push_back(neighJ);
    }
  }

  // Traverse through adjascency list and search for 4-connected 'complete' quads
  std::vector<cv::Vec4i> segQuads;
  cv::Vec4i currSegQuad;
  /*
  // NOTE: 4-multiplying the adjacency matrix is costlier than the rest of this
  //       function combined, so instead we will just perform DFT over all
  //       segments
  cv::Mat quadsCyclesAdj = redAdj*redAdj*redAdj*redAdj;
  std::vector<unsigned int> quadCandIdx;
  for (i = 0; i < toOrigSegIDs.size(); i++) {
    if (quadsCyclesAdj.at<int>(i, i) > 0) {
      quadCandVisited[i] = false;
      quadCandIdx.push_back(i);
    }
  }
  for (unsigned int currCandIdx: quadCandIdx) {
    completeQuadDFT(redAdjList, segQuads, currSegQuad, currCandIdx, currCandIdx, 0);
  }
  */
  for (unsigned int currCandIdx = 0; currCandIdx < toOrigSegIDs.size(); currCandIdx++) {
    completeQuadDFT(redAdjList, segQuads, currSegQuad, currCandIdx, currCandIdx, 0);
  }
  vc_math::unique(segQuads);

  // Compute corners of quads
  cv::Point2d corner;
  for (cv::Vec4i& segQuad: segQuads) {
    const cv::Vec4i& segA = segments[toOrigSegIDs[segQuad[0]]];
    const cv::Vec4i& segB = segments[toOrigSegIDs[segQuad[1]]];
    const cv::Vec4i& segC = segments[toOrigSegIDs[segQuad[2]]];
    const cv::Vec4i& segD = segments[toOrigSegIDs[segQuad[3]]];
    Quad quad;
    getSegmentIntersection(segA, segB, corner);
    quad.corners[0].x = corner.x; quad.corners[0].y = corner.y;
    getSegmentIntersection(segB, segC, corner);
    quad.corners[1].x = corner.x; quad.corners[1].y = corner.y;
    getSegmentIntersection(segC, segD, corner);
    quad.corners[2].x = corner.x; quad.corners[2].y = corner.y;
    getSegmentIntersection(segD, segA, corner);
    quad.corners[3].x = corner.x; quad.corners[3].y = corner.y;
    quad.updateArea();
    if (quad.area > 0) {
      quads.push_back(quad);
    }
  }

  return quads;
};
#endif


std::list<Quad> detectQuads(const std::vector<cv::Vec4i>& segments,
    double intSegMinAngle, double maxTIntDistRatio, double maxEndptDistRatio,
    double maxCornerGapEndptDistRatio,
    double maxEdgeGapDistRatio, double maxEdgeGapAlignAngle,
    double minQuadWidth) {
  std::list<Quad> quads;

  // Compute lengths of each segment
  std::vector<double> segmentLengths;
  for (const cv::Vec4i& seg: segments) {
    segmentLengths.push_back(vc_math::dist(seg));
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
      double intAngle = vc_math::angularDist(vc_math::orientation(segments[i]),
          vc_math::orientation(segments[j]), vc_math::pi);
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


cv::Mat extractQuadImg(const cv::Mat img, const Quad& quad, unsigned int minWidth, bool oversample, bool grayscale) {
  int shortestEdgeWidth = floor(
      std::min(std::min(vc_math::dist(quad.corners[0], quad.corners[1]),
                        vc_math::dist(quad.corners[1], quad.corners[2])),
               std::min(vc_math::dist(quad.corners[2], quad.corners[3]),
                        vc_math::dist(quad.corners[3], quad.corners[0]))));

  if (shortestEdgeWidth < int(minWidth)) return cv::Mat();

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


cv::Mat cropFTag2Border(cv::Mat tag, unsigned int numRays, unsigned int borderBlocks) {
  const unsigned int numBlocks = numRays + 2*borderBlocks;
  double hBorder = double(tag.cols)/numBlocks*borderBlocks;
  double vBorder = double(tag.rows)/numBlocks*borderBlocks;
  cv::Mat croppedTag = tag(cv::Range(std::round(vBorder), std::round(tag.rows - vBorder)),
      cv::Range(std::round(hBorder), std::round(tag.cols - hBorder)));
  return croppedTag;
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
//  spatialPoints.push_back(cv::Point3d(-quadSizeHalved, -quadSizeHalved, 0.0));
//  spatialPoints.push_back(cv::Point3d(-quadSizeHalved,  quadSizeHalved, 0.0));
//  spatialPoints.push_back(cv::Point3d( quadSizeHalved,  quadSizeHalved, 0.0));
//  spatialPoints.push_back(cv::Point3d( quadSizeHalved, -quadSizeHalved, 0.0));

  spatialPoints.push_back(cv::Point3d(-quadSizeHalved, -quadSizeHalved, 0.0));
  spatialPoints.push_back(cv::Point3d( quadSizeHalved, -quadSizeHalved, 0.0));
  spatialPoints.push_back(cv::Point3d( quadSizeHalved,  quadSizeHalved, 0.0));
  spatialPoints.push_back(cv::Point3d(-quadSizeHalved,  quadSizeHalved, 0.0));

  cv::solvePnP(spatialPoints, cornersPx, cameraIntrinsic, cameraDistortion,
      rotVec, transVec);
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



bool validateTagBorder(cv::Mat tag,
    double meanPxMaxThresh, double stdPxMaxThresh,
    unsigned int numRays, unsigned int borderBlocks) {
  const unsigned int numBlocks = numRays + 2*borderBlocks;
  double hBorder = double(tag.cols)/numBlocks*borderBlocks;
  double vBorder = double(tag.rows)/numBlocks*borderBlocks;
  cv::Mat borderMask = cv::Mat::ones(tag.size(), CV_8UC1);
  borderMask(cv::Range(std::round(vBorder), std::round(tag.rows - vBorder)),
      cv::Range(std::round(hBorder), std::round(tag.cols - hBorder))).setTo(0);
  cv::Scalar meanPx;
  cv::Scalar stdPx;
  meanStdDev(tag, meanPx, stdPx, borderMask);

  return ((meanPx[0] <= meanPxMaxThresh) && (stdPx[0] <= stdPxMaxThresh));
};
