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
    cv::Point2d& intPt, double* distSegAInt = NULL, double* distSegBInt = NULL) {
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

  if (s >= 0 && s <= 1 && t >= 0 && t <= 1) {
    return 1;
  }
  return 0;
};


inline bool areEndpointsNear(const cv::Vec4i& segA, const cv::Vec4i& segB,
    double minEndptDist) {
  return ((vc_math::dist(segA[0], segA[1], segB[0], segB[1]) <= minEndptDist) ||
          (vc_math::dist(segA[0], segA[1], segB[2], segB[3]) <= minEndptDist) ||
          (vc_math::dist(segA[2], segA[3], segB[0], segB[1]) <= minEndptDist) ||
          (vc_math::dist(segA[2], segA[3], segB[2], segB[3]) <= minEndptDist));
};


inline bool overlap(const cv::Vec4i& a, const cv::Vec4i& b) {
  return (
    (a[0] == b[0]) ||(a[0] == b[1]) || (a[0] == b[2]) || (a[0] == b[3]) ||
    (a[1] == b[0]) ||(a[1] == b[1]) || (a[1] == b[2]) || (a[1] == b[3]) ||
    (a[2] == b[0]) ||(a[2] == b[1]) || (a[2] == b[2]) || (a[2] == b[3]) ||
    (a[3] == b[0]) ||(a[3] == b[1]) || (a[3] == b[2]) || (a[3] == b[3]));
};


// WARNING: this function uses the lengths of the segments to estimate the
//          area of the quad, which implicitly assumes that the segment
//          endpoints are sufficiently close to the corners of the quad
double area(const cv::Vec4i& segA, const cv::Vec4i& segB,
    const cv::Vec4i& segC, const cv::Vec4i& segD) {
  double lenA = vc_math::dist(segA);
  double lenB = vc_math::dist(segB);
  double lenC = vc_math::dist(segC);
  double lenD = vc_math::dist(segD);
  double angleAD = std::acos(vc_math::dot(segA, segD)/lenA/lenD);
  double angleBC = std::acos(vc_math::dot(segB, segC)/lenB/lenC);
  return 0.5*(lenA*lenD*std::sin(angleAD) + lenB*lenC*std::sin(angleBC));
};


// Flags smaller quad by changing its first segment idx to -1
//
// WARNING: this function uses the lengths of the segments to estimate the
//          area of the quad, which implicitly assumes that the segment
//          endpoints are sufficiently close to the corners of the quad
void flagSmallerQuad(cv::Vec4i& segQuadA, cv::Vec4i& segQuadB,
    const std::vector<int>& toOrigSegIDs, const std::vector<cv::Vec4i>& segments) {
  const cv::Vec4i& segAA = segments[toOrigSegIDs[segQuadA[0]]];
  const cv::Vec4i& segAB = segments[toOrigSegIDs[segQuadA[1]]];
  const cv::Vec4i& segAC = segments[toOrigSegIDs[segQuadA[2]]];
  const cv::Vec4i& segAD = segments[toOrigSegIDs[segQuadA[3]]];
  const cv::Vec4i& segBA = segments[toOrigSegIDs[segQuadB[0]]];
  const cv::Vec4i& segBB = segments[toOrigSegIDs[segQuadB[1]]];
  const cv::Vec4i& segBC = segments[toOrigSegIDs[segQuadB[2]]];
  const cv::Vec4i& segBD = segments[toOrigSegIDs[segQuadB[3]]];
  double areaQuadA = area(segAA, segAB, segAC, segAD);
  double areaQuadB = area(segBA, segBB, segBC, segBD);
  if (areaQuadA > areaQuadB) {
    segQuadB[0] = -1;
  } else {
    segQuadA[0] = -1;
  }
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


/*
// NOTE: This alternative implementation allows segments to be visited at most
 *       once. This returns unstable results when there are 2 quads that share
 *       2 segments (i.e. share a corner).
void completeQuadDFT(const std::vector< std::list<unsigned int> >& adjList,
    std::vector<cv::Vec4i>& quads,
    cv::Vec4i& currQuad,
    std::vector<bool>& visited,
    unsigned int startIdx, unsigned int currIdx, int depth) {
  currQuad[depth] = currIdx;
  if (currIdx != startIdx) visited[currIdx] = true;

  if (depth > 3) {
    return;
  } else if (depth == 3) {
    for (unsigned int adjIdx: adjList[currIdx]) {
      if (adjIdx == startIdx) {
        quads.push_back(vc_math::minCyclicOrder(currQuad));
        return;
      }
    }
  } else {
    for (unsigned int adjIdx: adjList[currIdx]) {
      if (adjIdx == startIdx) continue; // found a cycle shorter than a quad
      if (!visited[adjIdx])
        completeQuadDFT(adjList, quads, currQuad, visited, startIdx, adjIdx, depth + 1);
    }
  }
};
 */

void completeQuadDFT(const std::vector< std::list<unsigned int> >& adjList,
    std::vector<cv::Vec4i>& quads,
    cv::Vec4i& currQuad,
    unsigned int startIdx, unsigned int currIdx, int depth) {
  currQuad[depth] = currIdx;

  if (depth > 3) {
    return;
  } else if (depth == 3) {
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


std::list<Quad> detectQuads(const std::vector<cv::Vec4i>& segments,
    double intSegMinAngle, double minEndptDist) {
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
      // Do not connect nearby segments with sharp angles in between them
      if (vc_math::angularDist(
          vc_math::orientation(segments[i]),
          vc_math::orientation(segments[j]),
          vc_math::pi) < intSegMinAngle) { continue; }

      intersect = getSegmentIntersection(segments[i], segments[j], intPt,
          &distSegAInt, &distSegBInt);
      // Connect segments whose endpoints are nearby, and also whose
      // intersecting point is also near each of the segments' endpoints
      // (specifically where the triangle between the 2 endpoints and the
      // intersecting point is at most as sharp as a triangle with sides
      // endptThresh, 2*endptThresh, 2*endptThresh)
      if (intersect == 0 &&
          distSegAInt <= 2*minEndptDist &&
          distSegBInt <= 2*minEndptDist &&
          areEndpointsNear(segments[i], segments[j], minEndptDist)) {
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

  // TEMP: Display adjascent edges (blue -> red)
  /*
  char c;
  bool alive = true;
  bool done = false;
  cv::namedWindow("adj");
  cv::Mat segmentsImg, overlaidImg;
  img.copyTo(segmentsImg);
  drawLineSegments(segmentsImg, segments);
  for (i = 0; i < segments.size(); i++) {
    for (int jj: adjList[i]) {
      std::cout << i << " -> " << jj << std::endl;
      segmentsImg.copyTo(overlaidImg);
      cv::line(overlaidImg, cv::Point2i(segments[i][0], segments[i][1]), cv::Point2i(segments[i][2], segments[i][3]), CV_RGB(0, 0, 255), 3);
      cv::line(overlaidImg, cv::Point2i(segments[jj][0], segments[jj][1]), cv::Point2i(segments[jj][2], segments[jj][3]), CV_RGB(255, 0, 0), 3);
      cv::imshow("adj", overlaidImg);
      c = cv::waitKey();
      if ((c & 0x0FF) == 'x' || (c & 0x0FF) == 'X') {
        alive = false;
        done = true;
        break;
      } else if ((c & 0x0FF) == 'k' || (c & 0x0FF) == 'K') {
        done = true;
        break;
      }
    }
    if (done) { break; }
  }
  return alive;
  */

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

  // Find pairs of quads that share one or more edges with each other, and
  // remove the smaller quad among each pair found
  std::vector<cv::Vec4i>::iterator segQuadA, segQuadB;
  const std::vector<cv::Vec4i>::iterator segQuadEnd = segQuads.end();
  for (segQuadA = segQuads.begin(); segQuadA != segQuadEnd; segQuadA++) {
    if ((*segQuadA)[0] < 0) continue;

    segQuadB = segQuadA; segQuadB++;
    for (; segQuadB != segQuadEnd; segQuadB++) {
      if ((*segQuadB)[0] < 0) continue;
      if (overlap(*segQuadA, *segQuadB)) {
        flagSmallerQuad(*segQuadA, *segQuadB, toOrigSegIDs, segments);
      }
    }
  }
  segQuadA = segQuads.begin();
  while (segQuadA != segQuads.end()) {
    if ((*segQuadA)[0] < 0) {
      segQuadA = segQuads.erase(segQuadA);
    } else {
      segQuadA++;
    }
  }

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
    quads.push_back(quad);
  }

  // TEMP: show results in connectivity
  /*
  cv::Mat segmentsM(segments);
  segmentsM = segmentsM.reshape(1, segments.size());
  cv::Mat intSegIDs(toOrigSegIDs);
  cv::Mat quadsM(segQuads);
  quadsM = quadsM.reshape(1, quads.size());
  std::cout << "segments = ..." << std::endl << cv::format(segmentsM, "matlab") << ";" << std::endl << std::endl;
  std::cout << "intSegIDs = ..." << std::endl << cv::format(intSegIDs, "matlab") << ";" << std::endl << std::endl;
  std::cout << "adj = ..." << std::endl << cv::format(redAdj, "matlab") << ";" << std::endl << std::endl;
  std::cout << "quadsCyclesAdj = ..." << std::endl << cv::format(quadsCyclesAdj, "matlab") << ";" << std::endl << std::endl;
  std::cout << "segQuads = ..." << std::endl << cv::format(quadsM, "matlab") << ";" << std::endl << std::endl;
  */

  // TODO: 1 Determine possibly-obstructed quads by 5-multiplying and finding co-linear end-segments

  return quads;
};


cv::Mat extractQuadImg(cv::Mat img, Quad& quad) {
  int shortestEdgeWidth = floor(
      std::min(std::min(vc_math::dist(quad.corners[0], quad.corners[1]),
                        vc_math::dist(quad.corners[1], quad.corners[2])),
               std::min(vc_math::dist(quad.corners[2], quad.corners[3]),
                        vc_math::dist(quad.corners[3], quad.corners[0]))));

  cv::Mat quadImg;
  std::vector<cv::Point2f> rectifiedCorners;
  rectifiedCorners.push_back(cv::Point2f(0, 0));
  rectifiedCorners.push_back(cv::Point2f(shortestEdgeWidth, 0));
  rectifiedCorners.push_back(cv::Point2f(shortestEdgeWidth, shortestEdgeWidth));
  rectifiedCorners.push_back(cv::Point2f(0, shortestEdgeWidth));
  cv::Mat T = cv::getPerspectiveTransform(quad.corners, rectifiedCorners);
  cv::warpPerspective(img, quadImg, T, cv::Size(shortestEdgeWidth, shortestEdgeWidth),
      cv::INTER_LINEAR);
  return quadImg;
};


#ifdef DAVIDS_CODE
void extractRays(...) {
cv::Mat horiz_rays;
cv::Mat vert_rays;
int nRaysPerRow = 3;
for( unsigned int i=0; i<quads.size(); i++ )
{
  cv::Mat trimmedQuad = quads[i];
  cv::imshow("QuadNoBorder", trimmedQuad);

  vert_rays = extract_n_Vert_Rays_Per_Col(trimmedQuad,nRaysPerRow);
  cv::Mat plot8U = cv::Mat::zeros(vert_rays.rows,vert_rays.cols, CV_8U);
  vert_rays.convertTo(plot8U,CV_8U);
  cv::imshow("V1", plot8U);
  cv::waitKey();
  cv::Mat plot = cv::Mat::zeros(255, vert_rays.cols, CV_32S);
  for( int j = 0; j<vert_rays.cols-1; j++ )
  {
    plot.at<int>(255-vert_rays.at<int>(0,j),j) = 255;
  }
  plot.convertTo(plot8U,CV_8U);
  cv::imshow("V2", plot8U);

  horiz_rays = extract_n_Horiz_Rays_Per_Row(trimmedQuad,nRaysPerRow);
  plot8U = cv::Mat::zeros(horiz_rays.rows,vert_rays.cols, CV_8U);
  horiz_rays.convertTo(plot8U,CV_8U);
  cv::imshow("H1", plot8U);
  cv::waitKey();
  plot = cv::Mat::zeros(255, horiz_rays.cols, CV_32S);
  for( int j = 0; j<horiz_rays.cols-1; j++ )
  {
    //cout << (int)vert_rays.at<int>(0,j) << ", ";
    plot.at<int>(255-horiz_rays.at<int>(0,j),j) = 255;
  }
  plot.convertTo(plot8U,CV_8U);
  cv::imshow("H2", plot8U);
  cv::waitKey();
}


cv::Mat extract_n_Horiz_Rays_Per_Row(Mat quad, int n)
{
  float rowHeight = quad.rows/6.0;
  cv::Mat rays_added = Mat::zeros(6, quad.cols, CV_8U);
  for ( unsigned int i=0; i<6; i++ )
  {
    Mat addedRow = rays_added.row(i);
    for ( int j=0; j<n; j++ )
    {
      int row = (rowHeight)*i + rowHeight/(2*n) + j*rowHeight/n;
      Mat newRow = quad.row(row);
      addedRow = newRow/n + addedRow; // I'm currently averaging to make displaying
                      // easier but the division by n can be removed
    }
  }
//  cv::imshow("HorRAYS", rays_added);
  Mat raysCV32 ;
  rays_added.convertTo(raysCV32,CV_32S);
  return raysCV32;
}

cv::Mat extract_n_Vert_Rays_Per_Col(cv::Mat quad, int n)
{
  Mat q = quad.t();
  Mat raysCV32 = extract_n_Horiz_Rays_Per_Row(q,n);
  return raysCV32;
}


#endif
