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
//#define ENABLE_PROFILER


using namespace std;
using namespace std::chrono;
using namespace cv;
using namespace vc_math;


typedef dynamic_reconfigure::Server<ftag2::CamTestbenchConfig> ReconfigureServer;


/**
 * Returns 0 if the line segments intersect, 1 if their lines
 * intersect, and -1 if they are co-linear.
 * In addition, the intersection point is stored in intPt if available.
 */
char getSegmentIntersection(const cv::Vec4i& segA, const cv::Vec4i& segB, cv::Point2d& intPt) {
  double s1_x, s1_y, s2_x, s2_y, det, dx, dy, s, t;
  s1_x = segA[2] - segA[0]; s1_y = segA[3] - segA[1];
  s2_x = segB[2] - segB[0]; s2_y = segB[3] - segB[1];
  det = (-s2_x * s1_y + s1_x * s2_y);
  if (det == 0) {
    intPt.x = 0; intPt.y = 0;
    return -1;
  }

  dx = segA[0] - segB[0];
  dy = segA[1] - segB[1];
  s = (-s1_y * dx + s1_x * dy) / det;
  t = ( s2_x * dy - s2_y * dx) / det;
  intPt.x = segA[0] + t*s1_x;
  intPt.y = segA[1] + t*s1_y;

  if (s >= 0 && s <= 1 && t >= 0 && t <= 1) {
    return 1;
  }
  return 0;
};


inline bool areEndpointsNear(const cv::Vec4i& segA, const cv::Vec4i& segB, double endptThresh) {
  return ((dist(segA[0], segA[1], segB[0], segB[1]) <= endptThresh) ||
          (dist(segA[0], segA[1], segB[2], segB[3]) <= endptThresh) ||
          (dist(segA[2], segA[3], segB[0], segB[1]) <= endptThresh) ||
          (dist(segA[2], segA[3], segB[2], segB[3]) <= endptThresh));
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


bool detectQuads(const std::vector<cv::Vec4i> segments, double endptThresh, const cv::Mat img) { // TODO: 000 remove img after debug
  // Identify connected segments
  // TODO: 0 filter out segments with steep angles
  std::vector< std::list<unsigned int> > adjList(segments.size());
  std::vector<bool> incomingAdj(segments.size(), false); // whether segment has incoming adjacent segment(s)
  unsigned int i, j;
  cv::Point2d intPt;
  char intersect;
  for (i = 0; i < segments.size(); i++) {
    for (j = i+1; j < segments.size(); j++) {
      intersect = getSegmentIntersection(segments[i], segments[j], intPt);
      if (intersect == 0 && areEndpointsNear(segments[i], segments[j], endptThresh)) {
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

  // Keep only intersecting edgels and create reduced adjacency matrix
  std::vector<int> toIntSegIDs(segments.size(), -1);
  std::vector<int> toOrigSegIDs;
  for (i = 0; i < segments.size(); i++) {
    if (adjList[i].size() > 0 || incomingAdj[i]) {
      j = toOrigSegIDs.size();
      toIntSegIDs[i] = j;
      toOrigSegIDs.push_back(i);
    }
  }
  cv::Mat adj = cv::Mat::zeros(toOrigSegIDs.size(), toOrigSegIDs.size(), CV_64FC1);
  for (j = 0; j < toOrigSegIDs.size(); j++) {
    i = toOrigSegIDs[j];
    for (unsigned int neighI: adjList[i]) {
      adj.at<double>(j, toIntSegIDs[neighI]) = 1;
    }
  }

  // Determine quads by 4-multiplying
  cv::Mat quadsCyclesAdj = adj*adj*adj*adj;
  std::vector<int> quad_idx;
  std::vector<int> quad_k(quadsCyclesAdj.rows, 0);
  for (i = 0; i < toOrigSegIDs.size(); i++) {
    if (quadsCyclesAdj.at<double>(i, i) > 0) {
      quad_k[i] = 1;
      quad_idx.push_back(i);
    }
  }

  // TEMP: show results in connectivity
  cv::Mat segmentsM(segments);
  segmentsM = segmentsM.reshape(1, segments.size());
  cv::Mat intSegIDs(toOrigSegIDs);
  std::cout << "segments = " << std::endl << cv::format(segmentsM, "matlab") << std::endl << std::endl;
  std::cout << "intSegIDs = " << std::endl << cv::format(intSegIDs, "matlab") << std::endl << std::endl;
  std::cout << "adj = " << std::endl << cv::format(adj, "matlab") << std::endl << std::endl;
  std::cout << "quadsCyclesAdj = " << std::endl << cv::format(quadsCyclesAdj, "matlab") << std::endl << std::endl;

  // Determine possible obstructed quads by 5-multiplying

  // Show result

  #ifdef DAVIDS_CODE
  std::vector<cv::Vec4i > quad_vect;
  cv::Vec4i quad_current;

  // FOR ALL THE CYCLES FOUND OF LENGTH 4, TRACE USING DFS AND EXTRACT QUADS
  for ( unsigned int i = 0; i < quad_idx.size(); i++)
  {
    cout << "Node " << i << ":\tAdj. list: " ;
    for ( unsigned int j = 0; j < adj_list[quad_idx[i]].size(); j++ )
      cout << adj_list[quad_idx[i]][j] << "\t";
    cout << endl;
    DFS(adj_list, quad_vect, quad_current, quad_k, quad_idx[i], quad_idx[i], 0);
  }
  std::sort (quad_vect.begin(), quad_vect.end(), vec4iCompare);

  displayQuadList(quad_vect);

  // REMOVE DUPLICATES
  for ( unsigned int i=0; i < quad_vect.size()-1; i++ )
  {
    cout << "i= " << i << ":\t";
    unsigned int j = i+1;
    bool sameQuad;
    cv::Vec4i v1 = quad_vect[i];
    do
    {
      cv::Vec4i v2 = quad_vect[j];
      sameQuad = ( vec4iCompare(v1,v2) == false && vec4iCompare(v2,v1) == false );
      cout << sameQuad << "\t";
      if (sameQuad==true)
        quad_vect.erase(quad_vect.begin()+j);
      else
        j++;
    }while(sameQuad==true && j<quad_vect.size());
    cout << endl;
  }

  cout << "NO MORE DUPLICATESS: " << endl;
  displayQuadList(quad_vect);
  // DISPLAY ALL QUADS
  for (unsigned int i = 0; i < quad_vect.size(); i++)
  {
    cv::Vec4i qi = quad_vect[i];
    cv::Vec4i li0 = lines_Backup[qi[0]];
    cv::Vec4i li1 = lines_Backup[qi[1]];
    cv::Vec4i li2 = lines_Backup[qi[2]];
    cv::Vec4i li3 = lines_Backup[qi[3]];
    cv::line(quads_Image, cv::Point(li0[0], li0[1]), cv::Point(li0[2], li0[3]), CV_RGB(0,255,0),2);
    cv::line(quads_Image, cv::Point(li1[0], li1[1]), cv::Point(li1[2], li1[3]), CV_RGB(0,255,0),2);
    cv::line(quads_Image, cv::Point(li2[0], li2[1]), cv::Point(li2[2], li2[3]), CV_RGB(0,255,0),2);
    cv::line(quads_Image, cv::Point(li3[0], li3[1]), cv::Point(li3[2], li3[3]), CV_RGB(0,255,0),2);
  }
  cv::imshow("FINAL QUADS", quads_Image);
  cv::waitKey();

  // EXTRACT EXACT CORNERS OF QUADS
  std::vector< std::vector<cv::Point2f> > corners;
  for (unsigned int i = 0; i < quad_vect.size(); i++)
  {
    std::vector<cv::Point2f> quadCorner;
    cv::Vec4i qi = quad_vect[i];
    for (int j = 0; j<4; j++)
    {
      cv::Point2f pt;
      pt.x = intMat_x.at<int>(qi[j],qi[(j+1)%4]);
      pt.y = intMat_y.at<int>(qi[j],qi[(j+1)%4]);
      quadCorner.push_back(pt);
    }
    corners.push_back(quadCorner);
  }

  cout << "ALL CORNERS OF ALL QUADS: " << endl;
  for(unsigned int i=0; i<corners.size(); i++)
  {
    cout << "Quad " << i << ":\t";
    for(unsigned int j=0; j<corners[i].size(); j++ )
    {
      cout << "( " << corners[i][j].x << ", " << corners[i][j].y << " )\t";
    }
    cout << endl;
  }

  std::vector<float> quad_shortest_edge_size;
  for (unsigned int i = 0; i < quad_vect.size(); i++)
  {
    cv::Point2f p0 = corners[i][0];
    cv::Point2f p1 = corners[i][1];
    cv::Point2f p2 = corners[i][2];
    cv::Point2f p3 = corners[i][3];

    float length1 = sqrt( (p0.x-p1.x)*(p0.x-p1.x) + (p0.y-p1.y)*(p0.y-p1.y) );
    float length2 = sqrt( (p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) );
    float length3 = sqrt( (p2.x-p3.x)*(p2.x-p3.x) + (p2.y-p3.y)*(p2.y-p3.y) );
    float length4 = sqrt( (p3.x-p0.x)*(p3.x-p0.x) + (p3.y-p0.y)*(p3.y-p0.y) );
    float minLength = min(length1, min(length2, min(length3,length4)));
    quad_shortest_edge_size.push_back(minLength);
  }

  // Get Perspective Transform only for 1st quad
  cv::Mat quad;
  std::vector< cv::Mat > quads;
  for(unsigned  int i=0; i<corners.size(); i++)
  {
    cv::Mat quad = cv::Mat::zeros((int)quad_shortest_edge_size[i], (int)quad_shortest_edge_size[i], CV_8U);
    std::vector<cv::Point2f> quad_pts;
    quad_pts.push_back(cv::Point2f(0, 0));
    quad_pts.push_back(cv::Point2f(quad.cols, 0));
    quad_pts.push_back(cv::Point2f(quad.cols, quad.rows));
    quad_pts.push_back(cv::Point2f(0, quad.rows));
    cv::Mat transmtx = cv::getPerspectiveTransform(corners[i], quad_pts);
    cv::warpPerspective(origbw, quad, transmtx, quad.size());
    std::ostringstream stringStream;
    //cv::imshow(stringStream.str(), quad);
    Mat trimmedQuad = quad.colRange(quad.cols / 8,7*quad.cols / 8);
    trimmedQuad = trimmedQuad.rowRange(quad.rows / 8,7*quad.rows / 8);
    quads.push_back(trimmedQuad);
  }

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
#endif
  return true;
};


#ifdef DAVIDS_CODE
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

void DFS(std::vector<std::vector <int> > adj_list, std::vector<cv::Vec4i> &quad_vect, cv::Vec4i &quad_current, std::vector<int> quad_k, int start_node, int current_node, int depth)
{
//  cout << endl << endl << "Depth: " << depth << "\t Current node: " << current_node << "\t Start node: " << start_node << endl;
  if ( current_node == start_node && depth == 4 )
  {
    cv::Vec4i aux (quad_current);
//    cout << endl << "Pushing quad: ";
    vec4iSort(aux, 0, 3);
/*    for (int i = 0; i<=depth; i++)
      cout << aux[i] << "\t" << endl << endl;
*/
    quad_vect.push_back(aux);
    return;
  }
  if ( depth > 4 )
    return;
  quad_current[depth] = current_node;
/*  cout << "Current quad: ";
  for (int i = 0; i<=depth; i++)
    cout << quad_current[i] << "\t";
  cout << endl;
*/
  if(current_node != start_node)
    quad_k[current_node] = -1;

  for (unsigned int i=0; i < (adj_list[current_node]).size(); i++)
  {
    int newNode = adj_list[current_node][i];
    if ( quad_k[newNode] <= 0 )
      continue;
    DFS(adj_list, quad_vect, quad_current, quad_k, start_node, newNode, depth+1);
  }
}

void vec4iSort(cv::Vec4i &vect4, int ini, int end)
{
   if (end<=ini)
     return;
   int mid = (ini+end)/2;
   vec4iSort(vect4, ini, mid);
   vec4iSort(vect4, mid+1,end);
   int i=ini;
   int j=mid+1;
   cv::Vec4i aux;
   int k = i;
   while(i<=mid && j<=end)
   {
     if(vect4[i]<=vect4[j])
     {
      aux[k] = vect4[i];
      i++;
     }
     else
     {
       aux[k] = vect4[j];
       j++;
     }
     k++;
   }
   while(i<=mid)
   {
     aux[k] = vect4[i];
     k++;
     i++;
   }

   while(j<=end)
   {
     aux[k] = vect4[j];
     k++;
     j++;
   }

   for(k=ini;k<=end;k++)
     vect4[k]=aux[k];
}

bool vec4iCompare (cv::Vec4i v1,cv::Vec4i v2)
{
  if(v1[0] < v2[0])
    return true;
  else
    return false;

  if(v1[1] < v2[1])
    return true;
  else
    return false;

  if(v1[2] < v2[2])
    return true;
  else
    return false;

  if(v1[3] < v2[3])
    return true;
  else
    return false;
}
#endif

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

    //namedWindow("source", CV_GUI_EXPANDED);
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
      double rho_res = params.houghRhoRes;
      double theta_res = params.houghThetaRes*degree;
      std::list<cv::Vec4i> lineSegments = detectLineSegmentsHough(grayImg,
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
      std::vector<cv::Vec4i> segments = detectLineSegments(grayImg,
          params.sobelThreshHigh, params.sobelThreshLow, params.sobelBlurWidth,
          (unsigned int) params.houghMinAccumValue, params.houghEdgelThetaMargin*degree,
          params.houghMinSegmentLength);
      sourceImgRot.copyTo(overlaidImg);
      drawLineSegments(overlaidImg, segments);
      cv::imshow("segments", overlaidImg);

      // Detect quads
      alive = detectQuads(segments, params.houghMaxSegmentGap, sourceImgRot);

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
      c = waitKey();
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
