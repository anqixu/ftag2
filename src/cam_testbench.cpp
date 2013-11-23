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
      latestProfTime(ros::Time::now()),
      waitKeyDelay(30) {
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
    local_nh.param("waitkey_delay", waitKeyDelay, waitKeyDelay);

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
    //namedWindow("debug", CV_GUI_EXPANDED);
    namedWindow("edgels", CV_GUI_EXPANDED);
    //namedWindow("accum", CV_GUI_EXPANDED);
    //namedWindow("lines", CV_GUI_EXPANDED);
    namedWindow("segments", CV_GUI_EXPANDED);
    namedWindow("quads", CV_GUI_EXPANDED);
    namedWindow("quad_1", CV_GUI_EXPANDED);
    namedWindow("quad_2", CV_GUI_EXPANDED);
    namedWindow("quad_3", CV_GUI_EXPANDED);
    namedWindow("quad_4", CV_GUI_EXPANDED);

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
      std::list<Quad> quads = detectQuads(segments,
          params.houghBlurThetaWidth*degree,
          params.houghMaxSegmentGap); // TODO: 0 stop borrowing outdated params; create cam_testbench2 with proper dyncfg instead
      cv::Mat quadsImg = sourceImgRot.clone();
      //drawLineSegments(quadsImg, segments);
      std::cout << "found " << quads.size() << " quads" << std::endl;
      drawQuads(quadsImg, quads);
      cv::imshow("quads", quadsImg);

      std::list<Quad>::iterator quadIt = quads.begin();
      if (quads.size() >= 1) {
        cv::imshow("quad_1", extractQuadImg(sourceImgRot, *quadIt));
      }
      if (quads.size() >= 2) {
        quadIt++;
        cv::Mat q = extractQuadImg(sourceImgRot, *quadIt);
        cv::imshow("quad_2", q);
      }
      if (quads.size() >= 3) {
        quadIt++;
        cv::Mat q = extractQuadImg(sourceImgRot, *quadIt);
        cv::imshow("quad_3", q);
      }
      if (quads.size() >= 4) {
        quadIt++;
        cv::Mat q = extractQuadImg(sourceImgRot, *quadIt);
        cv::imshow("quad_4", q);
      }

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
      c = waitKey(waitKeyDelay);
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

  int waitKeyDelay;
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
