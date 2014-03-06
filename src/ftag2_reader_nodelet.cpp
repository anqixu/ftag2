#include "detector/FTag2Detector.hpp"
#include "decoder/FTag2Decoder.hpp"

#include "common/Profiler.hpp"
#include "common/VectorAndCircularMath.hpp"

#include <ros/ros.h>
#include <dynamic_reconfigure/server.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <nodelet/nodelet.h>
#include "ftag2/FTag2ReaderConfig.h"
#include "ftag2/TagDetections.h"


using namespace std;
using namespace cv;
using namespace vc_math;


typedef dynamic_reconfigure::Server<ftag2::FTag2ReaderConfig> ReconfigureServer;


#define CV_SHOW_IMAGES


namespace ftag2 {


class FTag2ReaderNodelet : public nodelet::Nodelet {
protected:
  bool alive;

  ReconfigureServer* dynCfgServer;
  boost::recursive_mutex dynCfgMutex;
  bool dynCfgSyncReq;

  image_transport::Subscriber imageSub;
  image_transport::CameraSubscriber cameraSub;
  ros::Publisher tagDetectionsPub;
  image_transport::Publisher processedImagePub;
  image_transport::Publisher firstTagImagePub;

  ftag2::FTag2ReaderConfig params;

  cv::Mat cameraIntrinsic, cameraDistortion;

  PhaseVariancePredictor phaseVariancePredictor;

  // DEBUG VARIABLES
  Profiler lineSegP, quadP, quadExtractorP, decoderP, durationP, rateP;
  ros::Time latestProfTime;
  double profilerDelaySec;


public:
  FTag2ReaderNodelet() : nodelet::Nodelet(),
      alive(false),
      dynCfgServer(NULL),
      dynCfgSyncReq(false),
      latestProfTime(ros::Time::now()),
      profilerDelaySec(0) {
    // Set default parameter values
    params.sobelThreshHigh = 100;
    params.sobelThreshLow = 30;
    params.sobelBlurWidth = 3;
    params.lineAngleMargin = 20.0; // *degree
    params.lineMinEdgelsCC = 50;
    params.lineMinEdgelsSeg = 10;
    params.quadMinWidth = 15;
    params.quadMinAngleIntercept = 30.0;
    params.quadMaxEndptDistRatio = 0.1;
    params.quadMaxTIntDistRatio = 0.05;
    params.quadMaxCornerGapEndptDistRatio = 0.2;
    params.quadMaxEdgeGapDistRatio = 0.5;
    params.quadMaxEdgeGapAlignAngle = 10.0;
    params.quadMaxStripAvgDiff = 15.0;
    params.maxQuadsToScan = 10;
    params.phaseVarWeightR = 0;
    params.phaseVarWeightZ = 0;
    params.phaseVarWeightAngle = 0;
    params.phaseVarWeightFreq = 0;
    params.phaseVarWeightBias = 10*10;
    params.markerWidthM = 0.07;
  };


  ~FTag2ReaderNodelet() {
    alive = false;
  };


  virtual void onInit() {
    // Obtain node handles
    //ros::NodeHandle& nh = getNodeHandle();
    ros::NodeHandle& local_nh = getPrivateNodeHandle();

    // Load misc. non-dynamic parameters
    local_nh.param("profiler_delay_sec", profilerDelaySec, profilerDelaySec);

    // Load static camera calibration information
    std::string cameraIntrinsicStr, cameraDistortionStr;
    local_nh.param("camera_intrinsic", cameraIntrinsicStr, cameraIntrinsicStr);
    local_nh.param("camera_distortion", cameraDistortionStr, cameraDistortionStr);
    if (cameraIntrinsicStr.size() > 0) {
      cameraIntrinsic = str2mat(cameraIntrinsicStr, 3);
    } else {
      cameraIntrinsic = cv::Mat::zeros(3, 3, CV_64FC1); // prepare buffer for camera_info
    }
    if (cameraDistortionStr.size() > 0) {
      cameraDistortion = str2mat(cameraDistortionStr, 1);
    } else {
      cameraDistortion = cv::Mat::zeros(1, 5, CV_64FC1); // prepare buffer for camera_info
    }

    // Setup and initialize dynamic reconfigure server
    dynCfgServer = new ReconfigureServer(dynCfgMutex, local_nh);
    dynCfgServer->setCallback(bind(&FTag2ReaderNodelet::configCallback, this, _1, _2));

    // Parse static parameters and update dynamic reconfigure values
    #define GET_PARAM(v) \
      local_nh.param(std::string(#v), params.v, params.v)
    GET_PARAM(sobelThreshHigh);
    GET_PARAM(sobelThreshLow);
    GET_PARAM(sobelBlurWidth);
    GET_PARAM(lineAngleMargin);
    GET_PARAM(lineMinEdgelsCC);
    GET_PARAM(lineMinEdgelsSeg);
    GET_PARAM(quadMinWidth);
    GET_PARAM(quadMinAngleIntercept);
    GET_PARAM(quadMaxEndptDistRatio);
    GET_PARAM(quadMaxTIntDistRatio);
    GET_PARAM(quadMaxCornerGapEndptDistRatio);
    GET_PARAM(quadMaxEdgeGapDistRatio);
    GET_PARAM(quadMaxEdgeGapAlignAngle);
    GET_PARAM(quadMaxStripAvgDiff);
    GET_PARAM(maxQuadsToScan);
    GET_PARAM(phaseVarWeightR);
    GET_PARAM(phaseVarWeightZ);
    GET_PARAM(phaseVarWeightAngle);
    GET_PARAM(phaseVarWeightFreq);
    GET_PARAM(phaseVarWeightBias);
    GET_PARAM(markerWidthM);
    #undef GET_PARAM
    dynCfgSyncReq = true;
    phaseVariancePredictor.updateParams(params.phaseVarWeightR,
        params.phaseVarWeightZ, params.phaseVarWeightAngle,
        params.phaseVarWeightFreq, params.phaseVarWeightBias);

#ifdef CV_SHOW_IMAGES
    // Configure windows
    namedWindow("quad_1", CV_GUI_EXPANDED);
    namedWindow("quad_1_trimmed", CV_GUI_EXPANDED);
    namedWindow("segments", CV_GUI_EXPANDED);
    namedWindow("quads", CV_GUI_EXPANDED);
    namedWindow("tags", CV_GUI_EXPANDED);
#endif

    // Setup ROS communication links
    image_transport::ImageTransport it(local_nh);
    tagDetectionsPub = local_nh.advertise<ftag2::TagDetections>("detected_tags", 1);
    firstTagImagePub = it.advertise("first_tag_image", 1);
    processedImagePub = it.advertise("overlaid_image", 1);
    imageSub = it.subscribe("image_in", 1, &FTag2ReaderNodelet::imageCallback, this);
    cameraSub = it.subscribeCamera("camera_in", 1, &FTag2ReaderNodelet::cameraCallback, this);

    // Finish initialization
    alive = true;
    NODELET_INFO("FTag2 reader nodelet initialized");
  };


  void configCallback(ftag2::FTag2ReaderConfig& config, uint32_t level) {
    if (!alive) return;
    params = config;
    phaseVariancePredictor.updateParams(params.phaseVarWeightR,
        params.phaseVarWeightZ, params.phaseVarWeightAngle,
        params.phaseVarWeightFreq, params.phaseVarWeightBias);
  };


  void updateDyncfg() {
    if (dynCfgSyncReq) {
      if (dynCfgMutex.try_lock()) { // Make sure that dynamic reconfigure server or config callback is not active
        dynCfgMutex.unlock();
        dynCfgServer->updateConfig(params);
        NODELET_DEBUG_STREAM("Updated params");
        dynCfgSyncReq = false;
      }
    }
  };


  void imageCallback(const sensor_msgs::Image::ConstPtr& msg) {
    updateDyncfg();

    cv_bridge::CvImageConstPtr img = cv_bridge::toCvShare(msg);
    processImage(img->image, msg->header.seq);
  };


  void cameraCallback(const sensor_msgs::Image::ConstPtr& msg,
      const sensor_msgs::CameraInfo::ConstPtr& cam_info) {
    updateDyncfg();

    if (cam_info->D.size() == 5) {
      cameraDistortion = cv::Mat(cam_info->D);
      cameraDistortion.reshape(1, 1);
    }
    if (cam_info->K.size() == 9) {
      double* cameraIntrinsicPtr = (double*) cameraIntrinsic.data;
      for (int i = 0; i < 9; i++, cameraIntrinsicPtr++) *cameraIntrinsicPtr = cam_info->K[i];
    }
    cv_bridge::CvImageConstPtr img = cv_bridge::toCvShare(msg);
    processImage(img->image, msg->header.seq);
  };


  void processImage(const cv::Mat sourceImg, int ID) {
    // Update profiler
    rateP.try_toc();
    rateP.tic();
    durationP.tic();

    // Convert source image to grayscale
    cv::Mat grayImg;
    cv::cvtColor(sourceImg, grayImg, CV_RGB2GRAY);

    // 1. Detect line segments
    lineSegP.tic();
    std::vector<cv::Vec4i> segments = detectLineSegments(grayImg,
        params.sobelThreshHigh, params.sobelThreshLow, params.sobelBlurWidth,
        params.lineMinEdgelsCC, params.lineAngleMargin*degree,
        params.lineMinEdgelsSeg);
    lineSegP.toc();
#ifdef CV_SHOW_IMAGES
    {
      cv::Mat overlaidImg;
      cv::cvtColor(sourceImg, overlaidImg, CV_RGB2BGR);
      drawLineSegments(overlaidImg, segments);
      cv::imshow("segments", overlaidImg);
    }
#endif

    // 2. Detect quadrilaterals
    quadP.tic();
    std::list<Quad> quads = detectQuadsNew(segments,
        params.quadMinAngleIntercept*degree,
        params.quadMaxTIntDistRatio,
        params.quadMaxEndptDistRatio,
        params.quadMaxCornerGapEndptDistRatio,
        params.quadMaxEdgeGapDistRatio,
        params.quadMaxEdgeGapAlignAngle*degree,
        params.quadMinWidth);
    quads.sort(Quad::compareArea);
    quadP.toc();
#ifdef CV_SHOW_IMAGES
    {
      cv::Mat overlaidImg;
      cv::cvtColor(sourceImg, overlaidImg, CV_RGB2BGR);
      for (const Quad& quad: quads) {
        drawQuad(overlaidImg, quad.corners);
      }
      cv::imshow("quads", overlaidImg);
    }
#endif

    // 3. Decode tags from quads
    int quadCount = 0;
    cv::Mat quadImg;
    std::vector<FTag2Marker6S5F3B> tags;
    FTag2Marker6S5F3B currTag;
    for (const Quad& currQuad: quads) {
      // Check whether we have scanned enough quads
      quadCount++;
      if (quadCount > params.maxQuadsToScan) break;

      // Extract rectified quad image from frame
      quadExtractorP.tic();
      quadImg = extractQuadImg(sourceImg, currQuad, params.quadMinWidth*(8/6));
      quadExtractorP.toc();
      if (quadImg.empty()) { continue; }

      // Decode tag
      decoderP.tic();
      try {
        currTag = FTag2Decoder::decodeTag(quadImg, currQuad,
            params.markerWidthM,
            cameraIntrinsic, cameraDistortion,
            params.quadMaxStripAvgDiff,
            phaseVariancePredictor);
      } catch (const std::string& err) {
        continue;
      }
      decoderP.toc();

      // Store tag in list
      tags.push_back(currTag);
    } // Scan through all detected quads




    // TODO: 5 remove notification
    if (tags.size() > 0) {
      NODELET_INFO_STREAM(ID << ": " << tags.size() << " tags (quads: " << quads.size() << ")");
    } else if (quads.size() > 0) {
      NODELET_WARN_STREAM(ID << ": " << tags.size() << " tags (quads: " << quads.size() << ")");
    } else {
      NODELET_ERROR_STREAM(ID << ": " << tags.size() << " tags (quads: " << quads.size() << ")");
    }



    // Post-process largest detected tag
    if (tags.size() >= 1) {
      const FTag2Marker6S5F3B& firstTag = tags[0];

      // Show cropped tag image
#ifdef CV_SHOW_IMAGES
      {
        cv::imshow("quad_1_trimmed", firstTag.img);
      }
#endif

      // Publish cropped tag image
      cv_bridge::CvImage cvCroppedTagImgRot(std_msgs::Header(),
          sensor_msgs::image_encodings::MONO8, firstTag.img);
      cvCroppedTagImgRot.header.frame_id = boost::lexical_cast<std::string>(ID);
      firstTagImagePub.publish(cvCroppedTagImgRot.toImageMsg());
    }

    // Publish image overlaid with detected markers
    cv::Mat processedImg = sourceImg.clone();
    for (const FTag2Marker6S5F3B& tag: tags) {
      drawTag(processedImg, tag.corners);
    }
    cv_bridge::CvImage cvProcessedImg(std_msgs::Header(),
        sensor_msgs::image_encodings::RGB8, processedImg);
    cvProcessedImg.header.frame_id = boost::lexical_cast<std::string>(ID);
    processedImagePub.publish(cvProcessedImg.toImageMsg());
#ifdef CV_SHOW_IMAGES
    {
      cv::Mat overlaidImg;
      cv::cvtColor(processedImg, overlaidImg, CV_RGB2BGR);
      cv::imshow("tags", overlaidImg);
    }
#endif

    // Publish tag detections
    if (tags.size() > 0) {
      ftag2::TagDetections tagsMsg;
      tagsMsg.frameID = ID;

      for (const FTag2Marker6S5F3B& tag: tags) {
        ftag2::TagDetection tagMsg;
        tagMsg.pose.position.x = tag.position_x;
        tagMsg.pose.position.y = tag.position_y;
        tagMsg.pose.position.z = tag.position_z;
        tagMsg.pose.orientation.w = tag.orientation_w;
        tagMsg.pose.orientation.x = tag.orientation_x;
        tagMsg.pose.orientation.y = tag.orientation_y;
        tagMsg.pose.orientation.z = tag.orientation_z;
        tagMsg.markerPixelWidth = tag.rectifiedWidth;
        const double* magsPtr = (double*) tag.mags.data;
        tagMsg.mags = std::vector<double>(magsPtr, magsPtr + tag.mags.rows * tag.mags.cols);
        const double* phasesPtr = (double*) tag.phases.data;
        tagMsg.phases = std::vector<double>(phasesPtr, phasesPtr + tag.phases.rows * tag.phases.cols);
        tagsMsg.tags.push_back(tagMsg);
      }
      tagDetectionsPub.publish(tagsMsg);
    }

    // Update profiler
    durationP.toc();
    if (profilerDelaySec > 0) {
      ros::Time currTime = ros::Time::now();
      ros::Duration td = currTime - latestProfTime;
      if (td.toSec() > profilerDelaySec) {
        cout << "detectLineSegments: " << lineSegP.getStatsString() << endl;
        cout << "detectQuads: " << quadP.getStatsString() << endl;
        cout << "extractTags: " << quadExtractorP.getStatsString() << endl;
        cout << "decodeTag: " << decoderP.getStatsString() << endl;

        cout << "Pipeline Duration: " << durationP.getStatsString() << endl;
        cout << "Pipeline Rate: " << rateP.getStatsString() << endl;
        latestProfTime = currTime;
      }
    }

    // Allow OpenCV HighGUI events to process
#ifdef CV_SHOW_IMAGES
    char c = waitKey(1);
    if (c == 'x' || c == 'X') {
      ros::shutdown();
    }
#endif
  };
};


};


#include <pluginlib/class_list_macros.h>
PLUGINLIB_DECLARE_CLASS(ftag2, FTag2ReaderNodelet, ftag2::FTag2ReaderNodelet, nodelet::Nodelet)
