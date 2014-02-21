#include "detector/FTag2Detector.hpp"
#include "decoder/FTag2Decoder.hpp"

#include "common/Profiler.hpp"
#include "common/VectorAndCircularMath.hpp"

#include <ros/ros.h>
#include <dynamic_reconfigure/server.h>
#include <ftag2/FTag2ReaderConfig.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <nodelet/nodelet.h>


using namespace std;
using namespace cv;
using namespace vc_math;


typedef dynamic_reconfigure::Server<ftag2::FTag2ReaderConfig> ReconfigureServer;


#define CV_SHOW_IMAGES


namespace ftag2 {


class FTag2ReaderNodelet : public nodelet::Nodelet {
protected:
  bool alive;

  ros::NodeHandle nh;
  ros::NodeHandle local_nh;
  image_transport::ImageTransport it;

  ReconfigureServer* dynCfgServer;
  boost::recursive_mutex dynCfgMutex;
  bool dynCfgSyncReq;

  image_transport::Subscriber imageSub;
  image_transport::CameraSubscriber cameraSub;
  ros::Publisher tagDetectionsPub;
  image_transport::Publisher firstTagImagePub;

  ftag2::FTag2ReaderConfig params;

  cv::Mat cameraIntrinsic, cameraDistortion;

  // DEBUG VARIABLES
  Profiler lineSegP, quadP, quadExtractorP, decoderP, durationP, rateP;
  ros::Time latestProfTime;


public:
  FTag2ReaderNodelet() : nodelet::Nodelet(),
      alive(false),
      nh(),
      local_nh("~"),
      it(nh),
      dynCfgServer(NULL),
      dynCfgSyncReq(false),
      latestProfTime(ros::Time::now()) {
    // Set default parameter values
    params.sobelThreshHigh = 100;
    params.sobelThreshLow = 30;
    params.sobelBlurWidth = 3;
    params.lineAngleMargin = 20.0; // *degree
    params.lineMinEdgelsCC = 50;
    params.lineMinEdgelsSeg = 15;
    params.quadMinWidth = 15.0;
    params.quadMinAngleIntercept = 30.0;
    params.quadMinEndptDist = 4.0;
    params.quadMaxStripAvgDiff = 15.0;
    params.maxQuadsToScan = 10;
    params.markerWidthM = 0.07;
  };


  ~FTag2ReaderNodelet() {
    alive = false;
  };


  virtual void onInit() {
    // Load static camera calibration information
    std::string cameraIntrinsicStr, cameraDistortionStr;
    local_nh.param("camera_intrinsic", cameraIntrinsicStr, cameraIntrinsicStr);
    local_nh.param("camera_distortion", cameraDistortionStr, cameraDistortionStr);
    if (cameraIntrinsicStr.size() > 0) {
      cameraIntrinsic = str2mat(cameraIntrinsicStr, 3);
    } else {
      cameraIntrinsic = cv::Mat::zeros(3, 3, CV_64FC1);
    }
    if (cameraDistortionStr.size() > 0) {
      cameraDistortion = str2mat(cameraDistortionStr, 1);
    } else {
      cameraDistortion = cv::Mat::zeros(1, 5, CV_64FC1);
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
    GET_PARAM(quadMinEndptDist);
    GET_PARAM(quadMaxStripAvgDiff);
    GET_PARAM(maxQuadsToScan);
    GET_PARAM(markerWidthM);
    #undef GET_PARAM
    dynCfgSyncReq = true;

#ifdef CV_SHOW_IMAGES
    // Configure windows
    namedWindow("edgels", CV_GUI_EXPANDED);
    namedWindow("segments", CV_GUI_EXPANDED);
    namedWindow("quad_1", CV_GUI_EXPANDED);
    namedWindow("quad_1_trimmed", CV_GUI_EXPANDED);
    namedWindow("quads", CV_GUI_EXPANDED);
#endif

    // Setup ROS communication links
    imageSub = it.subscribe("image", 1, &FTag2ReaderNodelet::imageCallback, this);
    cameraSub = it.subscribeCamera("camera", 1, &FTag2ReaderNodelet::cameraCallback, this);
    //tagDetectionsPub = nh.advertise<ftag2::TagDetections>("detected_tags", 1);
    firstTagImagePub = it.advertise("first_tag_image", 1);

    // Finish initialization
    alive = true;
    NODELET_INFO("FTag2 reader nodelet initialized");
  };


  void configCallback(ftag2::FTag2ReaderConfig& config, uint32_t level) {
    if (!alive) return;
    params = config;
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

    // Extract line segments using optimized segment detector using
    // angle-bounded connected edgel components
    lineSegP.tic();
    std::vector<cv::Vec4i> segments = detectLineSegments(grayImg,
        params.sobelThreshHigh, params.sobelThreshLow, params.sobelBlurWidth,
        params.lineMinEdgelsCC, params.lineAngleMargin*degree,
        params.lineMinEdgelsSeg);
    lineSegP.toc();

#ifdef CV_SHOW_IMAGES
    {
      cv::Mat overlaidImg = sourceImg.clone();
      drawLineSegments(overlaidImg, segments);
      cv::imshow("segments", overlaidImg);
    }
#endif

    // Detect quads
    quadP.tic();
    std::list<Quad> quads = detectQuads(segments,
        params.quadMinAngleIntercept*degree,
        params.quadMinEndptDist);
    quads.sort(Quad::compareArea);
    quadP.toc();
    if (quads.empty()) return;

    // TODO: 0 remove after debugging flickering bug
#ifdef REMOVE_AFTER_DEBUG
    if (quads.empty()) {
      ROS_WARN_STREAM("NO QUADS IN FRAME");
    }
    if (false) {
      cout << "Quads: " << quads.size() << endl;
      for (const Quad& q: quads) {
        cout << "- " << q.area << endl;
      }
    }
#endif

    // Scan through all detected quads
    bool foundTag = false;
    int quadCount = 0;
    cv::Mat tagImg, trimmedTagImg, croppedTagImg, croppedTagImgRot;
    FTag2Marker6S5F3B currTag;
    for (const Quad& currQuad: quads) {
      // Check whether we have scanned enough quads
      quadCount++;
      if (quadCount > params.maxQuadsToScan) break;

      // Extract, rectify, and crop tag payload image, corresponding to quad
      quadExtractorP.tic();
      tagImg = extractQuadImg(sourceImg, currQuad, params.quadMinWidth);
      if (tagImg.empty()) { continue; }
      trimmedTagImg = trimFTag2Quad(tagImg, params.quadMaxStripAvgDiff);
      croppedTagImg = cropFTag2Border(trimmedTagImg);
      if (croppedTagImg.rows < params.quadMinWidth ||
          croppedTagImg.cols < params.quadMinWidth) { continue; }
      quadExtractorP.toc();

      // Decode tag
      decoderP.tic();
      currTag = FTag2Marker6S5F3B(croppedTagImg);
      decoderP.toc();
      if (!currTag.hasSignature) { continue; }

      // Compute pose of tag
      std::vector<cv::Point2f> tagCorners;
      switch ((currTag.imgRotDir/90) % 4) {
      case 1:
        tagCorners.push_back(currQuad.corners[1]);
        tagCorners.push_back(currQuad.corners[2]);
        tagCorners.push_back(currQuad.corners[3]);
        tagCorners.push_back(currQuad.corners[0]);
        break;
      case 2:
        tagCorners.push_back(currQuad.corners[2]);
        tagCorners.push_back(currQuad.corners[3]);
        tagCorners.push_back(currQuad.corners[0]);
        tagCorners.push_back(currQuad.corners[1]);
        break;
      case 3:
        tagCorners.push_back(currQuad.corners[3]);
        tagCorners.push_back(currQuad.corners[0]);
        tagCorners.push_back(currQuad.corners[1]);
        tagCorners.push_back(currQuad.corners[2]);
        break;
      default:
        tagCorners = currQuad.corners;
        break;
      }
      solvePose(tagCorners, params.markerWidthM,
          cameraIntrinsic, cameraDistortion,
          currTag.position_x, currTag.position_y, currTag.position_z,
          currTag.orientation_w, currTag.orientation_x, currTag.orientation_y,
          currTag.orientation_z);

      // Show quad, tag, and cropped tag images
      BaseCV::rotate90(croppedTagImg, croppedTagImgRot, currTag.imgRotDir/90);
#ifdef CV_SHOW_IMAGES
      {
        cv::Mat quadsImg = sourceImg.clone();
        drawQuad(quadsImg, currQuad);
        cv::imshow("quads", quadsImg);
        cv::imshow("quad_1", tagImg);
        cv::imshow("quad_1_trimmed", croppedTagImgRot);
      }
#endif

      // Publish cropped tag image
      cv_bridge::CvImage cvCroppedTagImgRot(std_msgs::Header(),
          sensor_msgs::image_encodings::MONO8, croppedTagImgRot);
      cvCroppedTagImgRot.header.frame_id = boost::lexical_cast<std::string>(ID);
      firstTagImagePub.publish(cvCroppedTagImgRot.toImageMsg());

      // TODO: 000 publish marker info
      /*
                      ftag2::FreqTBMarkerInfo markerInfoMsg;
                      markerInfoMsg.frameID = frameID;
                      markerInfoMsg.pose.position.x = tag.position_x;
                      markerInfoMsg.pose.position.y = tag.position_y;
                      markerInfoMsg.pose.position.z = tag.position_z;
                      markerInfoMsg.pose.orientation.w = tag.orientation_w;
                      markerInfoMsg.pose.orientation.x = tag.orientation_x;
                      markerInfoMsg.pose.orientation.y = tag.orientation_y;
                      markerInfoMsg.pose.orientation.z = tag.orientation_z;
                      const double* magsPtr = (double*) tag.mags.data;
                      markerInfoMsg.mags = std::vector<double>(magsPtr, magsPtr + tag.mags.rows * tag.mags.cols);
                      const double* phasesPtr = (double*) tag.phases.data;
                      markerInfoMsg.phases = std::vector<double>(phasesPtr, phasesPtr + tag.phases.rows * tag.phases.cols);
                      markerInfoMsg.hasSignature = tag.hasSignature;
                      markerInfoMsg.hasValidXORs = tag.hasValidXORs;
                      markerInfoMsg.hasValidCRC = tag.hasValidCRC;
                      markerInfoMsg.payloadOct = tag.payloadOct;
                      markerInfoMsg.xorBin = tag.xorBin;
                      markerInfoMsg.signature = tag.signature;
                      markerInfoMsg.CRC12Expected = tag.CRC12Expected;
                      markerInfoMsg.CRC12Decoded = tag.CRC12Decoded;
                      markerInfoPub.publish(markerInfoMsg);
      */

      // Display tag info
      std::ostringstream oss;
      if (currTag.hasValidXORs && currTag.hasValidCRC) {
        oss << "=> RECOG  : ";
      } else if (currTag.hasValidXORs) {
        oss << "x> BAD CRC: ";
      } else {
        oss << "x> BAD XOR: ";
      }
      oss << currTag.payloadOct << "; XOR: " << currTag.xorBin << "; Rot=" << currTag.imgRotDir << "'";
      if (currTag.hasValidXORs && currTag.hasValidCRC) {
        oss << "\tID: " << currTag.payload.to_ullong();
      }
      NODELET_INFO_STREAM(oss.str());

      // For now, terminate after finding 1 tag
      foundTag = true;
      break;
    } // Scan through all detected quads
    if (!foundTag) {
      cv::imshow("quads", sourceImg);
    }

    // Update profiler
    durationP.toc();
    // TODO: 000 deal with profiler verbosity (have profiler spit time as rosparam)
#ifdef TODO
    ros::Time currTime = ros::Time::now();
    ros::Duration td = currTime - latestProfTime;
    if (td.toSec() > 1.0) {

      cout << "detectLineSegments: " << lineSegP.getStatsString() << endl;
      cout << "detectQuads: " << quadP.getStatsString() << endl;
      cout << "extractTags: " << quadExtractorP.getStatsString() << endl;
      cout << "decodeTag: " << decoderP.getStatsString() << endl;

      cout << "Pipeline Duration: " << durationP.getStatsString() << endl;
      cout << "Pipeline Rate: " << rateP.getStatsString() << endl;
      latestProfTime = currTime;
    }
#endif
  };
};


};


#include <pluginlib/class_list_macros.h>
PLUGINLIB_DECLARE_CLASS(ftag2, FTag2ReaderNodelet, ftag2::FTag2ReaderNodelet, nodelet::Nodelet)
