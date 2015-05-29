#include "detector/FTag2Detector.hpp"
#include "decoder/FTag2Decoder.hpp"

#include "common/Profiler.hpp"
#include "common/VectorAndCircularMath.hpp"

#include "tracker/FTag2Tracker.hpp"

#include <ros/ros.h>
#include <dynamic_reconfigure/server.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <nodelet/nodelet.h>
#include "ftag2/FTag2ReaderConfig.h"
#include "ftag2_core/TagDetections.h"
#include "ftag2_core/TagDetection.h"

#include "ftag2_core/ARMarkerFT.h"
#include "ftag2_core/ARMarkersFT.h"
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_broadcaster.h>

using namespace std;
using namespace cv;
using namespace vc_math;

typedef dynamic_reconfigure::Server<ftag2::FTag2ReaderConfig> ReconfigureServer;

#define CV_SHOW_IMAGES
#define ROS_PUBLISHING_DETECTIONS
#undef DISPLAY_DECODED_TAG_PAYLOADS

#define DISPLAY_HYPOTHESIS_MIN_NUM_DECODED_PHASES (0)
#define DISPLAY_HYPOTHESIS_MIN_NUM_DECODED_SECTIONS (0)

namespace ftag2 {


class FTag2TrackerNodelet : public nodelet::Nodelet {
protected:
  bool alive;

  int tagType;

  ReconfigureServer* dynCfgServer;
  boost::recursive_mutex dynCfgMutex;
  bool dynCfgSyncReq;

  image_transport::Subscriber imageSub;
  image_transport::CameraSubscriber cameraSub;
  ros::Publisher tagDetectionsPub;
  ros::Publisher firstTagDetectionPub;
  image_transport::Publisher processedImagePub;
  image_transport::Publisher firstTagImagePub;

  ros::Publisher rvizMarkersPub_;
  ros::Publisher arMarkerPub_;

  ftag2::FTag2ReaderConfig params;

  cv::Mat cameraIntrinsic, cameraDistortion;

  PhaseVariancePredictor phaseVariancePredictor;

  ros::Timer idleSpinTimer;

  // DEBUG VARIABLES
  Profiler lineSegP, quadP, quadExtractorP, decoderP, durationP, rateP;
  ros::Time latestProfTime;
  double profilerDelaySec;

  FTag2Tracker FT;



public:
  FTag2TrackerNodelet() : nodelet::Nodelet(),
      alive(false),
      tagType(FTag2Payload::FTAG2_6S2F22B),
      dynCfgServer(NULL),
      dynCfgSyncReq(false),
      latestProfTime(ros::Time::now()),
      profilerDelaySec(0) {
    params.quadFastDetector = false;
    params.quadRefineCorners = true;
    params.quadMaxScans = 30;
    params.tagMaxStripAvgDiff = 15.0;
    params.tagBorderMeanMaxThresh = 80.0;
    params.tagBorderStdMaxThresh = 40.0;
    params.tagMagFilGainNeg = 0.6;
    params.tagMagFilGainPos = 0.6;
    params.tagMagFilPowNeg = 1.0;
    params.tagMagFilPowPos = 1.0;
    params.phaseVarWeightR = 0;
    params.phaseVarWeightZ = 0;
    params.phaseVarWeightAngle = 0;
    params.phaseVarWeightFreq = 0;
    params.phaseVarWeightBias = 10*10;
    params.numSamplesPerRow = 1;
    params.markerWidthM = 0.055;
    params.tempTagDecodeStd = 3;
    phaseVariancePredictor.updateParams(params.phaseVarWeightR,
        params.phaseVarWeightZ, params.phaseVarWeightAngle,
        params.phaseVarWeightFreq, params.phaseVarWeightBias);
  };


  ~FTag2TrackerNodelet() {
    alive = false;
  };


  virtual void onInit() {
    // Obtain node handles
    //ros::NodeHandle& nh = getNodeHandle();
    ros::NodeHandle& local_nh = getPrivateNodeHandle();

    // Obtain image transport parameter
    std::string transportType = "raw";
    local_nh.param("transport_type", transportType, transportType);    

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

    // Obtain expected tag type to decode
    local_nh.param("tag_type", tagType, tagType);

    // Setup and initialize dynamic reconfigure server
    dynCfgServer = new ReconfigureServer(dynCfgMutex, local_nh);
    dynCfgServer->setCallback(bind(&FTag2TrackerNodelet::configCallback, this, _1, _2));

    // Parse static parameters and update dynamic reconfigure values
    #define GET_PARAM(v) \
      local_nh.param(std::string(#v), params.v, params.v)
    GET_PARAM(quadFastDetector);
    GET_PARAM(quadRefineCorners);
    GET_PARAM(quadMaxScans);
    GET_PARAM(tagMaxStripAvgDiff);
    GET_PARAM(tagBorderMeanMaxThresh);
    GET_PARAM(tagBorderStdMaxThresh);
    GET_PARAM(tagMagFilGainNeg);
    GET_PARAM(tagMagFilGainPos);
    GET_PARAM(tagMagFilPowNeg);
    GET_PARAM(tagMagFilPowPos);
    GET_PARAM(phaseVarWeightR);
    GET_PARAM(phaseVarWeightZ);
    GET_PARAM(phaseVarWeightAngle);
    GET_PARAM(phaseVarWeightFreq);
    GET_PARAM(phaseVarWeightBias);
    GET_PARAM(numSamplesPerRow);
    GET_PARAM(markerWidthM);
    GET_PARAM(tempTagDecodeStd);
    #undef GET_PARAM
    dynCfgSyncReq = true;
    phaseVariancePredictor.updateParams(params.phaseVarWeightR,
        params.phaseVarWeightZ, params.phaseVarWeightAngle,
        params.phaseVarWeightFreq, params.phaseVarWeightBias);

#ifdef CV_SHOW_IMAGES
    // Configure windows
    //namedWindow("quads", CV_GUI_EXPANDED);
    namedWindow("tags", CV_GUI_EXPANDED);
#endif

    // Resolve image topic names
    std::string imageTopic = local_nh.resolveName("image_in");
    std::string cameraTopic = local_nh.resolveName("camera_in");

    // Setup ROS communication links
    image_transport::ImageTransport it(local_nh);
    tagDetectionsPub = local_nh.advertise<ftag2_core::TagDetections>("detected_tags", 1);
    firstTagDetectionPub = local_nh.advertise<ftag2_core::TagDetection>("first_tag", 1);
    firstTagImagePub = it.advertise("first_tag_image", 1);
    processedImagePub = it.advertise("overlaid_image", 1);
    imageSub = it.subscribe(imageTopic, 1, &FTag2TrackerNodelet::imageCallback, this, transportType);
    cameraSub = it.subscribeCamera(cameraTopic, 1, &FTag2TrackerNodelet::cameraCallback, this, transportType);

    rvizMarkersPub_ = local_nh.advertise < visualization_msgs::MarkerArray > ("ftag2_vis_Marker", 1);
    arMarkerPub_ = local_nh.advertise < ftag2_core::ARMarkersFT > ("ft_pose_markers", 1);

    // Finish initialization
    alive = true;
    NODELET_INFO("FTag2 tracker nodelet initialized");
    idleSpinTimer = local_nh.createTimer(ros::Duration(0.5), &FTag2TrackerNodelet::handleIdleSpinOnce, this);
    idleSpinTimer.start();
  };


  void handleIdleSpinOnce(const ros::TimerEvent& event) {
#ifdef CV_SHOW_IMAGES
    char c = waitKey(1);
    if (c == 'x' || c == 'X') {
      ros::shutdown();
    }
#endif
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
    try {
      processImage(img->image, msg->header.seq);
    } catch (const std::string& err) {
      ROS_ERROR_STREAM("Nodelet shutting down: " << err);
      ros::shutdown();
    } catch (char const* err) {
      ROS_ERROR_STREAM("Nodelet shutting down: (caught char*!) " << err);
      ros::shutdown();
    }
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
    try {
      processImage(img->image, msg->header.seq);
    } catch (const std::string& err) {
      ROS_ERROR_STREAM("Nodelet shutting down: " << err);
      ros::shutdown();
    } catch (char const* err) {
      ROS_ERROR_STREAM("Nodelet shutting down: (caught char*!) " << err);
      ros::shutdown();
    }
  };


  void processImage(const cv::Mat sourceImg, int ID) {
    // Update profiler
    rateP.try_toc();
    rateP.tic();
    durationP.tic();

    // 1. Convert source image to grayscale
    cv::Mat grayImg;
    cv::cvtColor(sourceImg, grayImg, CV_RGB2GRAY);

    // 2. Detect quadrilaterals in image
    quadP.tic();
    std::list<Quad> quads;
    if (params.quadFastDetector) {
      quads = detectQuadsViaContour(grayImg); // TODO: 9 consider using downsized image to speed up search of quads
    } else {
      std::vector<cv::Vec4i> segments = detectLineSegments(grayImg);
      quads = scanQuadsSpatialHash(segments, grayImg.cols, grayImg.rows);
    }
    if (params.quadRefineCorners) {
      refineQuadCorners(grayImg, quads);
    }
    quads.sort(Quad::greaterArea);
    quads.erase(std::unique(quads.begin(), quads.end()), quads.end()); // Remove duplicates
    quadP.toc();

    // TODO: 0 Delete after gathering data for training variance model and mags.
    cv::Mat first_quad_img;
    bool first_tag = true;
    /////////////////////////////////////////////////////////////////////////

    // 3. Decode FTag2 markers
    int quadCount = 0;
    cv::Mat quadImg;
    std::vector<FTag2Marker> tags;
    FTag2Marker currTag(tagType);
    for (const Quad& currQuad: quads) {
      // Reject quads that overlap with already-detected tags (which have larger area than current quad)
      bool overlap = false;
      for (const FTag2Marker& prevTag: tags) {
        if (vc_math::checkPolygonOverlap(prevTag.tagCorners, currQuad.corners)) {
          overlap = true;
          break;
        }
      }
      if (overlap) continue;

      // Check whether we have scanned enough quads
      quadCount++;
      if (quadCount > params.quadMaxScans) break; // TODO: 1 measure how much impact does this limiting have (i.e. if we simply tried to decode all quads); suspect little additional computing + no false positive (when using magn filter)

      // Extract rectified quad image from frame
      quadExtractorP.tic();
      quadImg = extractQuadImg(sourceImg, currQuad);
      quadExtractorP.toc();
      if (quadImg.empty()) { continue; }

      // Decode tag
      decoderP.tic();
      try {
        currTag = decodeQuad(quadImg, currQuad,
            tagType,
            params.markerWidthM,
            params.numSamplesPerRow,
            cameraIntrinsic, cameraDistortion,
            params.tagMaxStripAvgDiff,
            params.tagBorderMeanMaxThresh, params.tagBorderStdMaxThresh,
            params.tagMagFilGainNeg,
            params.tagMagFilGainPos,
            params.tagMagFilPowNeg,
            params.tagMagFilPowPos,
            phaseVariancePredictor);
        decodePayload(currTag.payload, params.tempTagDecodeStd);
        
        if ( first_tag )
        {
          first_quad_img = quadImg.clone();
          first_tag = false;

          ftag2_core::TagDetection tag_msg;

          tag_msg.pose.position.x = currTag.pose.position_x;
          tag_msg.pose.position.y = currTag.pose.position_y;
          tag_msg.pose.position.z = currTag.pose.position_z;
          tag_msg.pose.orientation.w = currTag.pose.orientation_w;
          tag_msg.pose.orientation.x = currTag.pose.orientation_x;
          tag_msg.pose.orientation.y = currTag.pose.orientation_y;
          tag_msg.pose.orientation.z = currTag.pose.orientation_z;
          tag_msg.markerWidthPx = currTag.tagWidth;
          tag_msg.markerRot90 = currTag.tagImgCCRotDeg/90;
          for (const cv::Point2f& p: currTag.tagCorners) {
          	tag_msg.markerCornersPx.push_back(p.x);
          	tag_msg.markerCornersPx.push_back(p.y);
          }
          tag_msg.markerWidthM = params.markerWidthM;

          const double* magsPtr = (double*) currTag.payload.mags.data;
          tag_msg.mags = std::vector<double>(magsPtr, magsPtr + currTag.payload.mags.rows * currTag.payload.mags.cols);
          const double* phasesPtr = (double*) currTag.payload.phases.data;
          tag_msg.phases = std::vector<double>(phasesPtr, phasesPtr + currTag.payload.phases.rows * currTag.payload.phases.cols);
          tag_msg.bitChunksStr = currTag.payload.bitChunksStr;
          tag_msg.decodedPayloadStr = currTag.payload.decodedPayloadStr;

          firstTagDetectionPub.publish(tag_msg);
        }
        /////////////////////////////////////////////////////////////////////////
      } catch (const std::string& err) {
        // TODO: 9 remove debug code once API stabilized
        /*
        const std::vector<cv::Point2f>& corners = currQuad.corners; // assumed stored in clockwise order (in image space)
        double lenA = vc_math::dist(corners[0], corners[1]);
        double lenB = vc_math::dist(corners[1], corners[2]);
        double lenC = vc_math::dist(corners[2], corners[3]);
        double lenD = vc_math::dist(corners[3], corners[0]);
        double angleAD = std::acos(vc_math::dot(corners[1], corners[0], corners[0], corners[3])/lenA/lenD);
        double angleBC = std::acos(vc_math::dot(corners[1], corners[2], corners[2], corners[3])/lenB/lenC);
        ROS_WARN_STREAM(err);
        ROS_WARN_STREAM("corners: " <<
            "(" << corners[0].x << ", " << corners[0].y << ") - " <<
            "(" << corners[1].x << ", " << corners[1].y << ") - " <<
            "(" << corners[2].x << ", " << corners[2].y << ") - " <<
            "(" << corners[3].x << ", " << corners[3].y << ")");
        ROS_WARN_STREAM("lengths: " << lenA << ", " << lenB << ", " << lenC << ", " << lenD);
        ROS_WARN_STREAM("angles: " << angleAD << ", " << angleBC << std::endl);

        {
          cv::imshow("debug", quadImg);
          cv::imwrite("/tmp/quadImg.png", quadImg);
          waitKey();
        }*/

        continue;
      }
      decoderP.toc();

      // Store tag in list
      tags.push_back(currTag);
    } // Scan through all detected quads
    FT.step( tags , params.markerWidthM, cameraIntrinsic, cameraDistortion );

#ifdef CV_SHOW_IMAGES
    {
      cv::Mat overlaidImg;
      cv::cvtColor(sourceImg, overlaidImg, CV_RGB2BGR);
      for (const Quad& quad: quads) {
        drawQuad(overlaidImg, quad.corners);
      }
      for (const FTag2Marker& tag: tags) {
        drawQuadWithCorner(overlaidImg, tag.tagCorners);
        drawMarkerLabel(overlaidImg, tag.tagCorners, tag.payload.bitChunksStr, 0.8);
      }
      cv::imshow("tags", overlaidImg);
    }
#endif

#ifdef ROS_PUBLISHING_DETECTIONS
    if ( first_tag == false )
    {
		cv_bridge::CvImage cvCroppedTagImgRot(std_msgs::Header(),
			sensor_msgs::image_encodings::MONO8, first_quad_img);
		cvCroppedTagImgRot.header.frame_id = boost::lexical_cast<std::string>(ID);
		firstTagImagePub.publish(cvCroppedTagImgRot.toImageMsg());
    }

    ftag2_core::TagDetections tags_msg;
    tags_msg.frameID = ID;
    for (const FTag2Marker& tag: tags) {
		ftag2_core::TagDetection tag_msg;

		tag_msg.pose.position.x = tag.pose.position_x;
		tag_msg.pose.position.y = tag.pose.position_y;
		tag_msg.pose.position.z = tag.pose.position_z;
		tag_msg.pose.orientation.w = tag.pose.orientation_w;
		tag_msg.pose.orientation.x = tag.pose.orientation_x;
		tag_msg.pose.orientation.y = tag.pose.orientation_y;
		tag_msg.pose.orientation.z = tag.pose.orientation_z;
		tag_msg.markerWidthPx = tag.tagWidth;
		tag_msg.markerRot90 = tag.tagImgCCRotDeg/90;
		for (const cv::Point2f& p: tag.tagCorners) {
			tag_msg.markerCornersPx.push_back(p.x);
			tag_msg.markerCornersPx.push_back(p.y);
		}
		tag_msg.markerWidthM = params.markerWidthM;

		const double* magsPtr = (double*) tag.payload.mags.data;
		tag_msg.mags = std::vector<double>(magsPtr, magsPtr + tag.payload.mags.rows * tag.payload.mags.cols);
		const double* phasesPtr = (double*) tag.payload.phases.data;
		tag_msg.phases = std::vector<double>(phasesPtr, phasesPtr + tag.payload.phases.rows * tag.payload.phases.cols);
		tag_msg.bitChunksStr = tag.payload.bitChunksStr;
		tag_msg.decodedPayloadStr = tag.payload.decodedPayloadStr;

		tags_msg.tags.push_back(tag_msg);
    }
    tagDetectionsPub.publish(tags_msg);
#endif

    cv::Mat overlaidImg;
    cv::cvtColor(sourceImg, overlaidImg, CV_RGB2BGR);

    // Draw matched tag hypotheses: blue border, cyan-on-blue text
#ifdef DISPLAY_DECODED_TAG_PAYLOADS
    for (const MarkerFilter& trackedTag: FT.filters) {
//      if (!trackedTag.active) continue;
      const FTag2Payload& trackedPayload = trackedTag.hypothesis.payload;
      if (trackedPayload.numDecodedSections >= DISPLAY_HYPOTHESIS_MIN_NUM_DECODED_SECTIONS) {
        drawQuadWithCorner(overlaidImg, trackedTag.hypothesis.back_proj_corners,
            CV_RGB(255, 0, 0), CV_RGB(0, 255, 255),
            CV_RGB(0, 255, 255), CV_RGB(255, 0, 0));
      }
    }
    for (const MarkerFilter& trackedTag: FT.filters) {
//      if (!trackedTag.active) continue;
      const FTag2Payload& trackedPayload = trackedTag.hypothesis.payload;
      if (trackedPayload.numDecodedSections >= DISPLAY_HYPOTHESIS_MIN_NUM_DECODED_SECTIONS) {
        drawMarkerLabel(overlaidImg, trackedTag.hypothesis.back_proj_corners,
            trackedPayload.decodedPayloadStr, 0.8,
            cv::FONT_HERSHEY_SIMPLEX, 1, 0.4,
            CV_RGB(0, 255, 255), CV_RGB(0, 0, 255));
      }
    }
#else
    for (const MarkerFilter& trackedTag: FT.filters) {
//      if (!trackedTag.active) continue;
      const FTag2Payload& trackedPayload = trackedTag.hypothesis.payload;
      if (trackedPayload.numDecodedPhases >= DISPLAY_HYPOTHESIS_MIN_NUM_DECODED_PHASES) {
        drawQuadWithCorner(overlaidImg, trackedTag.hypothesis.back_proj_corners,
            CV_RGB(255, 0, 0), CV_RGB(0, 255, 255),
            CV_RGB(0, 255, 255), CV_RGB(255, 0, 0));
      }
    }
    for (const MarkerFilter& trackedTag: FT.filters) {
//      if (!trackedTag.active) continue;
      const FTag2Payload& trackedPayload = trackedTag.hypothesis.payload;
      if (trackedPayload.numDecodedPhases >= DISPLAY_HYPOTHESIS_MIN_NUM_DECODED_PHASES) {
        drawMarkerLabel(overlaidImg, trackedTag.hypothesis.back_proj_corners,
            trackedPayload.bitChunksStr, 0.8,
            cv::FONT_HERSHEY_SIMPLEX, 1, 0.4,
            CV_RGB(0, 255, 255), CV_RGB(0, 0, 255));
      }
    }
#endif
    cv_bridge::CvImage cvProcImg(std_msgs::Header(), sensor_msgs::image_encodings::BGR8, overlaidImg);
    cvProcImg.header.frame_id = boost::lexical_cast<std::string>(ID);
    processedImagePub.publish(cvProcImg.toImageMsg());


// Draw detected tag observations: red
//    for (const FTag2Marker& tagObs: tags) {
//      drawQuadWithCorner(overlaidImg, tagObs.tagCorners);
//    }

    cv::imshow("hypotheses", overlaidImg);

    // -. Update profiler
    durationP.toc();
    if (profilerDelaySec > 0) {
    	ros::Time currTime = ros::Time::now();
    	ros::Duration td = currTime - latestProfTime;
    	if (td.toSec() > profilerDelaySec) {
    		ROS_WARN_STREAM("===== PROFILERS =====");
    		ROS_WARN_STREAM("detectQuads: " << quadP.getStatsString());
    		ROS_WARN_STREAM("Pipeline Duration:: " << durationP.getStatsString());
    		ROS_WARN_STREAM("Pipeline Rate: " << rateP.getStatsString());
    		latestProfTime = currTime;
    	}
    }

    // -. Allow OpenCV HighGUI events to process
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
PLUGINLIB_DECLARE_CLASS(ftag2, FTag2TrackerNodelet, ftag2::FTag2TrackerNodelet, nodelet::Nodelet)
