#include "detector/FTag2Detector.hpp"
#include "decoder/FTag2Decoder.hpp"

#include "common/Profiler.hpp"
#include "common/VectorAndCircularMath.hpp"
#include "common/FTag2.hpp"

#include "tracker/FTag2Tracker.hpp"
#include "tracker/MarkerFilter.hpp"

#include <ros/ros.h>
#include <dynamic_reconfigure/server.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <nodelet/nodelet.h>
#include <ftag2/CamTestbenchConfig.h>
#include "ftag2/TagDetections.h"

#include "tracker/ParticleFilter.hpp"
#include "std_msgs/Float64MultiArray.h"

#include <ftag2/FreqTBMarkerInfo.h>

#include <visualization_msgs/Marker.h>
using namespace std;
using namespace cv;
using namespace vc_math;

typedef dynamic_reconfigure::Server<ftag2::CamTestbenchConfig> ReconfigureServer;


#define CV_SHOW_IMAGES
#undef DISPLAY_DECODED_TAG_PAYLOADS
#undef PROFILER

#undef PARTICLE_FILTER

#define DECODE_PAYLOAD_N_STD_THRESH (3)
#define DISPLAY_HYPOTHESIS_MIN_NUM_DECODED_PHASES (8)
#define DISPLAY_HYPOTHESIS_MIN_NUM_DECODED_SECTIONS (2)



namespace ftag2 {


class FTag2TrackerNodelet : public nodelet::Nodelet {
protected:

  FTag2Tracker FT;

  bool alive;

  ReconfigureServer* dynCfgServer;
  boost::recursive_mutex dynCfgMutex;
  bool dynCfgSyncReq;

  image_transport::Subscriber imageSub;
  image_transport::CameraSubscriber cameraSub;
  ros::Publisher tagDetectionsPub;
  image_transport::Publisher processedImagePub;
  image_transport::Publisher firstTagImagePub;

  ros::Publisher markerInfoPub;
  ros::Publisher vis_pub;
  visualization_msgs::Marker marker;
  int frameID;

  ftag2::CamTestbenchConfig params;

  cv::Mat cameraIntrinsic, cameraDistortion;

  PhaseVariancePredictor phaseVariancePredictor;

#ifdef PARTICLE_FILTER
  std::vector<FTag2Pose> tag_observations;
  bool tracking;
  ParticleFilter PF;
  ParticleFilter::time_point last_frame_time;
  ParticleFilter::time_point starting_time;
  ros::Publisher pubTrack;
#endif


  // DEBUG VARIABLES
  Profiler lineSegP, quadP, quadExtractorP, decodeQuadP, trackerP, decodePayloadP, durationP, rateP;
  ros::Time latestProfTime;
  double profilerDelaySec;

public:
  FTag2TrackerNodelet() : nodelet::Nodelet(),
  alive(false),
  dynCfgServer(NULL),
  dynCfgSyncReq(false),
  frameID(0),
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
    params.quadMaxScans = 10;
    params.tagMaxStripAvgDiff = 15.0;
    params.tagBorderMeanMaxThresh = 150.0;
    params.tagBorderStdMaxThresh = 30.0;
    params.phaseVarWeightR = 0;
    params.phaseVarWeightZ = 0;
    params.phaseVarWeightAngle = 0;
    params.phaseVarWeightFreq = 0;
    params.phaseVarWeightBias = 10*10;
    params.markerWidthM = 0.07;
    params.numberOfParticles = 1000;
    params.position_std = 0.1;
    params.orientation_std = 0.1;
    params.position_noise_std = 0.2;
    params.orientation_noise_std = 0.2;
    params.velocity_noise_std = 0.05;
    params.acceleration_noise_std = 0.01;
    params.run_id = 1;
    params.within_phase_range_n_sigma = 10.0;
    params.within_phase_range_allowed_missmatches = 10;
    params.within_phase_range_threshold = 200;
    FTag2Payload::updateParameters(params.within_phase_range_n_sigma, params.within_phase_range_allowed_missmatches, params.within_phase_range_threshold);
    phaseVariancePredictor.updateParams(params.phaseVarWeightR,
        params.phaseVarWeightZ, params.phaseVarWeightAngle,
        params.phaseVarWeightFreq, params.phaseVarWeightBias);
  };


  ~FTag2TrackerNodelet() {
    alive = false;
  };


  virtual void onInit() {
    frameID = 0;

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
    dynCfgServer->setCallback(bind(&FTag2TrackerNodelet::configCallback, this, _1, _2));

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
    GET_PARAM(quadMaxScans);
    GET_PARAM(tagMaxStripAvgDiff);
    GET_PARAM(tagBorderMeanMaxThresh);
    GET_PARAM(tagBorderStdMaxThresh);
    GET_PARAM(phaseVarWeightR);
    GET_PARAM(phaseVarWeightZ);
    GET_PARAM(phaseVarWeightAngle);
    GET_PARAM(phaseVarWeightFreq);
    GET_PARAM(phaseVarWeightBias);
    GET_PARAM(markerWidthM);
    GET_PARAM(numberOfParticles);
    GET_PARAM(position_std);
    GET_PARAM(orientation_std);
    GET_PARAM(position_noise_std);
    GET_PARAM(orientation_noise_std);
    GET_PARAM(run_id);
    GET_PARAM(within_phase_range_n_sigma);
    GET_PARAM(within_phase_range_allowed_missmatches);
    GET_PARAM(within_phase_range_threshold);
#undef GET_PARAM
    dynCfgSyncReq = true;
    phaseVariancePredictor.updateParams(params.phaseVarWeightR,
        params.phaseVarWeightZ, params.phaseVarWeightAngle,
        params.phaseVarWeightFreq, params.phaseVarWeightBias);
#ifdef PARTICLE_FILTER
    PF.updateParameters(params.numberOfParticles, params.position_std, params.orientation_std, params.position_noise_std, params.orientation_noise_std, params.velocity_noise_std, params.acceleration_noise_std);
#endif
    FT.updateParameters(params.numberOfParticles, params.position_std, params.orientation_std, params.position_noise_std, params.orientation_noise_std, params.velocity_noise_std, params.acceleration_noise_std);
    FTag2Payload::updateParameters(params.within_phase_range_n_sigma, params.within_phase_range_allowed_missmatches, params.within_phase_range_threshold);
#ifdef CV_SHOW_IMAGES
    // Configure windows
    namedWindow("quad_1_trimmed", CV_GUI_EXPANDED);
    namedWindow("segments", CV_GUI_EXPANDED);
    namedWindow("quads", CV_GUI_EXPANDED);
    namedWindow("tags", CV_GUI_EXPANDED);
    namedWindow("hypotheses", CV_GUI_EXPANDED);
#endif

    // Setup ROS communication links
    image_transport::ImageTransport it(local_nh);
    tagDetectionsPub = local_nh.advertise<ftag2::TagDetections>("detected_tags", 1);
    firstTagImagePub = it.advertise("first_tag_image", 1);
    processedImagePub = it.advertise("overlaid_image", 1);
    imageSub = it.subscribe("image_in", 1, &FTag2TrackerNodelet::imageCallback, this);
    cameraSub = it.subscribeCamera("camera_in", 1, &FTag2TrackerNodelet::cameraCallback, this);

#ifdef PARTICLE_FILTER
    tag_observations = std::vector<FTag2Pose>();
    tracking = false;
    starting_time = ParticleFilter::clock::now();
    last_frame_time = ParticleFilter::clock::now();
    pubTrack = local_nh.advertise<std_msgs::Float64MultiArray>("detected_and_tracked_pose", 1);
#endif

    markerInfoPub = local_nh.advertise<ftag2::FreqTBMarkerInfo>("marker_info", 1);
    vis_pub = local_nh.advertise<visualization_msgs::Marker>( "visualization_marker", 0 );
	marker.type = visualization_msgs::Marker::SPHERE;
	marker.action = visualization_msgs::Marker::ADD;
	marker.header.frame_id = "camera";
	marker.lifetime = ros::Duration();

    // Finish initialization
    alive = true;
    NODELET_INFO("FTag2 reader nodelet initialized");
  };


  void configCallback(ftag2::CamTestbenchConfig& config, uint32_t level) {
    if (!alive) return;
    params = config;
    phaseVariancePredictor.updateParams(params.phaseVarWeightR,
        params.phaseVarWeightZ, params.phaseVarWeightAngle,
        params.phaseVarWeightFreq, params.phaseVarWeightBias);
#ifdef PARTICLE_FILTER
    PF.updateParameters(params.numberOfParticles, params.position_std, params.orientation_std, params.position_noise_std, params.orientation_noise_std, params.velocity_noise_std, params.acceleration_noise_std);
#endif
    FT.updateParameters(params.numberOfParticles, params.position_std, params.orientation_std, params.position_noise_std, params.orientation_noise_std, params.velocity_noise_std, params.acceleration_noise_std);
    FTag2Payload::updateParameters(params.within_phase_range_n_sigma, params.within_phase_range_allowed_missmatches, params.within_phase_range_threshold);
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

    frameID++;
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

	//	cout << "Params position_std = " << params.position_std << endl;
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
    std::list<Quad> quads = detectQuads(segments,
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
    std::vector<FTag2Marker> tags;
    FTag2Marker currTag;
    for (const Quad& currQuad: quads) {
      // Reject quads that overlap with already-detected tags (which have larger area than current quad)
      bool overlap = false;
      for (const FTag2Marker& prevTag: tags) {
        if (vc_math::checkPolygonOverlap(prevTag.corners, currQuad.corners)) {
          overlap = true;
          break;
        }
      }
      if (overlap) continue;

      // Check whether we have scanned enough quads
      quadCount++;
      if (quadCount > params.quadMaxScans) break;

      // Extract rectified quad image from frame
      quadExtractorP.tic();
      quadImg = extractQuadImg(sourceImg, currQuad, params.quadMinWidth*(8/6));
      quadExtractorP.toc();
      if (quadImg.empty()) { continue; }

      // Decode tag
      decodeQuadP.tic();
      try {
        currTag = FTag2Decoder::decodeQuad(quadImg, currQuad,
            params.markerWidthM,
            cameraIntrinsic, cameraDistortion,
            params.tagMaxStripAvgDiff,
            params.tagBorderMeanMaxThresh, params.tagBorderStdMaxThresh,
            phaseVariancePredictor);
      } catch (const std::string& err) {
        continue;
      }
      decodeQuadP.toc();

      // Store tag in list
      tags.push_back(currTag);
    } // Scan through all detected quads


    // Post-process largest detected tag
    if (tags.size() >= 1) {
      const FTag2Marker& firstTag = tags[0];

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
    for (const FTag2Marker& tag: tags) {
      drawQuadWithCorner(processedImg, tag.corners);
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

      //      double k=10.0;
      for (const FTag2Marker& tag: tags) {
        ftag2::TagDetection tagMsg;
        tagMsg.pose.position.x = tag.pose.position_x;
        tagMsg.pose.position.y = tag.pose.position_y;
        tagMsg.pose.position.z = tag.pose.position_z;
        tagMsg.pose.orientation.w = tag.pose.orientation_w;
        tagMsg.pose.orientation.x = tag.pose.orientation_x;
        tagMsg.pose.orientation.y = tag.pose.orientation_y;
        tagMsg.pose.orientation.z = tag.pose.orientation_z;
        tagMsg.markerPixelWidth = tag.rectifiedWidth;
        const double* magsPtr = (double*) tag.payload.mags.data;
        tagMsg.mags = std::vector<double>(magsPtr, magsPtr + tag.payload.mags.rows * tag.payload.mags.cols);
        const double* phasesPtr = (double*) tag.payload.phases.data;
        tagMsg.phases = std::vector<double>(phasesPtr, phasesPtr + tag.payload.phases.rows * tag.payload.phases.cols);
        tagsMsg.tags.push_back(tagMsg);
      }
      tagDetectionsPub.publish(tagsMsg);

      FTag2Marker tag = tags[0];
      ftag2::FreqTBMarkerInfo markerInfoMsg;
      markerInfoMsg.frameID = frameID;
      markerInfoMsg.radius = sqrt(tag.pose.position_x*tag.pose.position_x + tag.pose.position_y*tag.pose.position_y);
      markerInfoMsg.z = tag.pose.position_z;
      markerInfoMsg.oop_rotation = tag.pose.getAngleFromCamera()*vc_math::radian;
      markerInfoMsg.pose.position.x = tag.pose.position_x;
      markerInfoMsg.pose.position.y = tag.pose.position_y;
      markerInfoMsg.pose.position.z = tag.pose.position_z;
      markerInfoMsg.pose.orientation.w = tag.pose.orientation_w;
      markerInfoMsg.pose.orientation.x = tag.pose.orientation_x;
      markerInfoMsg.pose.orientation.y = tag.pose.orientation_y;
      markerInfoMsg.pose.orientation.z = tag.pose.orientation_z;
      markerInfoMsg.markerPixelWidth = quadImg.rows;
      const double* magsPtr = (double*) tag.payload.mags.data;
      markerInfoMsg.mags = std::vector<double>(magsPtr, magsPtr + tag.payload.mags.rows * tag.payload.mags.cols);
      const double* phasesPtr = (double*) tag.payload.phases.data;
      markerInfoMsg.phases = std::vector<double>(phasesPtr, phasesPtr + tag.payload.phases.rows * tag.payload.phases.cols);
      markerInfoMsg.phaseVars = tag.payload.phaseVariances;
      markerInfoMsg.hasSignature = tag.payload.hasSignature;
      markerInfoMsg.hasValidXORs = tag.payload.hasValidXORs;
      markerInfoMsg.bitChunksStr = tag.payload.bitChunksStr;
      markerInfoMsg.decodedPayloadStr = tag.payload.decodedPayloadStr;
      markerInfoMsg.numDecodedPhases = tag.payload.numDecodedPhases;
      markerInfoMsg.numDecodedSections = tag.payload.numDecodedSections;
      markerInfoPub.publish(markerInfoMsg);

      tf::Quaternion rMat(tags[0].pose.orientation_x,tags[0].pose.orientation_y,tags[0].pose.orientation_z,tags[0].pose.orientation_w);
      static tf::TransformBroadcaster br;
      tf::Transform transform;
      transform.setOrigin( tf::Vector3( tags[0].pose.position_x, tags[0].pose.position_y, tags[0].pose.position_z ) );
      transform.setRotation( rMat );
      br.sendTransform( tf::StampedTransform( transform, ros::Time::now(), "camera", "last_obs" ) );

#ifdef PARTICLE_FILTER
      for ( FTag2Marker tag: tags )
        tag_observations.push_back(tag.pose);
      if ( tracking == false )
      {
        tracking = true;
        PF = ParticleFilter(params.numberOfParticles, tag_observations );
        //    	  cv::waitKey();
      }
#endif
    }

    // Udpate marker filter (with or without new tags)
    trackerP.tic();
    FT.updateParameters(params.numberOfParticles, params.position_std, params.orientation_std, params.position_noise_std, params.orientation_noise_std, params.velocity_noise_std, params.acceleration_noise_std);
    FTag2Payload::updateParameters(params.within_phase_range_n_sigma, params.within_phase_range_allowed_missmatches, params.within_phase_range_threshold);
    FT.step( tags, params.markerWidthM, cameraIntrinsic, cameraDistortion );
    trackerP.toc();

    /* TODO: fix */
//    for ( int c = 0; c < cornersInCamSpace.cols; c++ )
//    {
//		std::ostringstream frameName;
//		frameName << "cor_" << c;
////		marker.header.frame_id = frameName.str();
//		marker.header.stamp = ros::Time();
////		marker.ns = "ftag2";
//		marker.id = c;
//		marker.pose.position.x = cornersInCamSpace.at<double>(0,c);
//		marker.pose.position.y = cornersInCamSpace.at<double>(1,c);
//		marker.pose.position.z = cornersInCamSpace.at<double>(2,c);
//		marker.scale.x = 0.01;
//		marker.scale.y = 0.01;
//		marker.scale.z = 0.01;
//		marker.color.a = 1.0;
//		marker.color.r = 0.0+(double)(c/5.0);
//		marker.color.g = 1.0-(double)(c/5.0);
//		marker.color.b = 0.0+(double)(c/5.0);
//	//	//only if using a MESH_RESOURCE marker type:
//		vis_pub.publish( marker );
//    }
    {
        cv::Mat overlaidImg;
        cv::cvtColor(sourceImg, overlaidImg, CV_RGB2BGR);
        for (const auto filt: FT.filters) {
        	auto tag_ = filt.hypothesis;
        	if ( !tag_.back_proj_corners.empty() )
        		drawQuadWithCorner(overlaidImg, tag_.back_proj_corners );
        }
        cv::imshow("Back proj", overlaidImg);
    }

    // Decode tracked payloads
    decodePayloadP.tic();
    bool first = true;
    for (MarkerFilter& tracked_tag: FT.filters) {
      FTag2Decoder::decodePayload(tracked_tag.hypothesis.payload, DECODE_PAYLOAD_N_STD_THRESH);
      if (first) {
    	  first = false;
//    	  NODELET_INFO_STREAM("bitChunks: " << cv::format(tracked_tag.hypothesis.payload.bitChunks, "matlab"));
      }
    }
    decodePayloadP.toc();

#ifdef PARTICLE_FILTER
    if (tracking == true)
    {
      PF.motionUpdate();
      //cv::waitKey();
      PF.measurementUpdate(tag_observations);
      PF.normalizeWeights();
      //PF.computeMeanPose();
      FTag2Pose track = PF.computeModePose();
      //PF.displayParticles();
      PF.resample();

      tag_observations.clear();

      std_msgs::Float64MultiArray array;
      array.data.clear();

      array.data.push_back(params.run_id);
      array.data.push_back(params.position_noise_std);
      array.data.push_back(params.velocity_noise_std);
      array.data.push_back(params.acceleration_noise_std);
      array.data.push_back(params.orientation_noise_std);
      array.data.push_back(params.position_std);
      array.data.push_back(params.orientation_noise_std);
      array.data.push_back(track.position_x);
      array.data.push_back(track.position_y);
      array.data.push_back(track.position_z);
      array.data.push_back(track.orientation_x);
      array.data.push_back(track.orientation_y);
      array.data.push_back(track.orientation_z);
      array.data.push_back(track.orientation_w);

      if (tag_observations.size()>0)
      {
        array.data.push_back(tag_observations[0].position_x);
        array.data.push_back(tag_observations[0].position_y);
        array.data.push_back(tag_observations[0].position_z);
        array.data.push_back(tag_observations[0].orientation_x);
        array.data.push_back(tag_observations[0].orientation_y);
        array.data.push_back(tag_observations[0].orientation_z);
        array.data.push_back(tag_observations[0].orientation_w);
      }
      pubTrack.publish(array);

    }
#endif

    // Visualize tag hypotheses
#ifdef CV_SHOW_IMAGES
    {
      cv::Mat overlaidImg;
      cv::cvtColor(sourceImg, overlaidImg, CV_RGB2BGR);

      // Do not draw quads

      // Draw detected tag observations: red
      for (const FTag2Marker& tagObs: tags) {
        drawQuadWithCorner(overlaidImg, tagObs.corners);
      }

      // Draw matched tag hypotheses: blue border, cyan-on-blue text
#ifdef DISPLAY_DECODED_TAG_PAYLOADS
      // TODO: 1 passive tag hypotheses: grey border, grey-on-white text (and also for #else clause)
      for (const MarkerFilter& trackedTag: FT.filters) {
        const FTag2Payload& trackedPayload = trackedTag.hypothesis.payload;
        if (trackedPayload.numDecodedSections >= DISPLAY_HYPOTHESIS_MIN_NUM_DECODED_SECTIONS) {
          drawQuadWithCorner(overlaidImg, trackedTag.hypothesis.corners,
              CV_RGB(255, 0, 0), CV_RGB(0, 255, 255),
              CV_RGB(0, 255, 255), CV_RGB(255, 0, 0));
        }
      }
      for (const MarkerFilter& trackedTag: FT.filters) {
        const FTag2Payload& trackedPayload = trackedTag.hypothesis.payload;
        if (trackedPayload.numDecodedSections >= DISPLAY_HYPOTHESIS_MIN_NUM_DECODED_SECTIONS) {
          drawMarkerLabel(overlaidImg, trackedTag.hypothesis.corners,
              trackedPayload.decodedPayloadStr,
              cv::FONT_HERSHEY_SIMPLEX, 1, 0.4,
              CV_RGB(0, 255, 255), CV_RGB(0, 0, 255));
        }
      }
#else
      for (const MarkerFilter& trackedTag: FT.filters) {
        const FTag2Payload& trackedPayload = trackedTag.hypothesis.payload;
        if (trackedPayload.numDecodedPhases >= DISPLAY_HYPOTHESIS_MIN_NUM_DECODED_PHASES) {
          drawQuadWithCorner(overlaidImg, trackedTag.hypothesis.corners,
              CV_RGB(255, 0, 0), CV_RGB(0, 255, 255),
              CV_RGB(0, 255, 255), CV_RGB(255, 0, 0));
        }
      }
      for (const MarkerFilter& trackedTag: FT.filters) {
        const FTag2Payload& trackedPayload = trackedTag.hypothesis.payload;
        if (trackedPayload.numDecodedPhases >= DISPLAY_HYPOTHESIS_MIN_NUM_DECODED_PHASES) {
          drawMarkerLabel(overlaidImg, trackedTag.hypothesis.corners,
              trackedPayload.bitChunksStr,
              cv::FONT_HERSHEY_SIMPLEX, 1, 0.4,
              CV_RGB(0, 255, 255), CV_RGB(0, 0, 255));
        }
      }
#endif

      cv::imshow("hypotheses", overlaidImg);
    }
#endif

#ifdef PROFILER
    // Update profiler
    durationP.toc();
    if (profilerDelaySec > 0) {
      ros::Time currTime = ros::Time::now();
      ros::Duration td = currTime - latestProfTime;
      if (td.toSec() > profilerDelaySec) {
        cout << "detectLineSegments: " << lineSegP.getStatsString() << endl;
        cout << "detectQuads: " << quadP.getStatsString() << endl;
        cout << "extractTags: " << quadExtractorP.getStatsString() << endl;
        cout << "decodeQuad: " << decodeQuadP.getStatsString() << endl;
        cout << "tracker: " << trackerP.getStatsString() << endl;
        cout << "decodePayload: " << decodePayloadP.getStatsString() << endl;

        cout << "Pipeline Duration: " << durationP.getStatsString() << endl;
        cout << "Pipeline Rate: " << rateP.getStatsString() << endl;
        latestProfTime = currTime;
      }
    }
#endif

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
PLUGINLIB_DECLARE_CLASS(ftag2, FTag2TrackerNodelet, ftag2::FTag2TrackerNodelet, nodelet::Nodelet)
