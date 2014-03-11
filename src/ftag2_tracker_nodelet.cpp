#include "detector/FTag2Detector.hpp"
#include "decoder/FTag2Decoder.hpp"

#include "common/Profiler.hpp"
#include "common/VectorAndCircularMath.hpp"
#include "common/FTag2Payload.hpp"

#include "tracker/FTag2Tracker.hpp"

#include <ros/ros.h>
#include <dynamic_reconfigure/server.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <nodelet/nodelet.h>
#include <ftag2/CamTestbenchConfig.h>
#include "ftag2/TagDetections.h"

#include "common/FTag2Pose.hpp"
#include "tracker/ParticleFilter.hpp"
#include "std_msgs/Float64MultiArray.h"


using namespace std;
using namespace cv;
using namespace vc_math;


typedef dynamic_reconfigure::Server<ftag2::CamTestbenchConfig> ReconfigureServer;


#define CV_SHOW_IMAGES
#undef PARTICLE_FILTER

namespace ftag2 {


class FTag2TrackerNodelet : public nodelet::Nodelet {
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
  Profiler lineSegP, quadP, quadExtractorP, decoderP, durationP, rateP;
  ros::Time latestProfTime;
  double profilerDelaySec;

public:
  FTag2TrackerNodelet() : nodelet::Nodelet(),
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
    params.quadMaxScans = 10;
    params.tagMaxStripAvgDiff = 15.0;
    params.tagBorderMeanMaxThresh = 80.0;
    params.tagBorderStdMaxThresh = 30.0;
    params.phaseVarWeightR = 0;
    params.phaseVarWeightZ = 0;
    params.phaseVarWeightAngle = 0;
    params.phaseVarWeightFreq = 0;
    params.phaseVarWeightBias = 10*10;
    params.markerWidthM = 0.07;
    params.numberOfParticles = 100;
    params.position_std = 0.15;
    params.orientation_std = 0.15;
    params.position_noise_std = 0.15;
    params.orientation_noise_std = 0.15;
    params.velocity_noise_std = 0.05;
    params.acceleration_noise_std = 0.01;
    params.run_id = 1;
  };


  ~FTag2TrackerNodelet() {
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
#undef GET_PARAM
    dynCfgSyncReq = true;
#ifdef PARTICLE_FILTER
    PF.updateParameters(params.numberOfParticles, params.position_std, params.orientation_std, params.position_noise_std, params.orientation_noise_std, params.velocity_noise_std, params.acceleration_noise_std);
    phaseVariancePredictor.updateParams(params.phaseVarWeightR,
        params.phaseVarWeightZ, params.phaseVarWeightAngle,
        params.phaseVarWeightFreq, params.phaseVarWeightBias);
#endif
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
    imageSub = it.subscribe("image_in", 1, &FTag2TrackerNodelet::imageCallback, this);
    cameraSub = it.subscribeCamera("camera_in", 1, &FTag2TrackerNodelet::cameraCallback, this);

#ifdef PARTICLE_FILTER
    tag_observations = std::vector<FTag2Pose>();
    tracking = false;
    starting_time = ParticleFilter::clock::now();
    last_frame_time = ParticleFilter::clock::now();
    pubTrack = local_nh.advertise<std_msgs::Float64MultiArray>("detected_and_tracked_pose", 1);
#endif

    // Finish initialization
    alive = true;
    NODELET_INFO("FTag2 reader nodelet initialized");
  };


  void configCallback(ftag2::CamTestbenchConfig& config, uint32_t level) {
    if (!alive) return;
    params = config;
#ifdef PARTICLE_FILTER
    PF.updateParameters(params.numberOfParticles, params.position_std, params.orientation_std, params.position_noise_std, params.orientation_noise_std, params.velocity_noise_std, params.acceleration_noise_std);
    phaseVariancePredictor.updateParams(params.phaseVarWeightR,
        params.phaseVarWeightZ, params.phaseVarWeightAngle,
        params.phaseVarWeightFreq, params.phaseVarWeightBias);
#endif
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
      decoderP.tic();
      try {
        currTag = FTag2Decoder::decodeTag(quadImg, currQuad,
            params.markerWidthM,
            cameraIntrinsic, cameraDistortion,
            params.tagMaxStripAvgDiff,
            params.tagBorderMeanMaxThresh, params.tagBorderStdMaxThresh,
            phaseVariancePredictor);
      } catch (const std::string& err) {
        continue;
      }
      decoderP.toc();

      // Store tag in list
      tags.push_back(currTag);
    } // Scan through all detected quads




    if (tags.size() > 0) {
      NODELET_INFO_STREAM(ID << ": " << tags.size() << " tags (quads: " << quads.size() << ")");
    } else if (quads.size() > 0) {
      NODELET_WARN_STREAM(ID << ": " << tags.size() << " tags (quads: " << quads.size() << ")");
    } else {
//      NODELET_ERROR_STREAM(ID << ": " << tags.size() << " tags (quads: " << quads.size() << ")");
    }



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

      double k=10.0;
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


      /* DELETEME */
      for (int i=0; i<tags.size(); i++)
      {
    	  tags[i].payload.phaseVariances.push_back(k);
    	  k--;
      }
      /* ........ */


      FTag2Tracker FT;
      FT.director(tags);


#ifdef PARTICLE_FILTER
      for ( FTag2Marker tag: tags )
    	  tag_observations.push_back(tag.pose);
      if ( tracking == false )
      {
    	  tracking = true;
    	  PF = ParticleFilter(params.numberOfParticles, tag_observations, ParticleFilter::clock::now() );
//    	  cv::waitKey();
      }
#endif
    }

#ifdef PARTICLE_FILTER
    if (tracking == true)
    {
    	PF.motionUpdate(ParticleFilter::clock::now());
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

    	cout << "RUN ID: " << params.run_id << endl;

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
    //			  cout << "PUBLIQUE: " << track.position_x << endl;

    }
#endif


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
PLUGINLIB_DECLARE_CLASS(ftag2, FTag2TrackerNodelet, ftag2::FTag2TrackerNodelet, nodelet::Nodelet)
