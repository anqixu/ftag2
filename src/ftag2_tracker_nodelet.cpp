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
#include "ftag2/FTag2TrackerConfig.h"
#include "ftag2/TagDetections.h"

#include "std_msgs/Float64MultiArray.h"

#include "ftag2/ARMarkerFT.h"
#include "ftag2/ARMarkersFT.h"

#include <tf/transform_broadcaster.h>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
using namespace std;
using namespace cv;
using namespace vc_math;

typedef dynamic_reconfigure::Server<ftag2::FTag2TrackerConfig> ReconfigureServer;


//#undef CV_SHOW_IMAGES
#define CV_SHOW_IMAGES
#undef DISPLAY_DECODED_TAG_PAYLOADS


#define DECODE_PAYLOAD_N_STD_THRESH (0.01)
#define DISPLAY_HYPOTHESIS_MIN_NUM_DECODED_PHASES (8)
#define DISPLAY_HYPOTHESIS_MIN_NUM_DECODED_SECTIONS (2)



namespace ftag2 {


class FTag2TrackerNodelet : public nodelet::Nodelet {
protected:
  FTag2Tracker FT;

  bool alive;

  int tagType;

  ReconfigureServer* dynCfgServer;
  boost::recursive_mutex dynCfgMutex;
  bool dynCfgSyncReq;

  image_transport::Subscriber imageSub;
  image_transport::CameraSubscriber cameraSub;
  ros::Publisher rawTagDetectionsPub;
  ros::Publisher decodedTagDetectionsPub;
  ros::Publisher arMarkerPub_;
  ros::Publisher rvizMarkerPub_;
  visualization_msgs::Marker rvizMarker_;

  image_transport::Publisher processedImagePub;
  image_transport::Publisher firstTagImagePub;

//  ros::Publisher vis_pub;
  int frameID;

  ftag2::FTag2TrackerConfig params;

  cv::Mat cameraIntrinsic, cameraDistortion;

  PhaseVariancePredictor phaseVariancePredictor;

  // DEBUG VARIABLES
  Profiler lineSegP, quadP, quadExtractorP, decodeQuadP, trackerP, decodePayloadP, durationP, rateP;
  ros::Time latestProfTime;
  double profilerDelaySec;

public:
  FTag2TrackerNodelet() : nodelet::Nodelet(),
      alive(false),
      tagType(FTag2Payload::FTAG2_6S5F3B),
      dynCfgServer(NULL),
      dynCfgSyncReq(false),
      frameID(0),
      latestProfTime(ros::Time::now()),
      profilerDelaySec(0) {
    // Set default parameter values
    params.quadFastDetector = false;
    params.quadRefineCorners = true;
    params.quadMaxScans = 10;
    params.tagMaxStripAvgDiff = 15.0;
    params.tagBorderMeanMaxThresh = 80.0;
    params.tagBorderStdMaxThresh = 30.0;
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
    params.within_phase_range_threshold = 70.0;
    FTag2Payload::updateParameters(params.within_phase_range_threshold);
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
    GET_PARAM(within_phase_range_threshold);
#undef GET_PARAM
    dynCfgSyncReq = true;
    phaseVariancePredictor.updateParams(params.phaseVarWeightR,
        params.phaseVarWeightZ, params.phaseVarWeightAngle,
        params.phaseVarWeightFreq, params.phaseVarWeightBias);
//    FT.updateParameters(params.numberOfParticles, params.position_std, params.orientation_std, params.position_noise_std, params.orientation_noise_std, params.velocity_noise_std, params.acceleration_noise_std);
    FTag2Payload::updateParameters(params.within_phase_range_threshold);
#ifdef CV_SHOW_IMAGES
    // Configure windows
    //namedWindow("quads", CV_GUI_EXPANDED);
    namedWindow("hypotheses", CV_GUI_EXPANDED);
    namedWindow("tags", CV_GUI_EXPANDED);
#endif

    // Resolve image topic names
    std::string imageTopic = local_nh.resolveName("image_in");
    std::string cameraTopic = local_nh.resolveName("camera_in");

    // Setup ROS communication links
    image_transport::ImageTransport it(local_nh);
    rawTagDetectionsPub = local_nh.advertise<ftag2::TagDetections>("detected_tags", 1);
    decodedTagDetectionsPub = local_nh.advertise<ftag2::TagDetections>("decoded_tags", 1);
    arMarkerPub_ = local_nh.advertise < ARMarkersFT > ("ft_pose_markers", 1);
    rvizMarkerPub_ = local_nh.advertise < visualization_msgs::Marker > ("ftag2_vis_Marker", 1);
    firstTagImagePub = it.advertise("first_tag_image", 1);
    processedImagePub = it.advertise("overlaid_image", 1);
    imageSub = it.subscribe(imageTopic, 1, &FTag2TrackerNodelet::imageCallback, this, transportType);
    cameraSub = it.subscribeCamera(cameraTopic, 1, &FTag2TrackerNodelet::cameraCallback, this, transportType);

//    vis_pub = local_nh.advertise<visualization_msgs::MarkerArray>( "ftag2_array", 1);

    // Finish initialization
    alive = true;
    NODELET_INFO("FTag2 tracker nodelet initialized");
  };


  void configCallback(ftag2::FTag2TrackerConfig& config, uint32_t level) {
    if (!alive) return;
    params = config;
    phaseVariancePredictor.updateParams(params.phaseVarWeightR,
        params.phaseVarWeightZ, params.phaseVarWeightAngle,
        params.phaseVarWeightFreq, params.phaseVarWeightBias);
    //FT.updateParameters(params.numberOfParticles, params.position_std, params.orientation_std, params.position_noise_std, params.orientation_noise_std, params.velocity_noise_std, params.acceleration_noise_std);
    FTag2Payload::updateParameters(params.within_phase_range_threshold);
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
      quads = detectQuadsViaContour(grayImg);
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
      if (quadCount > params.quadMaxScans) break;

      // Extract rectified quad image from frame
      quadExtractorP.tic();
      quadImg = extractQuadImg(sourceImg, currQuad);
      quadExtractorP.toc();
      if (quadImg.empty()) { continue; }

      // Decode tag
      decodeQuadP.tic();
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

      // Publish cropped tag image
      cv_bridge::CvImage cvCroppedTagImgRot(std_msgs::Header(),
          sensor_msgs::image_encodings::MONO8, firstTag.tagImg);
      cvCroppedTagImgRot.header.frame_id = boost::lexical_cast<std::string>(ID);
      firstTagImagePub.publish(cvCroppedTagImgRot.toImageMsg());
    }

    // Publish image overlaid with detected markers
    cv::Mat processedImg = sourceImg.clone();
    for (const Quad& quad: quads) {
      drawQuad(processedImg, quad.corners);
    }
    for (const FTag2Marker& tag: tags) {
      drawQuadWithCorner(processedImg, tag.tagCorners);
      drawMarkerLabel(processedImg, tag.tagCorners, tag.payload.bitChunksStr);
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
        tagMsg.markerPixelWidth = tag.tagWidth;
        const double* magsPtr = (double*) tag.payload.mags.data;
        tagMsg.mags = std::vector<double>(magsPtr, magsPtr + tag.payload.mags.rows * tag.payload.mags.cols);
        const double* phasesPtr = (double*) tag.payload.phases.data;
        tagMsg.phases = std::vector<double>(phasesPtr, phasesPtr + tag.payload.phases.rows * tag.payload.phases.cols);
        tagsMsg.tags.push_back(tagMsg);
      }
      rawTagDetectionsPub.publish(tagsMsg);

//      {
//      tf::Quaternion rMat(tags[0].pose.orientation_x,tags[0].pose.orientation_y,tags[0].pose.orientation_z,tags[0].pose.orientation_w);
//      static tf::TransformBroadcaster br;
//      tf::Transform transform;
//     transform.setOrigin( tf::Vector3( tags[0].pose.position_x, tags[0].pose.position_y, tags[0].pose.position_z ) );
//      transform.setRotation( rMat );
//      br.sendTransform( tf::StampedTransform( transform, ros::Time::now(), "camera", "last_obs" ) );
//      }
    }

    // Udpate marker filter (with or without new tags)
    trackerP.tic();
//    FT.updateParameters(params.numberOfParticles, params.position_std, params.orientation_std, params.position_noise_std, params.orientation_noise_std, params.velocity_noise_std, params.acceleration_noise_std);
    FTag2Payload::updateParameters(params.within_phase_range_threshold);
    FT.step( tags, params.markerWidthM, cameraIntrinsic, cameraDistortion );
    trackerP.toc();


#ifdef CV_SHOW_IMAGES
//    {
//    cv::Mat overlaidImg;
//        cv::cvtColor(sourceImg, overlaidImg, CV_RGB2BGR);
//        for (const auto filt: FT.filters) {
//          auto tag_ = filt.hypothesis;
//          if ( !tag_.back_proj_corners.empty() )
//            drawQuadWithCorner(overlaidImg, tag_.back_proj_corners );
//        }
//        cv::imshow("Back proj", overlaidImg);
//    }
#endif

    // Decode tracked payloads
    decodePayloadP.tic();
    bool first = true;
    for (MarkerFilter& tracked_tag: FT.filters) {
      decodePayload(tracked_tag.hypothesis.payload, DECODE_PAYLOAD_N_STD_THRESH);
      if (first) {
        first = false;
//        NODELET_INFO_STREAM("bitChunks: " << cv::format(tracked_tag.hypothesis.payload.bitChunks, "matlab"));
      }
    }
    decodePayloadP.toc();

    ARMarkersFT arPoseMarkers_;
    for ( const MarkerFilter &filter: FT.filters )
    {
      if ( !filter.got_detection_in_current_frame ) {
//        cout << "No detection in current frame" << endl;
        continue;
      }

      std::string tf_frame = "192_168_0_23";

      cout << "Payload: ";
      cout << filter.hypothesis.payload.bitChunksStr << endl;
      std::ostringstream ostr;
      bool valid_id = true;
//      cout << "Payload (6x): ";
      for( unsigned int i=0; i<35; i+=6 ) {
//        cout << filter.hypothesis.payload.bitChunksStr[i];
        if ( filter.hypothesis.payload.bitChunksStr[i]>='0' && filter.hypothesis.payload.bitChunksStr[i] <= '9' ) {
          ostr << filter.hypothesis.payload.bitChunksStr[i];
        }
        else {
          ostr.clear();
          valid_id = false;
          break;
        }
      }
      unsigned int marker_id = 0;
      if (valid_id == true){
        marker_id = std::stoi(ostr.str());
        }
      else {
            cout << "Not a valid ID" << endl;
            continue;
        }
      cout << " Marker_id: " << marker_id << endl; // << "\tOstr = " << ostr.str() << endl;
    std::ostringstream frameName;
    frameName << "filt_" << marker_id;
    tf::Quaternion rMat(filter.hypothesis.pose.orientation_x,filter.hypothesis.pose.orientation_y,filter.hypothesis.pose.orientation_z,filter.hypothesis.pose.orientation_w);

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    transform.setOrigin( tf::Vector3( filter.hypothesis.pose.position_x, filter.hypothesis.pose.position_y, filter.hypothesis.pose.position_z ) );
    transform.setRotation( rMat );
    br.sendTransform( tf::StampedTransform( transform, ros::Time::now(), tf_frame, frameName.str() ) );

    tf::poseTFToMsg(tf::StampedTransform( transform, ros::Time::now(), tf_frame, frameName.str() ), rvizMarker_.pose);
    rvizMarker_.header.frame_id = tf_frame;
    rvizMarker_.header.stamp = ros::Time::now();
    rvizMarker_.id = marker_id;

    rvizMarker_.scale.x = params.markerWidthM;
    rvizMarker_.scale.y = params.markerWidthM;
    rvizMarker_.scale.z = params.markerWidthM/2.0;
    rvizMarker_.ns = "basic_shapes";
    rvizMarker_.type = visualization_msgs::Marker::CUBE;
    rvizMarker_.action = visualization_msgs::Marker::ADD;

        rvizMarker_.color.r = 0.0f;
        rvizMarker_.color.g = 0.0f;
        rvizMarker_.color.b = 1.0f;
        rvizMarker_.color.a = 1.0;
        rvizMarker_.lifetime = ros::Duration (1.0);

    rvizMarkerPub_.publish(rvizMarker_);

    ARMarkerFT ar_pose_marker_;
    ar_pose_marker_.header.frame_id = tf_frame;
    ar_pose_marker_.header.stamp    = ros::Time::now();

    ar_pose_marker_.id              = marker_id;

    ar_pose_marker_.pose.pose.position.x = filter.hypothesis.pose.position_x;
    ar_pose_marker_.pose.pose.position.y = filter.hypothesis.pose.position_y;
    ar_pose_marker_.pose.pose.position.z = filter.hypothesis.pose.position_z;

    ar_pose_marker_.pose.pose.orientation.x = filter.hypothesis.pose.orientation_x;
    ar_pose_marker_.pose.pose.orientation.y = filter.hypothesis.pose.orientation_y;
    ar_pose_marker_.pose.pose.orientation.z = filter.hypothesis.pose.orientation_z;
    ar_pose_marker_.pose.pose.orientation.w = filter.hypothesis.pose.orientation_w;

    arPoseMarkers_.markers.push_back(ar_pose_marker_);

    }
    arMarkerPub_.publish(arPoseMarkers_);


    ftag2::TagDetections tagsMsg;
    tagsMsg.frameID = ID;
    for (const MarkerFilter& filter: FT.filters ) {
      ftag2::TagDetection tagMsg;
      FTag2Marker tag = filter.hypothesis;
      tagMsg.pose.position.x = tag.pose.position_x;
      tagMsg.pose.position.y = tag.pose.position_y;
      tagMsg.pose.position.z = tag.pose.position_z;
      tagMsg.pose.orientation.w = tag.pose.orientation_w;
      tagMsg.pose.orientation.x = tag.pose.orientation_x;
      tagMsg.pose.orientation.y = tag.pose.orientation_y;
      tagMsg.pose.orientation.z = tag.pose.orientation_z;
      tagMsg.markerPixelWidth = tag.tagWidth;
      const double* magsPtr = (double*) tag.payload.mags.data;
      tagMsg.mags = std::vector<double>(magsPtr, magsPtr + tag.payload.mags.rows * tag.payload.mags.cols);
      const double* phasesPtr = (double*) tag.payload.phases.data;
      tagMsg.phases = std::vector<double>(phasesPtr, phasesPtr + tag.payload.phases.rows * tag.payload.phases.cols);
//      tagMsg.tag.payload.bitChunksStr;
      tagMsg.IDString = tag.payload.bitChunksStr;
      tagsMsg.tags.push_back(tagMsg);
    }
    decodedTagDetectionsPub.publish(tagsMsg);


    // Visualize tag hypotheses
#ifdef CV_SHOW_IMAGES
    {
      cv::Mat overlaidImg;
      cv::cvtColor(sourceImg, overlaidImg, CV_RGB2BGR);

      // Do not draw quads

      // Draw detected tag observations: red
      for (const FTag2Marker& tagObs: tags) {
        drawQuadWithCorner(overlaidImg, tagObs.tagCorners);
      }

      // Draw matched tag hypotheses: blue border, cyan-on-blue text
#ifdef DISPLAY_DECODED_TAG_PAYLOADS
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
          drawQuadWithCorner(overlaidImg, trackedTag.hypothesis.tagCorners,
              CV_RGB(255, 0, 0), CV_RGB(0, 255, 255),
              CV_RGB(0, 255, 255), CV_RGB(255, 0, 0));
        }
      }
      for (const MarkerFilter& trackedTag: FT.filters) {
        const FTag2Payload& trackedPayload = trackedTag.hypothesis.payload;
        if (trackedPayload.numDecodedPhases >= DISPLAY_HYPOTHESIS_MIN_NUM_DECODED_PHASES) {
          drawMarkerLabel(overlaidImg, trackedTag.hypothesis.tagCorners,
              trackedPayload.bitChunksStr,
              cv::FONT_HERSHEY_SIMPLEX, 1, 0.4,
              CV_RGB(0, 255, 255), CV_RGB(0, 0, 255));
        }
      }
#endif

      cv::imshow("hypotheses", overlaidImg);
    }
#endif

    // -. Update profiler
    durationP.toc();
    if (profilerDelaySec > 0) {
      ros::Time currTime = ros::Time::now();
      ros::Duration td = currTime - latestProfTime;
      if (td.toSec() > profilerDelaySec) {
        ROS_WARN_STREAM("===== PROFILERS =====");
        ROS_WARN_STREAM("detectQuads: " << quadP.getStatsString());
        ROS_WARN_STREAM("extractTags: " << quadExtractorP.getStatsString());
        ROS_WARN_STREAM("decodeQuad: " << decodeQuadP.getStatsString());
        ROS_WARN_STREAM("tracker: " << trackerP.getStatsString());
        ROS_WARN_STREAM("decodePayload: " << decodePayloadP.getStatsString());
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
