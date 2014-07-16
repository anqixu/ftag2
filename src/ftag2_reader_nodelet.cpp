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
#include "ftag2/TagDetections.h"


using namespace std;
using namespace cv;
using namespace vc_math;


typedef dynamic_reconfigure::Server<ftag2::FTag2ReaderConfig> ReconfigureServer;


#define CV_SHOW_IMAGES


namespace ftag2 {


class FTag2ReaderNodelet : public nodelet::Nodelet {
protected:
  //FTag2Tracker FT;

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

  PayloadFilter filter;

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
    params.quadMinWidth = 15;
    params.quadMinAngleIntercept = 30.0;
    params.quadHashMapWidth = 10;
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
    phaseVariancePredictor.updateParams(params.phaseVarWeightR,
        params.phaseVarWeightZ, params.phaseVarWeightAngle,
        params.phaseVarWeightFreq, params.phaseVarWeightBias);
  };


  ~FTag2ReaderNodelet() {
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

    // Setup and initialize dynamic reconfigure server
    dynCfgServer = new ReconfigureServer(dynCfgMutex, local_nh);
    dynCfgServer->setCallback(bind(&FTag2ReaderNodelet::configCallback, this, _1, _2));

    // Parse static parameters and update dynamic reconfigure values
    #define GET_PARAM(v) \
      local_nh.param(std::string(#v), params.v, params.v)
    GET_PARAM(quadMinWidth);
    GET_PARAM(quadMinAngleIntercept);
    GET_PARAM(quadHashMapWidth);
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
    #undef GET_PARAM
    dynCfgSyncReq = true;
    phaseVariancePredictor.updateParams(params.phaseVarWeightR,
        params.phaseVarWeightZ, params.phaseVarWeightAngle,
        params.phaseVarWeightFreq, params.phaseVarWeightBias);

#ifdef CV_SHOW_IMAGES
    // Configure windows
    //namedWindow("quad_1", CV_GUI_EXPANDED);
    //namedWindow("quad_1_trimmed", CV_GUI_EXPANDED);
    namedWindow("segments", CV_GUI_EXPANDED);
    namedWindow("quads", CV_GUI_EXPANDED);
    //namedWindow("hashquads", CV_GUI_EXPANDED);
    //namedWindow("tags", CV_GUI_EXPANDED);

    namedWindow("contours", CV_GUI_EXPANDED);
#endif

    _profilers["01_pp_duration"] = Profiler();
    _profilers["02_pp_rate"] = Profiler();
    _profilers["10_pp_rgb2gray"] = Profiler();
    _profilers["20_pp_line"] = Profiler();
    _profilers["30_pp_quad"] = Profiler();
    _profilers["31_quad_edgepair"] = Profiler();
    _profilers["32_quad_dft"] = Profiler();
    _profilers["40_pp_hashquad"] = Profiler();
    _profilers["41_hash_edgepair"] = Profiler();
    _profilers["42_hash_dft"] = Profiler();

    // Resolve image topic names
    std::string imageTopic = local_nh.resolveName("image_in");
    std::string cameraTopic = local_nh.resolveName("camera_in");

    // Setup ROS communication links
    image_transport::ImageTransport it(local_nh);
    tagDetectionsPub = local_nh.advertise<ftag2::TagDetections>("detected_tags", 1);
    firstTagImagePub = it.advertise("first_tag_image", 1);
    processedImagePub = it.advertise("overlaid_image", 1);
    imageSub = it.subscribe(imageTopic, 1, &FTag2ReaderNodelet::imageCallback, this, transportType);
    cameraSub = it.subscribeCamera(cameraTopic, 1, &FTag2ReaderNodelet::cameraCallback, this, transportType);

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
    int blockSize = 7;
    double meanWeightC = 7;
    bool doErosion = false;
    double _minQuadWidth = 10;
    if (params.quadMaxScans >= 3 && params.quadMaxScans <= 9) {
      blockSize = params.quadMaxScans;
      if (blockSize % 2 == 0) blockSize += 1;
    }
    meanWeightC = params.quadMaxEdgeGapAlignAngle;
    doErosion = (params.phaseVarWeightR != 0);
    _minQuadWidth = params.quadHashMapWidth;

    // Update profiler
    _profilers["02_pp_rate"].try_toc();
    _profilers["02_pp_rate"].tic();
    _profilers["01_pp_duration"].tic();

    // Convert source image to grayscale
    _profilers["10_pp_rgb2gray"].tic();
    cv::Mat grayImg;
    cv::cvtColor(sourceImg, grayImg, CV_RGB2GRAY);
    _profilers["10_pp_rgb2gray"].toc();

    _profilers["20_pp_line"].tic();
    // 2. Threshold image
    cv::Mat threshImg;
    cv::adaptiveThreshold(grayImg, threshImg, 255,
        ADAPTIVE_THRESH_MEAN_C,
        THRESH_BINARY_INV,
        blockSize, meanWeightC);

    // 3. Erode image
    if (doErosion) {
      cv::Mat threshImg2;
      cv::erode(threshImg, threshImg2, cv::Mat());
      threshImg = threshImg2;
    }
    _profilers["20_pp_line"].toc();
#ifdef CV_SHOW_IMAGES
    {
      cv::imshow("segments", threshImg);
    }
#endif

    // 4. Find all rectangles in thresholded image
    _profilers["30_pp_quad"].tic();
    std::list<Quad> quads;
    unsigned int minContourSize=0.04 * std::max(threshImg.cols, threshImg.rows) * 4;
    unsigned int maxContourSize=0.5 * std::max(threshImg.cols, threshImg.rows) * 4;
    std::vector< std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy; // Throw-away var; unpopulated due to CV_RETR_LIST arg
    std::vector<cv::Point> approxCurve;

    cv::Mat threshImg2;
    threshImg.copyTo(threshImg2); // cv::findContours will modify this image
    cv::findContours(threshImg2, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

    for (unsigned int i=0; i < contours.size(); i++) {
      // Reject small or large contours
      if (contours[i].size() < minContourSize || contours[i].size() > maxContourSize) continue;

      // Approximate to a polygon
      double approxPolyEpsilon = double(contours[i].size())*0.05;
      cv::approxPolyDP(contours[i], approxCurve, approxPolyEpsilon, true); // closed = true

      // Reject non-quadrilaterals
      if (approxCurve.size() != 4) continue;

      // Reject non-convex polygons
      if (!cv::isContourConvex(cv::Mat(approxCurve))) continue;

      // Reject quads that have too small widths
      double minQuadWidth = std::numeric_limits<double>::infinity();
      for (int j=0; j<4; j++) {
        double d = vc_math::dist(approxCurve[j].x, approxCurve[j].y, approxCurve[(j+1)%4].x, approxCurve[(j+1)%4].y);
        if (d < minQuadWidth) minQuadWidth = d;
      }

      if (minQuadWidth > _minQuadWidth) {
        Quad quad;
        for (int j = 0; j < 4; j++) {
          quad.corners[j].x = approxCurve[j].x;
          quad.corners[j].y = approxCurve[j].y;
        }
        quad.updateArea();
        quads.push_back(quad);
      }
    }
    _profilers["30_pp_quad"].toc();
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

    // 5. Refine corners

    // Update profiler
    _profilers["01_pp_duration"].toc();
    if (profilerDelaySec > 0) {
      ros::Time currTime = ros::Time::now();
      ros::Duration td = currTime - latestProfTime;
      if (td.toSec() > profilerDelaySec) {
        ROS_WARN_STREAM("===== PROFILERS =====");
        for (std::map<std::string, Profiler>::iterator it = _profilers.begin();
            it != _profilers.end(); it++) {
          ROS_WARN_STREAM(it->first << ": " << it->second.getStatsString());
        }
        ROS_WARN_STREAM("");
        latestProfTime = currTime;
      }
    }

    // Allow OpenCV HighGUI events to process
#ifdef CV_SHOW_IMAGES
    char c = waitKey(1);
    if (c == 'x' || c == 'X') {
      ros::shutdown();
    } else if (c == 'r' || c == 'R') {
      for (std::map<std::string, Profiler>::iterator it = _profilers.begin();
          it != _profilers.end(); it++) {
        it->second.reset();
      }
    }
#endif
  };


  void processImageNonAruco(const cv::Mat sourceImg, int ID) {
    // Update profiler
    _profilers["02_pp_rate"].try_toc();
    _profilers["02_pp_rate"].tic();
    _profilers["01_pp_duration"].tic();

    // Convert source image to grayscale
    _profilers["10_pp_rgb2gray"].tic();
    cv::Mat grayImg;
    cv::cvtColor(sourceImg, grayImg, CV_RGB2GRAY);
    _profilers["10_pp_rgb2gray"].toc();

    // 1. Detect line segments
    _profilers["20_pp_line"].tic();
    std::vector<cv::Vec4i> segments = detectLineSegments(grayImg);
    _profilers["20_pp_line"].toc();
#ifdef CV_SHOW_IMAGES
    {
      cv::Mat overlaidImg;
      cv::cvtColor(sourceImg, overlaidImg, CV_RGB2BGR);
      drawLineSegments(overlaidImg, segments);
      cv::imshow("segments", overlaidImg);
    }
#endif

    // 2. Detect quadrilaterals
    _profilers["30_pp_quad"].tic();
    std::list<Quad> quads = detectAllQuads(segments,
        params.quadMinAngleIntercept*degree,
        params.quadMaxTIntDistRatio,
        params.quadMaxEndptDistRatio,
        params.quadMaxCornerGapEndptDistRatio,
        params.quadMaxEdgeGapDistRatio,
        params.quadMaxEdgeGapAlignAngle*degree,
        params.quadMinWidth);
    quads.sort(Quad::compareArea);
    _profilers["30_pp_quad"].toc();
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

    // 3. Detect quadrilaterals, new way
    _profilers["40_pp_hashquad"].tic();
    std::list<Quad> hashquads = detectEndptQuads(segments,
        grayImg.cols, grayImg.rows,
        params.quadMinAngleIntercept*degree,
        params.quadHashMapWidth,
        params.quadMaxTIntDistRatio,
        params.quadMaxEndptDistRatio,
        params.quadMaxCornerGapEndptDistRatio,
        params.quadMaxEdgeGapDistRatio,
        params.quadMaxEdgeGapAlignAngle*degree,
        params.quadMinWidth);
    hashquads.sort(Quad::compareArea);
    _profilers["40_pp_hashquad"].toc();
#ifdef CV_SHOW_IMAGES
    {
      cv::Mat overlaidImg;
      cv::cvtColor(sourceImg, overlaidImg, CV_RGB2BGR);
      for (const Quad& quad: hashquads) {
        drawQuad(overlaidImg, quad.corners);
      }
      cv::imshow("hashquads", overlaidImg);
    }
#endif

    // Update profiler
    _profilers["01_pp_duration"].toc();
    if (profilerDelaySec > 0) {
      ros::Time currTime = ros::Time::now();
      ros::Duration td = currTime - latestProfTime;
      if (td.toSec() > profilerDelaySec) {
        ROS_WARN_STREAM("===== PROFILERS =====");
        for (std::map<std::string, Profiler>::iterator it = _profilers.begin();
            it != _profilers.end(); it++) {
          ROS_WARN_STREAM(it->first << ": " << it->second.getStatsString());
        }
        ROS_WARN_STREAM("");
        latestProfTime = currTime;
      }
    }

    // Allow OpenCV HighGUI events to process
#ifdef CV_SHOW_IMAGES
    char c = waitKey(1);
    if (c == 'x' || c == 'X') {
      ros::shutdown();
    } else if (c == 'r' || c == 'R') {
      for (std::map<std::string, Profiler>::iterator it = _profilers.begin();
          it != _profilers.end(); it++) {
        it->second.reset();
      }
    }
#endif
  };

  void processImageOLD(const cv::Mat sourceImg, int ID) {
    // Update profiler
    rateP.try_toc();
    rateP.tic();
    durationP.tic();

    // Convert source image to grayscale
    cv::Mat grayImg;
    cv::cvtColor(sourceImg, grayImg, CV_RGB2GRAY);

    // 1. Detect line segments
    lineSegP.tic();
    std::vector<cv::Vec4i> segments = detectLineSegments(grayImg);
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
    std::list<Quad> quads = detectAllQuads(segments,
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
        currTag = FTag2Decoder::decodeQuad(quadImg, currQuad,
            params.markerWidthM,
            1, /* TODO: change for params.num_samples_per_row */
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
        cout << "decodeQuad: " << decoderP.getStatsString() << endl;
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
