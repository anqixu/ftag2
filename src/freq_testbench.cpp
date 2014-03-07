#include "detector/FTag2Detector.hpp"
#include "decoder/FTag2Decoder.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>
#include <algorithm>
#include <cassert>
#include <list>
#include <vector>
#include <thread>

#include <ros/ros.h>
#include <dynamic_reconfigure/server.h>
#include <ftag2/FreqTestbenchConfig.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <ftag2/FreqTBMarkerInfo.h>
#include <ftag2/FreqTBPhaseStats.h>

#include "common/Profiler.hpp"
#include "common/VectorAndCircularMath.hpp"
#include "common/GNUPlot.hpp"


//#define SAVE_IMAGES_FROM sourceImgRot
//#define ENABLE_PROFILER


using namespace std;
using namespace std::chrono;
using namespace cv;
using namespace vc_math;


typedef dynamic_reconfigure::Server<ftag2::FreqTestbenchConfig> ReconfigureServer;


class FTag2Testbench {
protected:
  std::thread spinThread;

  ros::NodeHandle local_nh;
  image_transport::ImageTransport it;

  image_transport::Publisher imagePub;
  image_transport::Publisher tagPub;
  ros::Publisher markerInfoPub;
  ros::Publisher phaseStatsPub;

  ReconfigureServer* dynCfgServer;
  boost::recursive_mutex dynCfgMutex;
  bool dynCfgSyncReq;

  cv::VideoCapture cam;
  cv::Mat sourceImg, sourceImgRot, grayImg, overlaidImg;

  ftag2::FreqTestbenchConfig params;

  bool alive;

  int dstID;
  char* dstFilename;

  Profiler lineSegP, quadP, quadExtractorP, decoderP, durationProf, rateProf;
  ros::Time latestProfTime;

  int waitKeyDelay;
  std::string saveImgDir;

  cv::Mat cameraIntrinsic, cameraDistortion;

  PhaseVariancePredictor phaseVariancePredictor;

  std::string targetTagPhasesStr;
  cv::Mat targetTagPhases;

  int frameID;
  cv::Mat currPhaseErrors;
  cv::Mat phaseErrorsSum;
  cv::Mat phaseErrorsSqrdSum;
  cv::Mat phaseErrorsMax;
  cv::Mat currMagNorm;
  cv::Mat magNormSum;
  cv::Mat magNormSqrdSum;
  cv::Mat magNormMax;
  ftag2::FreqTBPhaseStats phaseStatsMsg;


public:
  FTag2Testbench() :
      local_nh("~"),
      it(local_nh),
      dynCfgSyncReq(false),
      alive(false),
      dstID(0),
      dstFilename((char*) calloc(1000, sizeof(char))),
      latestProfTime(ros::Time::now()),
      waitKeyDelay(30),
      frameID(0) {
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
    params.phaseVarWeightR = 0;
    params.phaseVarWeightZ = 0;
    params.phaseVarWeightAngle = 0;
    params.phaseVarWeightFreq = 0;
    params.phaseVarWeightBias = 10*10;
    params.markerWidthM = 0.07;
    params.maxQuadsToScan = 10;
    params.imRotateDeg = 0;

    std::string cameraIntrinsicStr, cameraDistortionStr;
    local_nh.param("camera_intrinsic", cameraIntrinsicStr, cameraIntrinsicStr);
    local_nh.param("camera_distortion", cameraDistortionStr, cameraDistortionStr);
    if (cameraIntrinsicStr.size() > 0) {
      cameraIntrinsic = vc_math::str2mat(cameraIntrinsicStr, 3);
    }
    if (cameraDistortionStr.size() > 0) {
      cameraDistortion = vc_math::str2mat(cameraDistortionStr, 1);
    }

    // Process ground truth tag phases
    local_nh.param("target_tag_phases", targetTagPhasesStr, targetTagPhasesStr);
    if (targetTagPhasesStr.size() > 0) {
      targetTagPhases = vc_math::str2mat(targetTagPhasesStr, 6);
      currPhaseErrors = cv::Mat::zeros(6, targetTagPhases.cols, CV_64FC1);
      phaseErrorsSum = cv::Mat::zeros(1, targetTagPhases.cols, CV_64FC1);
      phaseErrorsSqrdSum = cv::Mat::zeros(1, targetTagPhases.cols, CV_64FC1);
      phaseErrorsMax = cv::Mat::zeros(1, targetTagPhases.cols, CV_64FC1);
      currMagNorm = cv::Mat::zeros(6, targetTagPhases.cols, CV_64FC1);
      magNormSum = cv::Mat::zeros(1, targetTagPhases.cols, CV_64FC1);
      magNormSqrdSum = cv::Mat::zeros(1, targetTagPhases.cols, CV_64FC1);
      magNormMax = cv::Mat::zeros(1, targetTagPhases.cols, CV_64FC1);

      phaseStatsMsg.frameID = -1;
      double* targetTagPhasesPtr = (double*) targetTagPhases.data;
      phaseStatsMsg.target_tag_phases =
          std::vector<double>(targetTagPhasesPtr,
          targetTagPhasesPtr + targetTagPhases.cols * targetTagPhases.rows);
      phaseStatsMsg.num_samples = 0;

      ROS_INFO_STREAM("TARGET TAG PHASES:" << endl << cv::format(targetTagPhases, "matlab"));
    } else {
      ROS_INFO_STREAM("NO TARGET TAG PHASES PARSED");
    }

    std::string source = "";
    local_nh.param("source", source, source);
    if (source.length() <= 0) {
      cam.open(-1);
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

    // Parse rosparams
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
    GET_PARAM(imRotateDeg);
    #undef GET_PARAM
    dynCfgSyncReq = true;
    phaseVariancePredictor.updateParams(params.phaseVarWeightR,
        params.phaseVarWeightZ, params.phaseVarWeightAngle,
        params.phaseVarWeightFreq, params.phaseVarWeightBias);
    local_nh.param("waitkey_delay", waitKeyDelay, waitKeyDelay);
    local_nh.param("save_img_dir", saveImgDir, saveImgDir);

    // Configure windows
    namedWindow("quad_1", CV_GUI_EXPANDED);
    namedWindow("quad_1_trimmed", CV_GUI_EXPANDED);
    namedWindow("segments", CV_GUI_EXPANDED);
    namedWindow("quads", CV_GUI_EXPANDED);
    namedWindow("tags", CV_GUI_EXPANDED);

    // Setup ROS stuff
    imagePub = it.advertise("frame_img", 1);
    tagPub = it.advertise("marker_img", 1);
    markerInfoPub = local_nh.advertise<ftag2::FreqTBMarkerInfo>("marker_info", 1);
    phaseStatsPub = local_nh.advertise<ftag2::FreqTBPhaseStats>("phase_stats", 1);

    // Finish initialization
    alive = true;
    spinThread = std::thread(&FTag2Testbench::spin, this);
  };


  ~FTag2Testbench() {
    alive = false;
    free(dstFilename);
    dstFilename = NULL;
    //spinThread.join(); // No need to double-call, since FTag2Testbench::join() is calling it
  };


  void join() {
    spinThread.join();
  };


  void configCallback(ftag2::FreqTestbenchConfig& config, uint32_t level) {
    if (!alive) return;
    params = config;
    phaseVariancePredictor.updateParams(params.phaseVarWeightR,
        params.phaseVarWeightZ, params.phaseVarWeightAngle,
        params.phaseVarWeightFreq, params.phaseVarWeightBias);
  };


  void spin() {
    cv::Point2d sourceCenter;
    cv::Mat rotMat;
    char c;

    try {
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
            ROS_DEBUG_STREAM("Updated params");
            dynCfgSyncReq = false;
          }
        }

        // Fetch image
        if (cam.isOpened()) {
          cam >> sourceImg;
          frameID++;
        }

        // Rotate image and convert to grayscale
        if (params.imRotateDeg != 0) {
          sourceCenter = cv::Point2d(sourceImg.cols/2.0, sourceImg.rows/2.0);
          rotMat = cv::getRotationMatrix2D(sourceCenter, params.imRotateDeg, 1.0);
          cv::warpAffine(sourceImg, sourceImgRot, rotMat, sourceImg.size());
          cv::cvtColor(sourceImgRot, grayImg, CV_RGB2GRAY);
        } else {
          sourceImgRot = sourceImg;
          cv::cvtColor(sourceImg, grayImg, CV_RGB2GRAY);
        }

        // Publish frame image
        cv_bridge::CvImage cvSourceImgRot(std_msgs::Header(),
            sensor_msgs::image_encodings::BGR8, sourceImgRot);
        cvSourceImgRot.header.frame_id = boost::lexical_cast<std::string>(frameID);
        imagePub.publish(cvSourceImgRot.toImageMsg());

        // 1. Detect line segments
        lineSegP.tic();
        std::vector<cv::Vec4i> segments = detectLineSegments(grayImg,
            params.sobelThreshHigh, params.sobelThreshLow, params.sobelBlurWidth,
            params.lineMinEdgelsCC, params.lineAngleMargin*degree,
            params.lineMinEdgelsSeg);
        lineSegP.toc();
        sourceImgRot.copyTo(overlaidImg);
        drawLineSegments(overlaidImg, segments);
        cv::imshow("segments", overlaidImg);

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

        // 3. Decode tags from quads
        int quadCount = 0;
        cv::Mat quadImg;
        FTag2Marker currTag;
        std::vector<FTag2Marker> tags;
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

        // Post-process largest detected tag
        if (tags.size() >= 1) {
          const FTag2Marker& tag = tags[0];

          // Show cropped tag image
          cv::imshow("quad_1_trimmed", tag.img);

          // Publish cropped tag image
          cv_bridge::CvImage cvCroppedTagImgRot(std_msgs::Header(),
              sensor_msgs::image_encodings::MONO8, tag.img);
          cvCroppedTagImgRot.header.frame_id = boost::lexical_cast<std::string>(frameID);
          tagPub.publish(cvCroppedTagImgRot.toImageMsg());

          // Publish detected tag info
          ftag2::FreqTBMarkerInfo markerInfoMsg;
          markerInfoMsg.frameID = frameID;
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
          markerInfoMsg.hasSignature = tag.payload.hasSignature;
          markerInfoMsg.hasValidXORs = tag.payload.hasValidXORs;
          markerInfoMsg.hasValidCRC = false;
          markerInfoMsg.payloadOct = tag.payload.payloadOct;
          markerInfoMsg.xorBin = tag.payload.xorBin;
          markerInfoMsg.signature = tag.payload.signature;
          markerInfoMsg.CRC12Expected = 0;
          markerInfoMsg.CRC12Decoded = 0;
          markerInfoPub.publish(markerInfoMsg);

          // Compute and publish stats
          if (!targetTagPhasesStr.empty()) {
            phaseStatsMsg.frameID = frameID;
            phaseStatsMsg.num_samples += 1;

            // Compute phase stats
            double* targetTagPhasesPtr = (double*) targetTagPhases.data;
            double* currTagPhasesPtr = (double*) tag.payload.phases.data;
            double* currPhaseErrorsPtr = (double*) currPhaseErrors.data;
            for (int i = 0; i < targetTagPhases.rows * targetTagPhases.cols; i++) {
              *currPhaseErrorsPtr = vc_math::angularDist(*currTagPhasesPtr, *targetTagPhasesPtr, 360.0);
              targetTagPhasesPtr++;
              currTagPhasesPtr++;
              currPhaseErrorsPtr++;
            }

            cv::Mat currPhaseErrorsSum, currPhaseErrorsMax;
            cv::Mat currPhaseErrorsSqrd, currPhaseErrorsSqrdSum;
            cv::Mat phaseErrorsAvg, phaseErrorsAvgSqrd, phaseErrorsVar, phaseErrorsStd;
            cv::reduce(currPhaseErrors, currPhaseErrorsSum, 0, CV_REDUCE_SUM);
            cv::reduce(currPhaseErrors, currPhaseErrorsMax, 0, CV_REDUCE_MAX);
            cv::pow(currPhaseErrors, 2, currPhaseErrorsSqrd);
            cv::reduce(currPhaseErrorsSqrd, currPhaseErrorsSqrdSum, 0, CV_REDUCE_SUM);
            phaseErrorsSum = phaseErrorsSum + currPhaseErrorsSum;
            phaseErrorsMax = cv::max(phaseErrorsMax, currPhaseErrorsMax);
            phaseErrorsSqrdSum = phaseErrorsSqrdSum + currPhaseErrorsSqrdSum;
            phaseErrorsAvg = phaseErrorsSum / (phaseStatsMsg.num_samples * 6);
            cv::pow(phaseErrorsAvg, 2, phaseErrorsAvgSqrd);
            phaseErrorsVar = phaseErrorsSqrdSum / (phaseStatsMsg.num_samples * 6) - phaseErrorsAvgSqrd;
            cv::sqrt(phaseErrorsVar, phaseErrorsStd);

            // Compute mags stats
            double mag_norm_divisor = 0;
            tag.payload.mags.copyTo(currMagNorm);
            double* currMagNormPtr = (double*) currMagNorm.data;
            for (int r = 0; r < currMagNorm.rows; r++) {
              mag_norm_divisor = *currMagNormPtr;
              *currMagNormPtr = 1.0;
              currMagNormPtr++;
              for (int c = 1; c < currMagNorm.cols; c++) {
                *currMagNormPtr /= mag_norm_divisor;
                currMagNormPtr++;
              }
            }

            cv::Mat currMagNormSum, currMagNormMax;
            cv::Mat currMagNormSqrd, currMagNormSqrdSum;
            cv::Mat magNormAvg, magNormAvgSqrd, magNormVar, magNormStd;
            cv::reduce(currMagNorm, currMagNormSum, 0, CV_REDUCE_SUM);
            cv::reduce(currMagNorm, currMagNormMax, 0, CV_REDUCE_MAX);
            cv::pow(currMagNorm, 2, currMagNormSqrd);
            cv::reduce(currMagNormSqrd, currMagNormSqrdSum, 0, CV_REDUCE_SUM);
            magNormSum = magNormSum + currMagNormSum;
            magNormMax = cv::max(magNormMax, currMagNormMax);
            magNormSqrdSum = magNormSqrdSum + currMagNormSqrdSum;
            magNormAvg = magNormSum / (phaseStatsMsg.num_samples * 6);
            cv::pow(magNormAvg, 2, magNormAvgSqrd);
            magNormVar = magNormSqrdSum / (phaseStatsMsg.num_samples * 6) - magNormAvgSqrd;
            cv::sqrt(magNormVar, magNormStd);

            // Publish stats
            double* phaseErrorsAvgPtr = (double*) phaseErrorsAvg.data;
            double* phaseErrorsStdPtr = (double*) phaseErrorsStd.data;
            double* phaseErrorsMaxPtr = (double*) phaseErrorsMax.data;
            double* magNormAvgPtr = (double*) magNormAvg.data;
            double* magNormStdPtr = (double*) magNormStd.data;
            double* magNormMaxPtr = (double*) magNormMax.data;
            phaseStatsMsg.phase_errors_avg = std::vector<double>(phaseErrorsAvgPtr, phaseErrorsAvgPtr + phaseErrorsAvg.cols * phaseErrorsAvg.rows);
            phaseStatsMsg.phase_errors_std = std::vector<double>(phaseErrorsStdPtr, phaseErrorsStdPtr + phaseErrorsStd.cols * phaseErrorsStd.rows);
            phaseStatsMsg.phase_errors_max = std::vector<double>(phaseErrorsMaxPtr, phaseErrorsMaxPtr + phaseErrorsMax.cols * phaseErrorsMax.rows);
            phaseStatsMsg.mag_spectra_avg = std::vector<double>(magNormAvgPtr, magNormAvgPtr + magNormAvg.cols * magNormAvg.rows);
            phaseStatsMsg.mag_spectra_std = std::vector<double>(magNormStdPtr, magNormStdPtr + magNormStd.cols * magNormStd.rows);
            phaseStatsMsg.mag_spectra_max = std::vector<double>(magNormMaxPtr, magNormMaxPtr + magNormMax.cols * magNormMax.rows);

            phaseStatsPub.publish(phaseStatsMsg);
          } else {
            if (tag.payload.hasValidXORs) {
              cout << "=> RECOG  : ";
            } else {
              cout << "x> BAD XOR: ";
            }
            cout << tag.payload.payloadOct << "; XOR: " << tag.payload.xorBin << "; Rot=" << tag.imgRotDir << "'";
            if (tag.payload.hasValidXORs) {
              cout << "\tID: " << tag.payload.payloadBin;
            }
            cout << endl;
          }
        }

        // Display detected quads and tags
        overlaidImg = sourceImg.clone();
        for (const Quad& quad: quads) {
          drawQuad(overlaidImg, quad.corners);
        }
        cv::imshow("quads", overlaidImg);
        overlaidImg = sourceImg.clone();
        for (const FTag2Marker& tag: tags) {
          drawTag(overlaidImg, tag.corners);
        }
        cv::imshow("tags", overlaidImg);

  #ifdef SAVE_IMAGES_FROM
        if (saveImgDir.size() > 0) {
          sprintf(dstFilename, "%s/img%05d.jpg", saveImgDir.c_str(), dstID++);
        } else {
          sprintf(dstFilename, "img%05d.jpg", dstID++);
        }
        imwrite(dstFilename, SAVE_IMAGES_FROM);
        ROS_INFO_STREAM("Wrote to " << dstFilename);
  #endif

  #ifdef ENABLE_PROFILER
        durationProf.toc();

        ros::Time currTime = ros::Time::now();
        ros::Duration td = currTime - latestProfTime;
        if (td.toSec() > 1.0) {

          cout << "detectLineSegments: " << lineSegP.getStatsString() << endl;
          cout << "detectQuads: " << quadP.getStatsString() << endl;
          cout << "extractTags: " << quadExtractorP.getStatsString() << endl;
          cout << "decodeTag: " << decoderP.getStatsString() << endl;

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
        } else if ((c & 0x0FF) == 'r' || (c & 0x0FF) == 'R') {
          if (!targetTagPhasesStr.empty()) {
            phaseStatsMsg.num_samples = 0;
            phaseErrorsSum.setTo(cv::Scalar(0));
            phaseErrorsSqrdSum.setTo(cv::Scalar(0));
            phaseErrorsMax.setTo(cv::Scalar(0));
            magNormSum.setTo(cv::Scalar(0));
            magNormSqrdSum.setTo(cv::Scalar(0));
            magNormMax.setTo(cv::Scalar(0));
          }
        }
      }
    } catch (const cv::Exception& err) {
      ROS_ERROR_STREAM("Spin thread halted due to CV Exception: " << err.what());
    } catch (const std::string& err) {
      ROS_ERROR_STREAM("Spin thread halted due to code error: " << err);
    }
  };
};


int main(int argc, char** argv) {
  ros::init(argc, argv, "freq_testbench");

  try {
    FTag2Testbench testbench;
    testbench.join();
  } catch (const cv::Exception& err) {
    cout << "CV EXCEPTION: " << err.what() << endl;
  } catch (const std::string& err) {
    cout << "ERROR: " << err << endl;
  } catch (std::system_error& err) {
    cout << "SYSTEM ERROR: " << err.what() << endl;
  }

  return EXIT_SUCCESS;
};
