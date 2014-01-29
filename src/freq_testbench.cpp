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


bool compareArea(const Quad& first, const Quad& second) {
  return first.area > second.area;
};


cv::Mat str2mat(const std::string& s, unsigned int rows) {
  std::string input = s;
  auto it = std::remove_if(std::begin(input), std::end(input),
      [](char c) { return (c == ',' || c == ';' || c == ':'); });
  input.erase(it, std::end(input));

  cv::Mat mat(0, 0, CV_64FC1);
  std::istringstream iss(input);
  double currNum;
  while (!iss.eof()) {
    iss >> currNum;
    mat.push_back(currNum);
  }
  return mat.reshape(1, rows);
};


class FTag2Testbench {
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
    // Low-value params tuned for marginal acceptable results on synthetic images
    params.sobelThreshHigh = 100;
    params.sobelThreshLow = 30;
    params.sobelBlurWidth = 3;
    params.lineAngleMargin = 20.0; // *degree
    params.lineMinEdgelsCC = 50;
    params.lineMinEdgelsSeg = 15;
    params.quadMinWidth = 10.0;
    params.quadMinAngleIntercept = 30.0;
    params.quadMinEndptDist = 4.0;
    params.quadMaxStripAvgDiff = 15.0;
    params.imRotateDeg = 0;

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
    GET_PARAM(imRotateDeg);
    GET_PARAM(maxQuadsToScan);
    #undef GET_PARAM
    dynCfgSyncReq = true;
    local_nh.param("waitkey_delay", waitKeyDelay, waitKeyDelay);
    local_nh.param("save_img_dir", saveImgDir, saveImgDir);

    // Process ground truth tag phases
    local_nh.param("target_tag_phases", targetTagPhasesStr, targetTagPhasesStr);
    if (targetTagPhasesStr.size() > 0) {
      targetTagPhases = str2mat(targetTagPhasesStr, 6);
      currPhaseErrors = cv::Mat::zeros(6, targetTagPhases.cols, CV_64FC1);
      phaseErrorsSum = cv::Mat::zeros(1, targetTagPhases.cols, CV_64FC1);
      phaseErrorsSqrdSum = cv::Mat::zeros(1, targetTagPhases.cols, CV_64FC1);
      phaseErrorsMax = cv::Mat::zeros(1, targetTagPhases.cols, CV_64FC1);

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

    // Setup ROS stuff
    imagePub = it.advertise("frame_img", 1);
    tagPub = it.advertise("marker_img", 1);
    markerInfoPub = local_nh.advertise<ftag2::FreqTBMarkerInfo>("marker_info", 1);
    phaseStatsPub = local_nh.advertise<ftag2::FreqTBPhaseStats>("phase_stats", 1);

    //namedWindow("source", CV_GUI_EXPANDED);
    //namedWindow("debug", CV_GUI_EXPANDED);
    namedWindow("edgels", CV_GUI_EXPANDED);
    //namedWindow("accum", CV_GUI_EXPANDED);
    //namedWindow("lines", CV_GUI_EXPANDED);
    namedWindow("segments", CV_GUI_EXPANDED);
    namedWindow("quad_1", CV_GUI_EXPANDED);
    namedWindow("quad_1_trimmed", CV_GUI_EXPANDED);
    namedWindow("quads", CV_GUI_EXPANDED);

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
            ROS_INFO_STREAM("Updated params");
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

        // Extract line segments using optimized segment detector using
        // angle-bounded connected edgel components
        lineSegP.tic();
        std::vector<cv::Vec4i> segments = detectLineSegments(grayImg,
            params.sobelThreshHigh, params.sobelThreshLow, params.sobelBlurWidth,
            params.lineMinEdgelsCC, params.lineAngleMargin*degree,
            params.lineMinEdgelsSeg);
        lineSegP.toc();
        sourceImgRot.copyTo(overlaidImg);
        drawLineSegments(overlaidImg, segments);
        cv::imshow("segments", overlaidImg);

        // Detect quads
        quadP.tic();
        std::list<Quad> quads = detectQuads(segments,
            params.quadMinAngleIntercept*degree,
            params.quadMinEndptDist);
        quads.sort(compareArea);
        quadP.toc();


        if (quads.empty()) {
          ROS_WARN_STREAM("NO QUADS IN FRAME");
        }
        if (false) {
          cout << "Quads: " << quads.size() << endl;
          for (const Quad& q: quads) {
            cout << "- " << q.area << endl;
          }
        }

        bool foundTag = false;
        if (!quads.empty()) {
          std::list<Quad>::iterator currQuad = quads.begin();
          for (int quadI = 0; quadI < std::min(params.maxQuadsToScan, (int) quads.size()); quadI++, currQuad++) {
            cv::Mat tagImg = extractQuadImg(sourceImgRot, *currQuad, params.quadMinWidth);
            if (!tagImg.empty()) {
              cv::Mat croppedTagImg = trimFTag2Quad(tagImg, params.quadMaxStripAvgDiff);
              croppedTagImg = cropFTag2Border(croppedTagImg);

              decoderP.tic();
              FTag2Marker6S5F3B tag(croppedTagImg);
              decoderP.toc();

              if (tag.hasSignature) {
                // Show quad and tag images
                cv::Mat quadsImg = sourceImgRot.clone();
                drawQuad(quadsImg, *currQuad);
                cv::imshow("quads", quadsImg);
                cv::imshow("quad_1", tagImg);

                // Show and publish cropped tag image
                cv::Mat croppedTagImgRot;
                BaseCV::rotate90(croppedTagImg, croppedTagImgRot, tag.imgRotDir/90);
                cv::imshow("quad_1_trimmed", croppedTagImgRot);
                cv_bridge::CvImage cvCroppedTagImgRot(std_msgs::Header(),
                    sensor_msgs::image_encodings::MONO8, croppedTagImgRot);
                cvCroppedTagImgRot.header.frame_id = boost::lexical_cast<std::string>(frameID);
                tagPub.publish(cvCroppedTagImgRot.toImageMsg());

                // Publish detected tag info
                ftag2::FreqTBMarkerInfo markerInfoMsg;
                markerInfoMsg.frameID = frameID;
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

                // Compute and publish phase error stats
                if (!targetTagPhasesStr.empty()) {
                  phaseStatsMsg.frameID = frameID;
                  phaseStatsMsg.num_samples += 1;

                  double* targetTagPhasesPtr = (double*) targetTagPhases.data;
                  double* currTagPhasesPtr = (double*) tag.phases.data;
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

                  double* phaseErrorsAvgPtr = (double*) phaseErrorsAvg.data;
                  double* phaseErrorsStdPtr = (double*) phaseErrorsStd.data;
                  double* phaseErrorsMaxPtr = (double*) phaseErrorsMax.data;
                  phaseStatsMsg.phase_errors_avg = std::vector<double>(phaseErrorsAvgPtr, phaseErrorsAvgPtr + phaseErrorsAvg.cols * phaseErrorsAvg.rows);
                  phaseStatsMsg.phase_errors_std = std::vector<double>(phaseErrorsStdPtr, phaseErrorsStdPtr + phaseErrorsStd.cols * phaseErrorsStd.rows);
                  phaseStatsMsg.phase_errors_max = std::vector<double>(phaseErrorsMaxPtr, phaseErrorsMaxPtr + phaseErrorsMax.cols * phaseErrorsMax.rows);

                  phaseStatsPub.publish(phaseStatsMsg);
                } else {
                  if (tag.hasValidXORs && tag.hasValidCRC) {
                    cout << "=> RECOG  : ";
                  } else if (tag.hasValidXORs) {
                    cout << "x> BAD CRC: ";
                  } else {
                    cout << "x> BAD XOR: ";
                  }
                  cout << tag.payloadOct << "; XOR: " << tag.xorBin << "; Rot=" << tag.imgRotDir << "'";
                  if (tag.hasValidXORs && tag.hasValidCRC) {
                    cout << "\tID: " << tag.payload.to_ullong();
                  }
                  cout << endl;
                }

                foundTag = true;
                break; // stop scanning for more quads
              }
            }
          }
        }
        if (!foundTag) {
          cv::imshow("quads", sourceImgRot);
        }



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
          }
        }
      }
    } catch (const cv::Exception& err) {
      std::cerr << "CV Exception: " << err.what() << std::endl;
    }
  };


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

  Profiler lineSegP, quadP, decoderP, durationProf, rateProf;
  ros::Time latestProfTime;

  int waitKeyDelay;
  std::string saveImgDir;

  std::string targetTagPhasesStr;
  cv::Mat targetTagPhases;

  int frameID;
  cv::Mat currPhaseErrors;
  cv::Mat phaseErrorsSum;
  cv::Mat phaseErrorsSqrdSum;
  cv::Mat phaseErrorsMax;
  ftag2::FreqTBPhaseStats phaseStatsMsg;
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
