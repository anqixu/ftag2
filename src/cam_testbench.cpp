#include "detector/FTag2Detector.hpp"
#include "decoder/FTag2Decoder.hpp"
#include "tracker/ParticleFilter.hpp"
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
#include <ftag2/CamTestbenchConfig.h>

#include "common/Profiler.hpp"
#include "common/VectorAndCircularMath.hpp"
#include "common/GNUPlot.hpp"

#include <visualization_msgs/Marker.h>
#include "tf/LinearMath/Transform.h"
#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_broadcaster.h>

#include <fstream>
#include "yaml-cpp/yaml.h"

#include <ompl/base/spaces/SO3StateSpace.h>
#include <ompl/base/State.h>
#include <ompl/base/ScopedState.h>


//#define SAVE_IMAGES_FROM sourceImgRot
//#define ENABLE_PROFILER


using namespace std;
using namespace std::chrono;
using namespace cv;
using namespace vc_math;
using namespace ompl::base;


typedef dynamic_reconfigure::Server<ftag2::CamTestbenchConfig> ReconfigureServer;

std::vector<cv::Point2f> Generate2DPoints( Quad quad )
{
	std::vector<cv::Point2f> points;

	for ( unsigned int i=0 ; i<4; i++ )
	{
		points.push_back(quad.corners[i]);
//		cv::circle(testImg, quad.corners[i], 5, cv::Scalar(255), 2, 8, 0);
	}

	for(unsigned int i = 0; i < points.size(); ++i)
	{
		std::cout << points[i] << std::endl;
	}

	return points;
}

std::vector<cv::Point3f> Generate3DPoints()
{
	std::vector<cv::Point3f> points;
	float x,y,z;

	x=-47.5;y=-47.5;z=0;
	points.push_back(cv::Point3f(x,y,z));

	x=47.5;y=-47.5;z=0;
	points.push_back(cv::Point3f(x,y,z));

	x=47.5;y=47.5;z=0;
	points.push_back(cv::Point3f(x,y,z));

	x=-47.5;y=47.5;z=0;
	points.push_back(cv::Point3f(x,y,z));

	for(unsigned int i = 0; i < points.size(); ++i)
    {
		std::cout << points[i] << std::endl;
    }

	return points;
}


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
	CvMat *intrinsic;
	CvMat *distortion;
    cv::Mat distCoeffs;
    cv::Mat cameraMatrix;
    ros::NodeHandle n;
    ros::Publisher marker_pub;
    uint32_t shape;
    //visualization_msgs::Marker marker;
    geometry_msgs::PoseStamped marker;
    YAML::Emitter out;
    int frameNo;
    VideoWriter outVideo;
    bool recording;
    bool tracking;
    ParticleFilter PF;
    std::vector <FTag2Marker> detections;

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
    #undef GET_PARAM
    dynCfgSyncReq = true;
    local_nh.param("waitkey_delay", waitKeyDelay, waitKeyDelay);
    local_nh.param("save_img_dir", saveImgDir, saveImgDir);

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

    //namedWindow("source", CV_GUI_EXPANDED);
    //namedWindow("debug", CV_GUI_EXPANDED);
    namedWindow("edgels", CV_GUI_EXPANDED);
    //namedWindow("accum", CV_GUI_EXPANDED);
    //namedWindow("lines", CV_GUI_EXPANDED);
    namedWindow("segments", CV_GUI_EXPANDED);
    namedWindow("quad_1", CV_GUI_EXPANDED);
    namedWindow("quad_1_trimmed", CV_GUI_EXPANDED);
    namedWindow("quads", CV_GUI_EXPANDED);
/*
    std::string camMatrixFname = "";
    local_nh.param("camMatrix", camMatrixFname, camMatrixFname);
    std::string distCoefFname = "";
    local_nh.param("distCoef", distCoefFname, distCoefFname);
*/
    intrinsic = (CvMat*)cvLoad("/home/dacocp/Dropbox/catkin_ws/Intrinsics.xml");
    distortion = (CvMat*)cvLoad("/home/dacocp/Dropbox/catkin_ws/Distortion.xml");
    distCoeffs = distortion;
    cameraMatrix = intrinsic;
    std::cout << "Camera Intrinsic Matrix: " << cameraMatrix << std::endl;

    //marker_pub = n.advertise<visualization_msgs::Marker>("visualization_marker", 1);
    marker_pub = n.advertise<geometry_msgs::PoseStamped>("PoseStamped", 1);
    shape = visualization_msgs::Marker::ARROW;
    marker.header.frame_id = "/my_frame";

    out << YAML::BeginSeq;
    frameNo = 0;

    outVideo.open ( "/home/dacocp/Dropbox/catkin_ws/outputVideo.avi", CV_FOURCC('D','I','V','X'), 21, cv::Size ( 300,200), true );
    recording = false;
    tracking = false;
    detections = std::vector<FTag2Marker>();

    ompl::base::StateSpacePtr space(new ompl::base::SO3StateSpace());
    ompl::base::ScopedState<ompl::base::SO3StateSpace> stateM(space);
    ompl::base::ScopedState<ompl::base::SO3StateSpace> stateN(space);
    ompl::base::ScopedState<> backup = stateM;
    ompl::base::State *abstractState = space->allocState();
    std::cout << "1: " << stateM << std::endl;
    std::cout << stateN << std::endl;
    std::cout << abstractState << std::endl;

    stateM = abstractState;
    std::cout << "2: " << stateM << std::endl;
    std::cout << stateN << std::endl;
    std::cout << abstractState << std::endl;

    stateM->as<ompl::base::SO3StateSpace::StateType>()->x = 1.0;
    stateM->as<ompl::base::SO3StateSpace::StateType>()->y = 0.0;
    stateM->as<ompl::base::SO3StateSpace::StateType>()->z = 1.0;
    stateM->as<ompl::base::SO3StateSpace::StateType>()->w = 0.0;
    stateN.random();

    std::cout << "3: " << stateM << std::endl;
    std::cout << stateN << std::endl;
    std::cout << abstractState << std::endl;

    ompl::base::SO3StateSampler SO3ss(space->as<ompl::base::SO3StateSpace>());

    SO3ss.sampleGaussian(stateN->as<ompl::base::SO3StateSpace::StateType>(), stateM->as<ompl::base::SO3StateSpace::StateType>(), 0.0001);

    std::cout << "4: " << stateM << std::endl;
    std::cout << stateN << std::endl;
    std::cout << abstractState << std::endl;

//    stateN = abstractState;
    std::cout << "5: " << stateM << std::endl;
    std::cout << stateN << std::endl;
    std::cout << abstractState << std::endl;

    alive = true;

    spinThread = std::thread(&FTag2Testbench::spin, this);
  };


  ~FTag2Testbench() {
    alive = false;
    free(dstFilename);
    dstFilename = NULL;
    out << YAML::EndSeq;
//    std::ofstream fout("/home/dacocp/Dropbox/catkin_ws/trajectory.yaml");
//    std::cout << "Here's the output YAML:\n" << out.c_str();
//    fout << out.c_str();
//    fout.close();
    outVideo.release();
    //spinThread.join(); // No need to double-call, since FTag2Testbench::join() is calling it
  };


  void join() {
    spinThread.join();
  };


  void configCallback(ftag2::CamTestbenchConfig& config, uint32_t level) {
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

        overlaidImg = sourceImg.clone();error
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
        quadP.toc();
        cv::Mat quadsImg = sourceImgRot.clone();
        //drawLineSegments(quadsImg, segments);
        drawQuads(quadsImg, quads);
        cv::imshow("quads", quadsImg);


        if (!quads.empty()) {
          std::list<Quad>::iterator quadIt, largestQuadIt;
          double largestQuadArea = -1;
          for (quadIt = quads.begin(); quadIt != quads.end(); quadIt++) {
            if (quadIt->area > largestQuadArea) {
              largestQuadArea = quadIt->area;
              largestQuadIt = quadIt;
            }
          }
          cv::Mat tagImg = extractQuadImg(sourceImgRot, *largestQuadIt, params.quadMinWidth);
          if (!tagImg.empty()) {
            cv::Mat croppedTagImg = trimFTag2Quad(tagImg, params.quadMaxStripAvgDiff);
            croppedTagImg = cropFTag2Border(croppedTagImg);

            FTag2 tag = FTag2Decoder::decodeTag(croppedTagImg);

            /*
            // Plot spatial signal
            std::vector<double> pts;
            for (int i = 0; i < tag.horzRays.cols; i++) { pts.push_back(tag.horzRays.at<double>(0, i)); }
            gp::bar(pts, 0, "First Ray");

            // Plot magnitude spectrum
            std::vector<cv::Point2d> spec;
            for (int i = 1; i < std::min(magSpec.cols, 9); i++) { spec.push_back(cv::Point2d(i, magSpec.at<double>(0, i))); }
            gp::plot(spec, 1, "Magnitude Spectrum");

            // Plot phase spectrum
            spec.clear();
            for (int i = 1; i < std::min(phaseSpec.cols, 9); i++) { spec.push_back(cv::Point2d(i, vc_math::wrapAngle(phaseSpec.at<double>(0, i), 360.0))); }
            gp::plot(spec, 2, "Phase Spectrum");
            */

            frameNo++;
            detections = std::vector<FTag2Marker>();
            if (tag.hasSignature) {
              cv::Mat tagImgRot, croppedTagImgRot;
              BaseCV::rotate90(tagImg, tagImgRot, tag.imgRotDir/90);
              BaseCV::rotate90(croppedTagImg, croppedTagImgRot, tag.imgRotDir/90);

              std::cout << "=====> RECOGNIZED TAG: " << tag.ID << " (@ rot=" << tag.imgRotDir << ")" << std::endl;
              //std::cout << "psk = ..." << std::endl << cv::format(tag.PSK, "matlab") << std::endl << std::endl;
              cv::imshow("quad_1", tagImgRot);
              cv::imshow("quad_1_trimmed", croppedTagImgRot);

              std::vector<cv::Point3f> objectPoints = Generate3DPoints();
              std::vector<cv::Point2f> imagePoints = quads.front().corners;

              for (int k = 0; k<tag.imgRotDir/90; k++)
              {
              	cv::Point2f temp = imagePoints[0];
                imagePoints.erase(imagePoints.begin());
                imagePoints.push_back(temp);
              }

              for(unsigned int k = 0; k < quads.front().corners.size(); ++k)
              {
            	if(k==0)
            		cv::circle(quadsImg, imagePoints[k], 5, cv::Scalar(0, 0, 255), 3, 8, 0);
            	else if (k==1)
            		cv::circle(quadsImg, imagePoints[k], 5, cv::Scalar(0, 255, 0), 3, 8, 0);
            	else if (k==2)
            		cv::circle(quadsImg, imagePoints[k], 5, cv::Scalar(255, 0, 0), 3, 8, 0);
            	else
            		cv::circle(quadsImg, imagePoints[k], 5, cv::Scalar(255, 255, 255), 3, 8, 0);
              }
              cv::imshow("quads", quadsImg);

              cv::Mat rvec(3,1,cv::DataType<double>::type);
              cv::Mat tvec(3,1,cv::DataType<double>::type);

              cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);

              std::cout << "rvec: (" << rvec.at<double>(0,0) << ", " << rvec.at<double>(1,0) << ", " << rvec.at<double>(2,0) << ")" << std::endl;
              std::cout << "tvec: (" << tvec.at<double>(0,0) << ", " << tvec.at<double>(1,0) << ", " << tvec.at<double>(2,0) << ")" << std::endl;

              marker.header.stamp = ros::Time::now();

              marker.pose.position.x = tvec.at<double>(0)/100.0;
              marker.pose.position.y = tvec.at<double>(1)/100.0;
              marker.pose.position.z = tvec.at<double>(2)/100.0;
              tf::Quaternion rMat;
              rMat.setRPY(rvec.at<double>(0,0), rvec.at<double>(1,0),rvec.at<double>(2,0));
              marker.pose.orientation.x = rMat.getX();
              marker.pose.orientation.y = rMat.getY();
              marker.pose.orientation.z = rMat.getZ();
              marker.pose.orientation.w = rMat.getW();

              // Publish the marker
              marker_pub.publish(marker);

              static tf::TransformBroadcaster br;
              tf::Transform transform;
              transform.setOrigin( tf::Vector3(tvec.at<double>(0)/100.0, tvec.at<double>(1)/100.0, tvec.at<double>(2)/100.0) );
              transform.setRotation( rMat );
              br.sendTransform( tf::StampedTransform( transform, ros::Time::now(), "camera", "ftag" ) );

              std::ostringstream oss;
              oss << frameNo;
              out << YAML::BeginMap;
//              out << YAML::Key << ros::Time::now().nsec;
              out << YAML::Key << oss.str();
                  out << YAML::Value;
              	  out << YAML::BeginMap;
              	  out << YAML::Key << "Frame No.";
              	  out << YAML::Value << frameNo;
              	  out << YAML::Key << "Position";
              	  out << YAML::Value
              			  << YAML::BeginSeq
              			  	  << marker.pose.position.x << marker.pose.position.y << marker.pose.position.z
              			  << YAML::EndSeq;
              	  out << YAML::Key << "Orientation";
              	  out << YAML::Value
              			  << YAML::BeginSeq
              			  	  << marker.pose.orientation.x << marker.pose.orientation.y
              			  	  << marker.pose.orientation.z << marker.pose.orientation.w
              			  << YAML::EndSeq ;
              	  out << YAML::EndMap;
              out << YAML::EndMap;
//              recording = true;
              detections = std::vector<FTag2Marker>(1);
              detections[0].pose_x = tvec.at<double>(0)/100.0;
              detections[0].pose_y = tvec.at<double>(1)/100.0;
              detections[0].pose_z = tvec.at<double>(2)/100.0;
              detections[0].orientation_x = rMat.getX();;
              detections[0].orientation_y = rMat.getY();;
              detections[0].orientation_z = rMat.getZ();;
              detections[0].orientation_w = rMat.getW();;
              if ( tracking == false )
              {
              	  tracking = true;
              	  PF = ParticleFilter(150, 10, detections);
              	  cv::waitKey();
              }
            } else {
              cv::imshow("quad_1", tagImg);
              cv::imshow("quad_1_trimmed", croppedTagImg);
              std::cout << std::endl << "==========" << std::endl;
              std::cout << "hmags = ..." << std::endl << cv::format(tag.horzMags, "matlab") << std::endl << std::endl;
              std::cout << "vmags = ..." << std::endl << cv::format(tag.vertMags, "matlab") << std::endl << std::endl;
              std::cout << "hpsk = ..." << std::endl << cv::format(tag.horzPhases, "matlab") << std::endl << std::endl;
              std::cout << "vpsk = ..." << std::endl << cv::format(tag.vertPhases, "matlab") << std::endl << std::endl;
            }
          }
        }

        if (recording == true)
        {
        	cv::Mat resizedImg = quadsImg.clone();
        	cv::resize(quadsImg,resizedImg,cv::Size(300,200));
        	outVideo.write(resizedImg);
        }

        if (tracking == true)
        {
        	PF.motionUpdate();
        	PF.measurementUpdate(detections);
//        	PF.normalizeWeights();
        	PF.computeMeanPose();
        	PF.resample();
//        	if (frameNo%10 == 0)
//        	PF.displayParticles();
//        	cv::waitKey();
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
        	//out << YAML::EndSeq;
        	//std::ofstream fout("trajectory.yaml");
        	//fout << out.c_str();
        	//std::cout << "Here's the output YAML:\n" << out.c_str();
//        	fout << emitter.c_str();
        }
      }
    } catch (const cv::Exception& err) {
      std::cerr << "CV Exception: " << err.what() << std::endl;
    }
  };


protected:
  std::thread spinThread;

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

  Profiler lineSegP, quadP, durationProf, rateProf;
  ros::Time latestProfTime;

  int waitKeyDelay;
  std::string saveImgDir;
};


int main(int argc, char** argv) {
  ros::init(argc, argv, "cam_testbench");

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
