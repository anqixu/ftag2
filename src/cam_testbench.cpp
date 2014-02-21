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

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <image_geometry/pinhole_camera_model.h>

//#define SAVE_IMAGES_FROM sourceImgRot
//#define ENABLE_PROFILER


using namespace std;
using namespace std::chrono;
using namespace cv;
using namespace vc_math;
using namespace ompl::base;

typedef dynamic_reconfigure::Server<ftag2::CamTestbenchConfig> ReconfigureServer;

bool compareArea(const Quad& first, const Quad& second) {
  return first.area > second.area;
};

void imageCallback(const sensor_msgs::ImageConstPtr& original_image)
{
    //Convert from the ROS image message to a CvImage suitable for working with OpenCV for processing
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        //Always copy, returning a mutable CvImage
        //OpenCV expects color images to use BGR channel order.
        cv_ptr = cv_bridge::toCvCopy(original_image, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
        //if there is an error during conversion, display it
        ROS_ERROR("tutorialROSOpenCV::main.cpp::cv_bridge exception: %s", e.what());
        return;
    }

    //Display the image using OpenCV
    cv::imshow("WINDOW", cv_ptr->image);
    //Add some delay in miliseconds. The function only works if there is at least one HighGUI window created and the window is active. If there are several HighGUI windows, any of them can be active.
    cv::waitKey(3);
};

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
    int currentNumberOfParticles;
    double current_position_std;
    double current_orientation_std;
    double current_position_noise_std;
    double current_orientation_noise_std;

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
    params.quadMinWidth = 15.0;
    params.quadMinAngleIntercept = 30.0;
    params.quadMinEndptDist = 4.0;
    params.quadMaxStripAvgDiff = 15.0;
    params.imRotateDeg = 0;
    params.numberOfParticles = 50;
    params.position_std = 0.5;
    params.orientation_std = 0.5;
    params.position_noise_std = 0.3;
    params.orientation_noise_std = 0.2;
    params.velocity_noise_std = 0.15;

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
    GET_PARAM(quadMinEndptDist);
    GET_PARAM(quadMaxStripAvgDiff);
    GET_PARAM(imRotateDeg);
    GET_PARAM(maxQuadsToScan);
    GET_PARAM(numberOfParticles);
    GET_PARAM(position_std);
    GET_PARAM(orientation_std);
    GET_PARAM(position_noise_std);
    GET_PARAM(orientation_noise_std);
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

    //namedWindow("source", CV_GUI_EXPANDED);
    //namedWindow("debug", CV_GUI_EXPANDED);
    namedWindow("edgels", CV_GUI_EXPANDED);
    //namedWindow("accum", CV_GUI_EXPANDED);
    //namedWindow("lines", CV_GUI_EXPANDED);
    namedWindow("segments", CV_GUI_EXPANDED);
    namedWindow("quad_1", CV_GUI_EXPANDED);
    namedWindow("quad_1_trimmed", CV_GUI_EXPANDED);
    namedWindow("quads", CV_GUI_EXPANDED);


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
            ROS_DEBUG_STREAM("Updated params");
            dynCfgSyncReq = false;
          }
        }

        // Fetch image
        if (cam.isOpened()) {
          cam >> sourceImg;
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

        cv::Mat quadsImg = sourceImgRot.clone();
        bool foundTag = false;
        if (!quads.empty()) {
          std::list<Quad>::iterator currQuad = quads.begin();
          for (int quadI = 0; quadI < std::min(params.maxQuadsToScan, (int) quads.size()); quadI++, currQuad++) {
            cv::Mat tagImg = extractQuadImg(sourceImgRot, *currQuad, params.quadMinWidth);
            if (!tagImg.empty()) {
              cv::Mat croppedTagImg = trimFTag2Quad(tagImg, params.quadMaxStripAvgDiff);
              croppedTagImg = cropFTag2Border(croppedTagImg);
              if (croppedTagImg.rows < params.quadMinWidth || croppedTagImg.cols < params.quadMinWidth) {
                continue;
              }

              decoderP.tic();
              FTag2Marker6S5F3B tag(croppedTagImg);
              decoderP.toc();

              frameNo++;

              detections = std::vector<FTag2Marker>();
              if (tag.hasSignature) {
                cv::Mat tagImgRot, croppedTagImgRot;
                BaseCV::rotate90(tagImg, tagImgRot, tag.imgRotDir/90);
                BaseCV::rotate90(croppedTagImg, croppedTagImgRot, tag.imgRotDir/90);

                std::cout << "=====> RECOGNIZED TAG: " << " (@ rot=" << tag.imgRotDir << ")" << std::endl;
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

                drawQuad(quadsImg, *currQuad);
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
                detections[0].position_x = tvec.at<double>(0)/100.0;
                detections[0].position_y = tvec.at<double>(1)/100.0;
                detections[0].position_z = tvec.at<double>(2)/100.0;
                detections[0].orientation_x = rMat.getX();
                detections[0].orientation_y = rMat.getY();
                detections[0].orientation_z = rMat.getZ();
                detections[0].orientation_w = rMat.getW();
                if ( tracking == false )
                {
                    tracking = true;
                    PF = ParticleFilter(params.numberOfParticles, 10, detections, params.position_std, params.orientation_std, params.position_noise_std, params.orientation_noise_std, params.velocity_noise_std, params.acceleration_noise_std, ParticleFilter::clock::now());
                    cv::waitKey();
                    currentNumberOfParticles = params.numberOfParticles;
                    current_position_std = params.position_std;
                    current_orientation_std = params.orientation_std;
                    current_position_noise_std = params.position_noise_std;
                    current_orientation_noise_std = params.orientation_noise_std;
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

        if (recording == true)
        {
        	cv::Mat resizedImg = quadsImg.clone();
        	cv::resize(quadsImg,resizedImg,cv::Size(300,200));
        	outVideo.write(resizedImg);
        }

        if ( tracking == true )
        {
        	cout << "PARAMETERS CHANGED!!!" << endl;
        	PF.setParameters(params.numberOfParticles, 10, params.position_std, params.orientation_std, params.position_noise_std, params.orientation_noise_std, params.velocity_noise_std, params.acceleration_noise_std);
        	currentNumberOfParticles = params.numberOfParticles;
        	current_position_std = params.position_std;
        	current_orientation_std = params.orientation_std;
        	current_position_noise_std = params.position_noise_std;
        	current_orientation_noise_std = params.orientation_noise_std;
        }

        if (tracking == true)
        {
        	PF.motionUpdate(ParticleFilter::clock::now());
        	PF.measurementUpdate(detections);
        	PF.normalizeWeights();
        	PF.computeMeanPose();
        	PF.resample();
//        	if (frameNo%50 == 0)
//       		PF.displayParticles();
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
        	//out << YAML::EndSeq;
        	//std::ofstream fout("trajectory.yaml");
        	//fout << out.c_str();
        	//std::cout << "Here's the output YAML:\n" << out.c_str();
//        	fout << emitter.c_str();
        }
      }
    } catch (const cv::Exception& err) {
      ROS_ERROR_STREAM("Spin thread halted due to CV Exception: " << err.what());
    } catch (const std::string& err) {
      ROS_ERROR_STREAM("Spin thread halted due to code error: " << err);
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

  Profiler lineSegP, quadP, decoderP, durationProf, rateProf;
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
