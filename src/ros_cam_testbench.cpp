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

#include <tf/LinearMath/Matrix3x3.h>

//#define SAVE_IMAGES_FROM sourceImgRot
//#define ENABLE_PROFILER

#define PARTICLE_FILTER


using namespace std;
using namespace std::chrono;
using namespace cv;
using namespace vc_math;
using namespace ompl::base;

typedef dynamic_reconfigure::Server<ftag2::CamTestbenchConfig> ReconfigureServer;

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

static const std::string OPENCV_WINDOW = "Image window";


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

	double tag_size = 70.0; // mm
	double scale_factor = 100.0; // 1.0 = mm , 10.0 = cm , 100.0 = m
	double offset = 0.0;

	x = (-tag_size/2.0)/scale_factor; y = offset + (-tag_size/2.0)/scale_factor; z = 0.0;
	points.push_back(cv::Point3f(x,y,z));

	x = (tag_size/2.0)/scale_factor; y = offset + (-tag_size/2.0)/scale_factor; z = 0.0;
	points.push_back(cv::Point3f(x,y,z));

	x = (tag_size/2.0)/scale_factor; y = offset + (tag_size/2.0)/scale_factor; z = 0.0;
	points.push_back(cv::Point3f(x,y,z));

	x = (-tag_size/2.0)/scale_factor; y = offset + (tag_size/2.0)/scale_factor; z = 0.0;
	points.push_back(cv::Point3f(x,y,z));

	for(unsigned int i = 0; i < points.size(); ++i)
    {
		std::cout << points[i] << std::endl;
    }

	return points;
}

bool compareArea(const Quad& first, const Quad& second) {
  return first.area > second.area;
};


class RosFTag2Testbench
{
  ros::NodeHandle local_nh;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;
  //message_filters::Subscriber<CameraInfo> cam_info_;


  public:
	CvMat *intrinsic;
	CvMat *distortion;
    cv::Mat distCoeffs;
    cv::Mat cameraMatrix;
    ros::NodeHandle n;
    ros::Publisher marker_pub;
    uint32_t shape;
    visualization_msgs::Marker marker;
    //geometry_msgs::PoseStamped marker;
    YAML::Emitter out;
    int frameNo;
    std::vector <FTag2Marker> detections;
    bool yaml_recording;
    #ifdef PARTICLE_FILTER
    bool tracking;
    ParticleFilter PF;
    int currentNumberOfParticles;
    double current_position_std;
    double current_orientation_std;
    double current_position_noise_std;
    double current_orientation_noise_std;
    double current_velocity_noise_std;
    double current_acceleration_noise_std;
#endif

    RosFTag2Testbench() : local_nh("~"), it_(local_nh), dynCfgSyncReq(false), alive(false), dstID(0), dstFilename((char*) calloc(1000, sizeof(char))), latestProfTime(ros::Time::now()), waitKeyDelay(30) {
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
	  params.numberOfParticles = 100;
	  params.position_std = 0.15;
	  params.orientation_std = 0.15;
	  params.position_noise_std = 0.15;
	  params.orientation_noise_std = 0.15;
	  params.velocity_noise_std = 0.05;
	  params.acceleration_noise_std = 0.01;

	  // Setup dynamic reconfigure server
	  dynCfgServer = new ReconfigureServer(dynCfgMutex, local_nh);
	  dynCfgServer->setCallback(bind(&RosFTag2Testbench::configCallback, this, _1, _2));

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

	  marker_pub = n.advertise<visualization_msgs::Marker>("visualization_marker", 0);
	  //marker_pub = n.advertise<geometry_msgs::PoseStamped>("visualization_marker", -1);
	  shape = visualization_msgs::Marker::ARROW;
	  marker.header.frame_id = "aqua_base";
	  marker.lifetime = ros::Duration();
	  marker.ns = "basic_shapes";
	  marker.type = visualization_msgs::Marker::ARROW;

	  marker.color.r = 1.0f;
	  marker.color.g = 0.0f;
	  marker.color.b = 0.0f;
	  marker.color.a = 1.0;

	  marker.action = visualization_msgs::Marker::ADD;

	  out << YAML::BeginSeq;
	  frameNo = 0;

#ifdef PARTICLE_FILTER
	  tracking = false;
	  detections = std::vector<FTag2Marker>();
#endif
	  yaml_recording = false;

	  alive = true;

    //	 Subscribe to input video feed and publish output video feed
	  //image_sub_ = it_.subscribeCamera("/usb_cam/image_raw", 1, &RosFTag2Testbench::imageCb, this);
	  image_sub_ = it_.subscribe("/usb_cam/image_raw", 1, &RosFTag2Testbench::imageCb, this);
  };

  ~RosFTag2Testbench()
  {
    cv::destroyWindow(OPENCV_WINDOW);
  }

  void imageCb(const sensor_msgs::ImageConstPtr& raw_image /*, const sensor_msgs::CameraInfoConstPtr& cam_info */) {
	  if (!alive) { return; }
	  cv_bridge::CvImagePtr input_bridge;
	  try {
		  input_bridge = cv_bridge::toCvCopy(raw_image, sensor_msgs::image_encodings::BGR8);
	  }
	  catch (cv_bridge::Exception& ex){
		  ROS_ERROR("[draw_frames] Failed to convert image");
		  return;
	  }
	  cv::Mat receivedImage = input_bridge->image;
	  sourceImg = receivedImage.clone();
	  //imshow("Original image",receivedImage);
	  //waitKey();

	  cv::Point2d sourceCenter;
	  cv::Mat rotMat;
	  char c;

	  try {
		  // Update params back to dyncfg
		  if (dynCfgSyncReq) {
			  if (dynCfgMutex.try_lock()) { // Make sure that dynamic reconfigure server or config callback is not active
				dynCfgMutex.unlock();
				dynCfgServer->updateConfig(params);
				ROS_DEBUG_STREAM("Updated params");
				dynCfgSyncReq = false;
			  }
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

						  drawQuad(quadsImg, currQuad->corners);
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

						  cv::Mat rotationMatrix(3,3,cv::DataType<double>::type);
						  cv::Rodrigues(rvec,rotationMatrix);
						  tf::Matrix3x3 rotMat( rotationMatrix.at<double>(0,0), rotationMatrix.at<double>(0,1), rotationMatrix.at<double>(0,2),
								  	  	  	  rotationMatrix.at<double>(1,0), rotationMatrix.at<double>(1,1), rotationMatrix.at<double>(1,2),
								  	  	  	  rotationMatrix.at<double>(2,0), rotationMatrix.at<double>(2,1), rotationMatrix.at<double>(2,2) );

						  tf::Quaternion quat;
						  rotMat.getRotation(quat);

						  static tf::TransformBroadcaster br;
						  tf::Transform transform;
						  double scale_factor = 1.0;
						  transform.setOrigin( tf::Vector3(tvec.at<double>(0)/scale_factor, tvec.at<double>(1)/scale_factor, tvec.at<double>(2)/scale_factor) );
						  transform.setRotation( quat );
						  br.sendTransform( tf::StampedTransform( transform, ros::Time::now(), "camera", "aqua_base" ) );

						  marker.header.stamp = ros::Time::now();

						  marker.pose.position.x = tvec.at<double>(0)/scale_factor;
						  marker.pose.position.y = tvec.at<double>(1)/scale_factor;
						  marker.pose.position.z = tvec.at<double>(2)/scale_factor;

						  marker.pose.orientation.x = quat.getX();
						  marker.pose.orientation.y = quat.getY();
						  marker.pose.orientation.z = quat.getZ();
						  marker.pose.orientation.w = quat.getW();

						  marker.lifetime = ros::Duration();
						  marker.type = visualization_msgs::Marker::ARROW;
						  marker.action = visualization_msgs::Marker::ADD;

						  marker.id = frameNo;
						  marker.scale.x = 0.1;
						  marker.scale.y = 0.1;
						  marker.scale.z = 0.1;
						  // Publish the marker
						  marker_pub.publish(marker);

						  std::ostringstream oss;
						  oss << frameNo;

						  if ( yaml_recording == true )
						  {
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
						  }
						  //              recording = true;
						  detections = std::vector<FTag2Marker>(1);
						  detections[0].position_x = tvec.at<double>(0)/scale_factor;
						  detections[0].position_y = tvec.at<double>(1)/scale_factor;
						  detections[0].position_z = tvec.at<double>(2)/scale_factor;

						  detections[0].orientation_x = quat.getX();
						  detections[0].orientation_y = quat.getY();
						  detections[0].orientation_z = quat.getZ();
						  detections[0].orientation_w = quat.getW();

#ifdef PARTICLE_FILTER
						  if ( tracking == false )
						  {
							  tracking = true;
							  PF = ParticleFilter(params.numberOfParticles, 10, detections, params.position_std, params.orientation_std, params.position_noise_std, params.orientation_noise_std, params.velocity_noise_std, params.acceleration_noise_std, ParticleFilter::clock::now() );
//							  cv::waitKey();
							  currentNumberOfParticles = params.numberOfParticles;
							  current_position_std = params.position_std;
							  current_orientation_std = params.orientation_std;
							  current_position_noise_std = params.position_noise_std;
							  current_orientation_noise_std = params.orientation_noise_std;
							  current_velocity_noise_std = params.velocity_noise_std;
							  current_acceleration_noise_std = params.acceleration_noise_std;
						  }
#endif
						  foundTag = true;
						  break; // stop scanning for more quads
					  }
				  }
			  }
		  }
		  if (!foundTag) {
			  cv::imshow("quads", sourceImgRot);
		  }

#ifdef PARTICLE_FILTER
		  if ( tracking == true )
		  {
			  cout << "PARAMETERS CHANGED!!!" << endl;
			  PF.setParameters(params.numberOfParticles, 10, params.position_std, params.orientation_std, params.position_noise_std, params.orientation_noise_std, params.velocity_noise_std, params.acceleration_noise_std);
			  currentNumberOfParticles = params.numberOfParticles;
			  current_position_std = params.position_std;
			  current_orientation_std = params.orientation_std;
			  current_position_noise_std = params.position_noise_std;
			  current_orientation_noise_std = params.orientation_noise_std;
			  current_velocity_noise_std = params.velocity_noise_std;
			  current_acceleration_noise_std = params.acceleration_noise_std;
		  }

		  if (tracking == true)
		  {
			  PF.motionUpdate(ParticleFilter::clock::now());
			  //cv::waitKey();
			  PF.measurementUpdate(detections);
			  PF.normalizeWeights();
			  //PF.computeMeanPose();
			  PF.computeModePose();
			  //PF.displayParticles();
			  PF.resample();
			  //        	if (frameNo%50 == 0)
			  //cv::waitKey();
		  }
#endif

		  // Spin ROS and HighGui
		  c = waitKey(waitKeyDelay);
		  if ((c & 0x0FF) == 'x' || (c & 0x0FF) == 'X') {
			  alive = false;
			  out << YAML::EndSeq;
			  std::ofstream fout("/home/dacocp/Dropbox/catkin_ws/trajectory.yaml");
			  fout << out.c_str();
			  std::cout << "Here's the output YAML:\n" << out.c_str();
			  ros::shutdown();
		  }
		  else if ( (c & 0x0FF) == 'r' || (c & 0x0FF) == 'R' )
		  {
			  if ( yaml_recording == true )
			  {
				  yaml_recording = false;
				  cout << "NOT RECORDING" << endl;
			  }
			  else
			  {
				  yaml_recording = true;
				  cout << "RECORDING" << endl;
			  }
		  }
	  } catch (const cv::Exception& err) {
		  ROS_ERROR_STREAM("Spin thread halted due to CV Exception: " << err.what());
	  } catch (const std::string& err) {
		  ROS_ERROR_STREAM("Spin thread halted due to code error: " << err);
	  }
  };

  void configCallback(ftag2::CamTestbenchConfig& config, uint32_t level) {
    if (!alive) return;
    params = config;
  };


protected:

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

int main(int argc, char** argv)
{
  ros::init(argc, argv, "ros_cam_testbench");
  RosFTag2Testbench ic;
  ros::spin();
  return 0;
}



/*void imageCb(const sensor_msgs::ImageConstPtr& msg)
{
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  // Draw an example circle on the video stream
  if (cv_ptr->image.rows > 60 && cv_ptr->image.cols > 60)
    cv::circle(cv_ptr->image, cv::Point(50, 50), 10, CV_RGB(255,0,0));

  // Update GUI Window
  cv::imshow(OPENCV_WINDOW, cv_ptr->image);
  cv::waitKey(3);

  // Output modified video stream
  image_pub_.publish(cv_ptr->toImageMsg());
}*/

