#ifndef FTAG2MARKER_HPP_
#define FTAG2MARKER_HPP_

#include "common/FTag2Pose.hpp"
#include "common/FTag2Payload.hpp"



struct FTag2Marker {

  cv::Mat img;

  double rectifiedWidth;

  std::vector<cv::Point2f> corners;
  int imgRotDir; // counter-clockwise degrees

  FTag2Pose pose;
  FTag2Payload payload;


  FTag2Marker(double quadWidth = 0.0) : rectifiedWidth(quadWidth),
      imgRotDir(0) {
  };
  virtual ~FTag2Marker() {};

};

#endif /* FTAG2MARKER_HPP_ */
