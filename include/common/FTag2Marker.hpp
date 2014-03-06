#ifndef FTAG2MARKER_HPP_
#define FTAG2MARKER_HPP_


#include <opencv2/core/core.hpp>
#include <boost/crc.hpp>
#include <string>
#include <vector>
#include <bitset>

#include "common/BaseCV.hpp"


struct FTag2Marker {
  constexpr static unsigned int MAX_NUM_FREQS = 5;

  cv::Mat img;

  double position_x;
  double position_y;
  double position_z;

  double orientation_x;
  double orientation_y;
  double orientation_z;
  double orientation_w;

  double rectifiedWidth;

  std::vector<cv::Point2f> corners;

  bool hasSignature;
  bool hasValidXORs;
  int imgRotDir; // counter-clockwise degrees

  std::string payloadOct;
  std::string payloadBin;
  std::string xorBin; // == same data as XORExpected

  cv::Mat XORExpected;
  cv::Mat XORDecoded;
  cv::Mat payloadChunks;

  // TEMP VARS
  unsigned long long signature;

  cv::Mat horzRays;
  cv::Mat vertRays;
  cv::Mat horzMagSpec;
  cv::Mat vertMagSpec;
  cv::Mat horzPhaseSpec;
  cv::Mat vertPhaseSpec;

  cv::Mat horzMags;
  cv::Mat vertMags;
  cv::Mat horzPhases;
  cv::Mat vertPhases;

  cv::Mat mags;
  cv::Mat phases;
  cv::Mat bitChunks;

  FTag2Marker() : position_x(0), position_y(0), position_z(0),
      orientation_x(0), orientation_y(0), orientation_z(0), orientation_w(0),
      rectifiedWidth(0),
      hasSignature(false), hasValidXORs(false),
      imgRotDir(0), payloadOct(""), payloadBin(""), xorBin(""), signature(0) {
  };
  FTag2Marker(cv::Mat tag); // Extracts and analyzes rays
  virtual ~FTag2Marker() {};

  virtual void decodePayload() {};

  // Returns angle between marker's normal vector and camera's ray vector,
  // in radians. Also known as angle for out-of-plane rotation.
  //
  // WARNING: behavior is undefined if pose has not been set.
  inline double getAngleFromCamera() const {
    cv::Mat rotMat = vc_math::quat2RotMat(orientation_w, orientation_x, orientation_y, orientation_z);
    // tagVec = rotMat * [0; 0; 1]; // obtain +z ray of tag's pose, in camera frame
    // angle = acos([0; 0; 1] (dot) tagVec) // obtain angle using dot product rule
    return acos(rotMat.at<double>(2, 2));
  }
};


struct FTag2Marker6S5F3B : FTag2Marker {
  constexpr static unsigned long long SIG_KEY = 0b00100011;
  constexpr static unsigned long long SIG_KEY_FLIPPED = 0b00110001;
  constexpr static unsigned long long CRC12_KEY = 0x01F1;

  boost::crc_optimal<12, CRC12_KEY, 0, 0, false, false> CRCEngine;

  bool hasValidCRC;

  // TEMP VARS
  std::bitset<54> payload;

  unsigned long long CRC12Expected;
  unsigned long long CRC12Decoded;

  FTag2Marker6S5F3B() : FTag2Marker(),
      hasValidCRC(false), CRC12Expected(0), CRC12Decoded(0) {
    initMatrices();
  };
  FTag2Marker6S5F3B(cv::Mat tag) : FTag2Marker(tag),
      hasValidCRC(false), CRC12Expected(0), CRC12Decoded(0) {
    rectifiedWidth = double(tag.cols)/6*8;
    initMatrices();
    decodePayload();
    if (hasSignature) {
      BaseCV::rotate90(tag, img, imgRotDir/90);
    }
  };
  virtual ~FTag2Marker6S5F3B() {};

  virtual void initMatrices() {
    XORExpected = cv::Mat::zeros(6, 3, CV_8UC1);
    XORDecoded = cv::Mat::zeros(6, 3, CV_8UC1);
    payloadChunks = cv::Mat::ones(6, 3, CV_8SC1) * -1;
  };

  virtual void decodePayload();
};


#endif /* FTAG2MARKER_HPP_ */
