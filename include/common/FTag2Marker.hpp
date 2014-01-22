#ifndef FTAG2MARKER_HPP_
#define FTAG2MARKER_HPP_


#include <opencv2/core/core.hpp>
#include <string>
#include <vector>


struct FTag2Marker {
  constexpr static unsigned int MAX_NUM_FREQS = 10;

  cv::Mat img;

  double position_x;
  double position_y;
  double position_z;

  double orientation_x;
  double orientation_y;
  double orientation_z;
  double orientation_w;

  std::vector<cv::Mat> rays;

  bool isSuccessful;
  int imgRotDir; // counter-clockwise degrees

  long long ID;
  std::string IDstring;

  // TEMP VARS
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
  cv::Mat PSK;

  FTag2Marker() : position_x(0), position_y(0), position_z(0),
      orientation_x(0), orientation_y(0), orientation_z(0), orientation_w(0),
      isSuccessful(false), imgRotDir(0), ID(-1), IDstring("") {};
  virtual ~FTag2Marker() {};

  virtual void decodePayload() {};
};


struct FTag2Marker6S5F3B : FTag2Marker {
  constexpr static long long SIG_KEY = 0b00010101;
  constexpr static long long CRC12_KEY = 0x08F8;

  // TEMP VARS
  long long signature;

  cv::Mat XORExpected;
  cv::Mat XORDecoded;
  std::vector<long long> payloadBits;

  long long CRC12;

  FTag2Marker6S5F3B() : FTag2Marker(), signature(0), payloadBits(6), CRC12(0) {
    XORExpected = cv::Mat::zeros(6, 3, CV_8UC1);
    XORDecoded = cv::Mat::zeros(6, 3, CV_8UC1);
  }
  virtual ~FTag2Marker6S5F3B() {};

  virtual void decodePayload();
};


struct FTag2Marker6S2F3B : FTag2Marker {
  constexpr static long long SIG_KEY = 0b00010101;
  constexpr static long long CRC8_KEY = 0x0EA;

  // TEMP VARS
  long long signature;

  std::vector<long long> payloadBits;

  long long CRC8;

  FTag2Marker6S2F3B() : FTag2Marker(), signature(0), payloadBits(6), CRC8(0) {
  }
  virtual ~FTag2Marker6S2F3B() {};

  virtual void decodePayload();
};


#endif /* FTAG2MARKER_HPP_ */
