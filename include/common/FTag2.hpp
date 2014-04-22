#ifndef FTAG2_HPP_
#define FTAG2_HPP_


#include <opencv2/core/core.hpp>


struct FTag2Pose{
  // Camera position in tag frame
  double position_x;
  double position_y;
  double position_z;

  // Camera rotation in tag frame
  double orientation_x;
  double orientation_y;
  double orientation_z;
  double orientation_w;

  FTag2Pose() : position_x(0), position_y(0), position_z(0),
      orientation_x(0), orientation_y(0), orientation_z(0), orientation_w(0) {

  };

  virtual ~FTag2Pose() {};
  // Returns angle between marker's normal vector and camera's ray vector,
  // in radians. Also known as angle for out-of-plane rotation.
  //
  // WARNING: behavior is undefined if pose has not been set.
  double getAngleFromCamera();
};


struct FTag2Payload {
  constexpr static unsigned long long SIG_KEY = 0b00100011;
  constexpr static unsigned int MAX_NUM_FREQS = 5;

  static double WITHIN_PHASE_RANGE_N_SIGMA;
  static int WITHIN_PHASE_RANGE_ALLOWED_MISSMATCHES;
  static double WITHIN_PHASE_RANGE_THRESHOLD;

  std::vector<double> phaseVariances;

  bool hasSignature;

  bool hasValidXORs;

  std::string bitChunksStr;
  std::string decodedPayloadStr;

  unsigned int numDecodedPhases;
  unsigned int numDecodedSections;

  cv::Mat payloadChunks; // TODO: 1 remove?

  // TODO: 2 remove TEMP VARS
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
  cv::Mat bitChunks; // TODO: 1 remove bitChunks and payloadChunks (since it's confusing whether they are storing decodedPayload contents, or single-tag contents)

  FTag2Payload (): phaseVariances(), hasSignature(false),
      hasValidXORs(false), bitChunksStr(""), decodedPayloadStr(""),
      numDecodedPhases(0), numDecodedSections(0), signature(0) {
    for (int i = 0; i < 5; i++) { phaseVariances.push_back(0); }
        payloadChunks = cv::Mat::ones(6, 5, CV_8SC1) * -1;
  };
  ~FTag2Payload() {};

  FTag2Payload(const FTag2Payload& other) :
    phaseVariances(other.phaseVariances),
    hasSignature(other.hasSignature),
    hasValidXORs(other.hasValidXORs),
    bitChunksStr(other.bitChunksStr),
    decodedPayloadStr(other.decodedPayloadStr),
    numDecodedPhases(other.numDecodedPhases),
    numDecodedSections(other.numDecodedSections),
    payloadChunks(other.payloadChunks.clone()),
    signature(other.signature),
    horzRays(other.horzRays.clone()),
    vertRays(other.vertRays.clone()),
    horzMagSpec(other.horzMagSpec.clone()),
    vertMagSpec(other.vertMagSpec.clone()),
    horzPhaseSpec(other.horzPhaseSpec.clone()),
    vertPhaseSpec(other.vertPhaseSpec.clone()),
    horzMags(other.horzMags.clone()),
    vertMags(other.vertMags.clone()),
    horzPhases(other.horzPhases.clone()),
    vertPhases(other.vertPhases.clone()),
    mags(other.mags.clone()),
    phases(other.phases.clone()),
    bitChunks(other.bitChunks.clone()) {};

  static void updateParameters(double WITHIN_PHASE_RANGE_N_SIGMA_, int WITHIN_PHASE_RANGE_ALLOWED_MISSMATCHES_, double WITHIN_PHASE_RANGE_THRESHOLD_ ) {
	  WITHIN_PHASE_RANGE_N_SIGMA = WITHIN_PHASE_RANGE_N_SIGMA_;
	  WITHIN_PHASE_RANGE_ALLOWED_MISSMATCHES = WITHIN_PHASE_RANGE_ALLOWED_MISSMATCHES_;
	  WITHIN_PHASE_RANGE_THRESHOLD = WITHIN_PHASE_RANGE_THRESHOLD_;
  }

  void operator=(const FTag2Payload& other) {
    phaseVariances = other.phaseVariances;
    hasSignature = other.hasSignature;
    hasValidXORs = other.hasValidXORs;
    bitChunksStr = other.bitChunksStr;
    decodedPayloadStr = other.decodedPayloadStr;
    numDecodedPhases = other.numDecodedPhases;
    numDecodedSections = other.numDecodedSections;
    payloadChunks = other.payloadChunks.clone();
    signature = other.signature;
    horzRays = other.horzRays.clone();
    vertRays = other.vertRays.clone();
    horzMagSpec = other.horzMagSpec.clone();
    vertMagSpec = other.vertMagSpec.clone();
    horzPhaseSpec = other.horzPhaseSpec.clone();
    vertPhaseSpec = other.vertPhaseSpec.clone();
    horzMags = other.horzMags.clone();
    vertMags = other.vertMags.clone();
    horzPhases = other.horzPhases.clone();
    vertPhases = other.vertPhases.clone();
    mags = other.mags.clone();
    phases = other.phases.clone();
    bitChunks = other.bitChunks.clone();
  };

  bool withinPhaseRange(const FTag2Payload& marker);

  double sumOfStds() {
    double sum = 0.0;
    for ( double d : phaseVariances )
      sum += sqrt(d);
    return sum;
  };
};


struct FTag2Marker {
  cv::Mat img;

  double rectifiedWidth;

  std::vector<cv::Point2f> corners;
  std::vector<cv::Point2f> back_proj_corners;
  cv::Mat cornersInCamSpace;
  int imgRotDir; // counter-clockwise degrees

  FTag2Pose pose;
  FTag2Payload payload;

  FTag2Marker(double quadWidth = 0.0) : rectifiedWidth(quadWidth),
      imgRotDir(0) { };
  virtual ~FTag2Marker() {};

  FTag2Marker(const FTag2Marker& other) :
    img(other.img.clone()),
    rectifiedWidth(other.rectifiedWidth),
    corners(other.corners),
    back_proj_corners(other.back_proj_corners),
    imgRotDir(other.imgRotDir),
    pose(other.pose),
    payload(other.payload) {};

  void operator=(const FTag2Marker& other) {
    img = other.img.clone();
    rectifiedWidth = other.rectifiedWidth;
    corners = other.corners;
    back_proj_corners = other.back_proj_corners;
    imgRotDir = other.imgRotDir;
    pose = other.pose;
    payload = other.payload;
  };
};


class FTag2 {
public:
  FTag2() {};
  ~FTag2() {};
};


#endif /* FTAG2_HPP_ */
