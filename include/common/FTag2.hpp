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
  enum {FTAG2_2F6S = 26, FTAG2_5F6S = 56};
  int type;

  cv::Mat mags;
  cv::Mat phases;
  std::vector<double> phaseVariances;

  bool hasSignature;
  bool hasValidXORs;

  std::string bitChunksStr;
  std::string decodedPayloadStr;

  unsigned int numDecodedPhases;
  unsigned int numDecodedSections;

  // TODO: 1 move these outside of payload struct
  static double WITHIN_PHASE_RANGE_N_SIGMA;
  static int WITHIN_PHASE_RANGE_ALLOWED_MISSMATCHES;
  static double WITHIN_PHASE_RANGE_THRESHOLD;

  static constexpr unsigned int SIG_PSK_SIZE() { return 3; }; // NOTE: in future, need to remove 'static constexpr' qualifier if sig PSK differs from 3 bits
  static constexpr unsigned long long SIG_KEY() { return 0b00100011; }; // NOTE: in future, need to remove 'static constexpr' qualifier if sig depends on num slices
  unsigned int NUM_FREQS() { return type/10; };
  unsigned int NUM_SLICES() { return type % 10; };

  static void updateParameters(double WITHIN_PHASE_RANGE_N_SIGMA_, int WITHIN_PHASE_RANGE_ALLOWED_MISSMATCHES_, double WITHIN_PHASE_RANGE_THRESHOLD_ ) {
    WITHIN_PHASE_RANGE_N_SIGMA = WITHIN_PHASE_RANGE_N_SIGMA_;
    WITHIN_PHASE_RANGE_ALLOWED_MISSMATCHES = WITHIN_PHASE_RANGE_ALLOWED_MISSMATCHES_;
    WITHIN_PHASE_RANGE_THRESHOLD = WITHIN_PHASE_RANGE_THRESHOLD_;
  };

  FTag2Payload (int tagType = FTAG2_5F6S) :
      type(tagType),
      hasSignature(false), hasValidXORs(false),
      bitChunksStr(""), decodedPayloadStr(""),
      numDecodedPhases(0), numDecodedSections(0) {
    const int MAX_NUM_FREQS = NUM_FREQS();
    for (int i = 0; i < MAX_NUM_FREQS; i++) { phaseVariances.push_back(0); }
    mags = cv::Mat::zeros(NUM_SLICES(), MAX_NUM_FREQS, CV_64FC1);
    phases = cv::Mat::zeros(NUM_SLICES(), MAX_NUM_FREQS, CV_64FC1);
  };
  ~FTag2Payload() {};

  FTag2Payload(const FTag2Payload& other) :
    type(other.type),
    mags(other.mags.clone()),
    phases(other.phases.clone()),
    phaseVariances(other.phaseVariances),
    hasSignature(other.hasSignature),
    hasValidXORs(other.hasValidXORs),
    bitChunksStr(other.bitChunksStr),
    decodedPayloadStr(other.decodedPayloadStr),
    numDecodedPhases(other.numDecodedPhases),
    numDecodedSections(other.numDecodedSections) {};

  void operator=(const FTag2Payload& other) {
    type = other.type;
    mags = other.mags.clone();
    phases = other.phases.clone();
    phaseVariances = other.phaseVariances;
    hasSignature = other.hasSignature;
    hasValidXORs = other.hasValidXORs;
    bitChunksStr = other.bitChunksStr;
    decodedPayloadStr = other.decodedPayloadStr;
    numDecodedPhases = other.numDecodedPhases;
    numDecodedSections = other.numDecodedSections;
  };

  bool withinPhaseRange(const FTag2Payload& marker);

  double sumOfStds() {
    double sum = 0.0;
    for (double& d: phaseVariances) sum += sqrt(d);
    return sum;
  };
};


struct FTag2Marker {
  std::vector<cv::Point2f> tagCorners;
  double tagWidth; /** in meters **/

  FTag2Pose pose;
  FTag2Payload payload;

  // DEBUG VARIABLES
  cv::Mat tagImg;
  int tagImgCCRotDeg;

  std::vector<cv::Point2f> back_proj_corners; // TODO: 1 is this needed in final API?
  cv::Mat cornersInCamSpace; // TODO: 1 is this needed in final API?

  FTag2Marker(double quadWidth = 0.0, int tagType = FTag2Payload::FTAG2_5F6S) :
    tagWidth(quadWidth),
    payload(tagType),
    tagImgCCRotDeg(0) { };
  virtual ~FTag2Marker() {};

  FTag2Marker(const FTag2Marker& other) :
    tagCorners(other.tagCorners),
    tagWidth(other.tagWidth),
    pose(other.pose),
    payload(other.payload),
    tagImg(other.tagImg.clone()),
    tagImgCCRotDeg(other.tagImgCCRotDeg),
    back_proj_corners(other.back_proj_corners),
    cornersInCamSpace(cornersInCamSpace.clone()) {};

  void operator=(const FTag2Marker& other) {
    tagCorners = other.tagCorners;
    tagWidth = other.tagWidth;
    pose = other.pose;
    payload = other.payload;
    tagImg = other.tagImg.clone();
    tagImgCCRotDeg = other.tagImgCCRotDeg;
    back_proj_corners = other.back_proj_corners;
    cornersInCamSpace = other.cornersInCamSpace.clone();
  };
};


#endif /* FTAG2_HPP_ */
