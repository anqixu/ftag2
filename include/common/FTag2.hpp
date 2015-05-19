#ifndef FTAG2_HPP_
#define FTAG2_HPP_


#include <opencv2/core/core.hpp>
#include <iterator>
#include <vector>


/**
 * This structure contains position and orientation information of a given tag,
 * defined with respect to the camera (a.k.a. static / world) frame.
 * More concretely, this pose information together defines a transformation
 * matrix T, such that:
 *
 * homogeneous_point_in_camera_frame = T * homogeneous_point_in_tag_frame
 *
 * Both the tag frame and camera frame adheres to right-handed conventions.
 *
 * +x in camera's frame: moving tag towards right of camera, a.k.a. towards left in image
 * +y in camera's frame: moving tag towards bottom of camera, a.k.a. towards bottom in image
 * +z in camera's frame: moving tag away from camera, a.k.a. decreasing size in image
 *
 * +x in tag's frame: towards right of tag image
 * +y in tag's frame: towards bottom of tag image
 * +z in tag's frame: point into tag image
 */
struct FTag2Pose{
  // Tag's position in camera frame
  double position_x;
  double position_y;
  double position_z;

  // Tag's orientation in camera frame
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
  double computeOutOfTagPlaneAngle();
};


struct FTag2Payload {
  // NOTE: when adding a new tag type, make sure to update the following fns:
  // - FTag2Payload::SIG_KEY()
  // - FTag2Payload::BITS_PER_FREQ()
  // - FTag2Payload::NUM_FREQS()
  // - FTag2Payload::NUM_SLICES()
  //
  // - FTag2Decoder's decodePayload(...)
  enum {FTAG2_6S5F3B = 653, FTAG2_6S5F33322B = 6533322, FTAG2_6S5F33222B = 6533222, FTAG2_6S5F22111B = 6522111, FTAG2_6S2F21B = 6221, FTAG2_6S2F22B = 6222, FTAG2_6S3F211B = 63211};
  int type;

  cv::Mat mags;
  cv::Mat phases;
  cv::Mat phasesBiasAdj;
  std::vector<double> phaseVariances;

  bool hasSignature;
  bool hasValidXORs;

  std::string bitChunksStr;
  std::string decodedPayloadStr;

  unsigned int numDecodedPhases;
  unsigned int numDecodedSections;

  // TODO: 8 since this is a param governed by tracker, ideally tracker.step() or the tracker object should take in this value, then use it in correspondence and pass directly to FTag2Payload::withinPhaseRange()
  static double WITHIN_PHASE_RANGE_THRESHOLD;
  static void updateParameters(double WITHIN_PHASE_RANGE_THRESHOLD_) {
    WITHIN_PHASE_RANGE_THRESHOLD = WITHIN_PHASE_RANGE_THRESHOLD_;
  };

  static constexpr unsigned long long SIG_KEY() { return 0b00100011; };
  std::vector<unsigned int> BITS_PER_FREQ() {
    std::vector<unsigned int> result;
    switch (type) {
    case FTAG2_6S5F3B:
      result = std::vector<unsigned int>({3, 3, 3, 3, 3});
      break;
    case FTAG2_6S5F33322B:
      result = std::vector<unsigned int>({3, 3, 3, 2, 2});
      break;
    case FTAG2_6S5F33222B:
      result = std::vector<unsigned int>({3, 3, 2, 2, 2});
      break;
    case FTAG2_6S5F22111B:
      result = std::vector<unsigned int>({2, 2, 1, 1, 1});
      break;
    case FTAG2_6S2F21B:
      result = std::vector<unsigned int>({2, 1});
      break;
    case FTAG2_6S2F22B:
      result = std::vector<unsigned int>({2, 2});
      break;
    case FTAG2_6S3F211B:
      result = std::vector<unsigned int>({2, 1, 1});
      break;
    default:
      break;
    }
    return result;
  };
  unsigned int NUM_FREQS() {
    unsigned int result = 0;
    switch (type) {
    case FTAG2_6S5F3B:
    case FTAG2_6S5F33322B:
    case FTAG2_6S5F33222B:
    case FTAG2_6S5F22111B:
      result = 5;
      break;
    case FTAG2_6S2F21B:
    case FTAG2_6S2F22B:
      result = 2;
      break;
    case FTAG2_6S3F211B:
      result = 3;
      break;
    default:
      break;
    }
    return result;
  };
  unsigned int NUM_SLICES() {
    unsigned int result = 0;
    switch (type) {
    case FTAG2_6S5F3B:
    case FTAG2_6S5F33322B:
    case FTAG2_6S5F33222B:
    case FTAG2_6S5F22111B:
    case FTAG2_6S2F21B:
    case FTAG2_6S2F22B:
    case FTAG2_6S3F211B:
      result = 6;
      break;
    default:
      break;
    }
    return result;
  };

  FTag2Payload (int tagType) :
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
  std::vector<cv::Point2f> tagCorners; /** vectorized as x1, y1, x2, y2, ...; in CCW order (w.r.t. tag's coordinate frame, where +x: right, +y: bottom); starting with top-right corner **/
  double tagWidth; /** in pixels **/

  FTag2Pose pose;
  FTag2Payload payload;

  // DEBUG VARIABLES
  cv::Mat tagImg;
  int tagImgCCRotDeg;

  std::vector<cv::Point2f> back_proj_corners; // TODO: 1 is this needed in final API?
  cv::Mat cornersInCamSpace; // TODO: 1 is this needed in final API?

  FTag2Marker(int tagType, double quadWidth = 0.0) :
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
