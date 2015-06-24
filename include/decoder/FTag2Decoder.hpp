#ifndef FTAG2DECODER_HPP_
#define FTAG2DECODER_HPP_


#include <opencv2/core/core.hpp>
#include "common/FTag2.hpp"
#include "detector/FTag2Detector.hpp"


/**
 * This class predicts the bias and deviation of the error associated to the
 * decoded payload phases (in degrees) inside a FTag2 marker.
 * Biases and deviations are separately modeled using linear regression.
 *
 * Specific model forms are currently in flux; see ftag2test/matlab synthetic
 * analyses and regression scripts for more info.
 *
 * The bias-adjusted phases are stored in tag->payload.phasesBiasAdj.
 * The predicted phase stdev are stored in tag->payload.phaseVariances.
 */
class PhaseErrorPredictor {
protected:
  std::mutex weightsMutex;
  std::vector<double> biasWeights;
  std::vector<double> stdevWeights;

public:
  enum BIAS_MODEL_TYPE {BIAS_MODEL_POSE=0, BIAS_MODEL_QUAD, BIAS_MODEL_SIMPLE, NUM_BIAS_MODELS};
  enum STDEV_MODEL_TYPE {STDEV_MODEL_POSE=0, STDEV_MODEL_QUAD, STDEV_MODEL_SIMPLE, NUM_STDEV_MODELS};
  constexpr static unsigned int NUM_BIAS_WEIGHTS[NUM_BIAS_MODELS] = {20, 8, 8}; // TODO: 0 update David's quad model after adjusting features to have linear dependencies
  constexpr static unsigned int NUM_STDEV_WEIGHTS[NUM_STDEV_MODELS] = {5, 8, 8}; // TODO: 0 update David's quad model
  constexpr static BIAS_MODEL_TYPE bias_model = BIAS_MODEL_POSE;
  constexpr static STDEV_MODEL_TYPE stdev_model = STDEV_MODEL_POSE;

  PhaseErrorPredictor() :
      biasWeights(NUM_BIAS_WEIGHTS[bias_model], 0.0),
      stdevWeights(NUM_STDEV_WEIGHTS[stdev_model], 0.0) {
  };

  void updateBiasParams(const std::vector<double>& newBiasWeights) {
    if (newBiasWeights.size() != NUM_BIAS_WEIGHTS[bias_model]) {
      throw std::string("Failed to update bias model weights due to size mismatch");
    }
    weightsMutex.lock();
    biasWeights = newBiasWeights;
    weightsMutex.unlock();
  };

  void updateStdevParams(const std::vector<double>& newStdevWeights) {
    if (newStdevWeights.size() != NUM_STDEV_WEIGHTS[stdev_model]) {
      throw std::string("Failed to update deviation model weights due to size mismatch");
    }
    weightsMutex.lock();
    stdevWeights = newStdevWeights;
    weightsMutex.unlock();
  };

  void predict(FTag2Marker* tag, double markerWidthM) {
    const double NUM_FREQS = tag->payload.NUM_FREQS();
    const double NUM_SLICES = tag->payload.NUM_SLICES();
    std::vector<double> biasFeatures(NUM_BIAS_WEIGHTS[bias_model], 0.0);
    std::vector<double> stdevFeatures(NUM_STDEV_WEIGHTS[stdev_model], 0.0);

    weightsMutex.lock();

    switch (bias_model) {
    case BIAS_MODEL_SIMPLE:
    case BIAS_MODEL_QUAD: // TODO: 0 update David's quad model after adjusting features to have linear dependencies
      {
      bool ftag2_tag_img_rot_1 = (tag->tagImgCCRotDeg == 1);
      bool ftag2_tag_img_rot_2 = (tag->tagImgCCRotDeg == 2);
      bool ftag2_tag_img_rot_3 = (tag->tagImgCCRotDeg == 3);

      tag->payload.phases.copyTo(tag->payload.phasesBiasAdj);

      /*
       MATLAB Model Form:
       % coeffs = biasFitSimpleAll.CoefficientNames;
       % for i = 1:length(coeffs), fprintf('biasFeatures[%2d] = %s;\n', i-1, strrep(coeffs{i}, ':', '*')); end;
       */
      biasFeatures[ 0] = 1;
      biasFeatures[ 1] = ftag2_tag_img_rot_1;
      biasFeatures[ 2] = ftag2_tag_img_rot_2;
      biasFeatures[ 3] = ftag2_tag_img_rot_3;

      for (unsigned int tag_freq = 1; tag_freq <= NUM_FREQS; tag_freq++) {
        biasFeatures[ 4] = tag_freq;
        biasFeatures[ 5] = ftag2_tag_img_rot_1*tag_freq;
        biasFeatures[ 6] = ftag2_tag_img_rot_2*tag_freq;
        biasFeatures[ 7] = ftag2_tag_img_rot_3*tag_freq;
        double phaseBias = 0;
        for (unsigned int wi = 0; wi < NUM_BIAS_WEIGHTS[bias_model]; wi++) {
          phaseBias += biasWeights[wi]*biasFeatures[wi];
        }
        for (unsigned int tag_slice = 1; tag_slice <= NUM_SLICES; tag_slice++) {
          tag->payload.phasesBiasAdj.at<double>(tag_slice-1, tag_freq-1) -= phaseBias;
        }
      }
      }
      break;

    case BIAS_MODEL_POSE:
    default:
      {
      bool ftag2_tag_img_rot_1 = (tag->tagImgCCRotDeg == 1);
      bool ftag2_tag_img_rot_2 = (tag->tagImgCCRotDeg == 2);
      bool ftag2_tag_img_rot_3 = (tag->tagImgCCRotDeg == 3);
      //double ftag2_txy_norm = sqrt(tag->pose.position_x*tag->pose.position_x +
      //    tag->pose.position_y*tag->pose.position_y)/markerWidthM;
      double ftag2_tz_norm = tag->pose.position_z/markerWidthM;

      std::array<double, 4> quat_cam_in_tag_frame = vc_math::quatInv(
          tag->pose.orientation_w, tag->pose.orientation_x,
          tag->pose.orientation_y, tag->pose.orientation_z);
      std::array<double, 3> rxyz_cam_in_tag_frame =
          vc_math::quat2euler(quat_cam_in_tag_frame);
      double ftag2_pitch_height_scale = cos(rxyz_cam_in_tag_frame[0]);
      double ftag2_yaw_width_scale = cos(rxyz_cam_in_tag_frame[1]);

      tag->payload.phases.copyTo(tag->payload.phasesBiasAdj);

      /*
       MATLAB Model Form:
       % coeffs = biasFitPoseAll.CoefficientNames;
       % for i = 1:length(coeffs), fprintf('biasFeatures[%2d] = %s;\n', i-1, strrep(coeffs{i}, ':', '*')); end;
       */
      biasFeatures[ 0] = 1;
      biasFeatures[ 1] = ftag2_tag_img_rot_1;
      biasFeatures[ 2] = ftag2_tag_img_rot_2;
      biasFeatures[ 3] = ftag2_tag_img_rot_3;
      biasFeatures[ 4] = ftag2_tz_norm;
      biasFeatures[ 5] = ftag2_pitch_height_scale;
      biasFeatures[ 6] = ftag2_yaw_width_scale;
      biasFeatures[ 8] = ftag2_tag_img_rot_1*ftag2_tz_norm;
      biasFeatures[ 9] = ftag2_tag_img_rot_2*ftag2_tz_norm;
      biasFeatures[10] = ftag2_tag_img_rot_3*ftag2_tz_norm;
      biasFeatures[11] = ftag2_tag_img_rot_1*ftag2_pitch_height_scale;
      biasFeatures[12] = ftag2_tag_img_rot_2*ftag2_pitch_height_scale;
      biasFeatures[13] = ftag2_tag_img_rot_3*ftag2_pitch_height_scale;
      biasFeatures[14] = ftag2_tag_img_rot_1*ftag2_yaw_width_scale;
      biasFeatures[15] = ftag2_tag_img_rot_2*ftag2_yaw_width_scale;
      biasFeatures[16] = ftag2_tag_img_rot_3*ftag2_yaw_width_scale;

      for (unsigned int tag_freq = 1; tag_freq <= NUM_FREQS; tag_freq++) {
        biasFeatures[ 7] = tag_freq;
        biasFeatures[17] = ftag2_tag_img_rot_1*tag_freq;
        biasFeatures[18] = ftag2_tag_img_rot_2*tag_freq;
        biasFeatures[19] = ftag2_tag_img_rot_3*tag_freq;
        double phaseBias = 0;
        for (unsigned int wi = 0; wi < NUM_BIAS_WEIGHTS[bias_model]; wi++) {
          phaseBias += biasWeights[wi]*biasFeatures[wi];
        }
        for (unsigned int tag_slice = 1; tag_slice <= NUM_SLICES; tag_slice++) {
          tag->payload.phasesBiasAdj.at<double>(tag_slice-1, tag_freq-1) -= phaseBias;
        }
      }
      }
      break;
    }


    switch (stdev_model) {
    case STDEV_MODEL_SIMPLE:
    case STDEV_MODEL_QUAD: // TODO: 0 update David's quad model
      {
      bool ftag2_tag_img_rot_1 = (tag->tagImgCCRotDeg == 1);
      bool ftag2_tag_img_rot_2 = (tag->tagImgCCRotDeg == 2);
      bool ftag2_tag_img_rot_3 = (tag->tagImgCCRotDeg == 3);

      /*
       MATLAB Model Form:
       % coeffs = stdevFitSimpleAll.CoefficientNames;
       % for i = 1:length(coeffs), fprintf('stdevFeatures[%2d] = %s;\n', i-1, strrep(coeffs{i}, ':', '*')); end;
       */
      stdevFeatures[ 0] = 1;
      stdevFeatures[ 1] = ftag2_tag_img_rot_1;
      stdevFeatures[ 2] = ftag2_tag_img_rot_2;
      stdevFeatures[ 3] = ftag2_tag_img_rot_3;

      for (unsigned int tag_freq = 1; tag_freq <= NUM_FREQS; tag_freq++) {
        stdevFeatures[ 4] = tag_freq;
        stdevFeatures[ 5] = ftag2_tag_img_rot_1*tag_freq;
        stdevFeatures[ 6] = ftag2_tag_img_rot_2*tag_freq;
        stdevFeatures[ 7] = ftag2_tag_img_rot_3*tag_freq;
        double phaseStdev = 0;
        for (unsigned int wi = 0; wi < NUM_STDEV_WEIGHTS[stdev_model]; wi++) {
          phaseStdev += stdevWeights[wi]*stdevFeatures[wi];
        }
        tag->payload.phaseVariances[tag_freq-1] = phaseStdev*phaseStdev;
      }
      }
      break;

    case STDEV_MODEL_POSE:
    default:
      {
      double ftag2_tz_norm = tag->pose.position_z/markerWidthM;

      std::array<double, 4> quat_cam_in_tag_frame = vc_math::quatInv(
          tag->pose.orientation_w, tag->pose.orientation_x,
          tag->pose.orientation_y, tag->pose.orientation_z);
      std::array<double, 3> rxyz_cam_in_tag_frame =
          vc_math::quat2euler(quat_cam_in_tag_frame);
      double ftag2_pitch_height_scale = cos(rxyz_cam_in_tag_frame[0]);
      double ftag2_yaw_width_scale = cos(rxyz_cam_in_tag_frame[1]);

      /*
       MATLAB Model Form:
       % coeffs = stdevFitPoseAll.CoefficientNames;
       % for i = 1:length(coeffs), fprintf('stdevFeatures[%2d] = %s;\n', i-1, strrep(coeffs{i}, ':', '*')); end;
       */
      stdevFeatures[ 0] = 1;
      stdevFeatures[ 1] = ftag2_tz_norm;
      stdevFeatures[ 2] = ftag2_pitch_height_scale;
      stdevFeatures[ 3] = ftag2_yaw_width_scale;

      for (unsigned int tag_freq = 1; tag_freq <= NUM_FREQS; tag_freq++) {
        stdevFeatures[ 4] = tag_freq;
        double phaseStdev = 0;
        for (unsigned int wi = 0; wi < NUM_STDEV_WEIGHTS[stdev_model]; wi++) {
          phaseStdev += stdevWeights[wi]*stdevFeatures[wi];
        }
        tag->payload.phaseVariances[tag_freq-1] = phaseStdev*phaseStdev;
      }
      }
      break;
    }


    weightsMutex.unlock();
  };
};


// TODO: 0 deprecate this code
class PhaseVariancePredictor {
protected:
  std::mutex paramsMutex;

  double weight_r; // norm of XY components of position
  double weight_z; // projective distance from camera, in camera's ray vector
  double weight_angle; // angle between tag's normal vector and camera's ray vector (in degrees)
  double weight_freq; // encoding frequency of phase
  double weight_bias; // constant bias


public:
  PhaseVariancePredictor() : weight_r(-0.433233403141656), weight_z(1.178509836433552), weight_angle(0.225729455615220),
      weight_freq(3.364693352246631), weight_bias(-4.412137643426274) {};

  void updateParams(double w_r, double w_z, double w_a, double w_f, double w_b) {
    paramsMutex.lock();
//    weight_r = w_r;
//    weight_z = w_z;
//    weight_angle = w_a;
//    weight_freq = w_f;
//    weight_bias = w_b;
    paramsMutex.unlock();
  };

  void predict(FTag2Marker* tag) {
    double r = sqrt(tag->pose.position_x*tag->pose.position_x + tag->pose.position_y*tag->pose.position_y);
    double z = tag->pose.position_z;
    double angle = tag->pose.computeOutOfTagPlaneAngle()*vc_math::radian;
    paramsMutex.lock();
    for (unsigned int freq = 1; freq <= tag->payload.NUM_FREQS(); freq++) {
      tag->payload.phaseVariances[freq-1]= pow(weight_bias + weight_r*r + weight_z*z +
          weight_angle*angle + weight_freq*freq,2);
    }
    paramsMutex.unlock();
  };
};


FTag2Marker decodeQuad(const cv::Mat quadImg, const Quad& quad,
    int tagType,
    double markerWidthM,
    unsigned int numSamplesPerRow,
    const cv::Mat cameraIntrinsic, const cv::Mat cameraDistortion,
    double quadMaxStripAvgDiff,
    double tagBorderMeanMaxThresh, double tagBorderStdMaxThresh,
    double magFilGainNeg, double magFilGainPos,
    double magFilPowNeg, double magFilPowPos,
    PhaseVariancePredictor& phaseVariancePredictor);


void decodePayload(FTag2Payload& tag, double nStdThresh);


#endif /* FTAG2DECODER_HPP_ */
