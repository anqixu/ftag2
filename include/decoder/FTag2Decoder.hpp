#ifndef FTAG2DECODER_HPP_
#define FTAG2DECODER_HPP_


#include <opencv2/core/core.hpp>
#include "common/FTag2.hpp"
#include "detector/FTag2Detector.hpp"


/**
 * This class predicts the error bias of the decoded payload phases (in degrees)
 * inside a FTag2 marker via a data-driven linear regression model.
 *
 * Specific model forms are currently in flux; see ftag2test/matlab synthetic
 * analyses and regression scripts for more info.
 *
 * The bias-adjusted phases are stored in tag->payload.phasesBiasAdj
 */
class PhaseBiasPredictor {
protected:
  std::mutex weightsMutex;
  std::vector<double> weights;

public:
  enum MODEL_TYPE {BIAS_MODEL_POSE=0, BIAS_MODEL_QUAD, BIAS_MODEL_SIMPLE, NUM_MODELS};
  constexpr static unsigned int NUM_WEIGHTS[NUM_MODELS] = {24, 8, 8}; // TODO: 000 update David's quad model after adjusting features to have linear dependencies
  constexpr static MODEL_TYPE model = BIAS_MODEL_POSE;

  PhaseBiasPredictor() : weights(NUM_WEIGHTS[model], 0.0) {
  };

  void updateParams(const std::vector<double>& newWeights) {
    if (newWeights.size() != NUM_WEIGHTS[model]) {
      throw std::string("Failed to update bias model weights due to size mismatch");
    }
    weightsMutex.lock();
    weights = newWeights;
    weightsMutex.unlock();
  };

  void predict(FTag2Marker* tag, double markerWidthM) {
    const double NUM_FREQS = tag->payload.NUM_FREQS();
    const double NUM_SLICES = tag->payload.NUM_SLICES();
    std::vector<double> features(NUM_WEIGHTS[model], 0.0);
    double phaseBias;

    weightsMutex.lock();

    switch (model) {
    case BIAS_MODEL_SIMPLE:
    case BIAS_MODEL_QUAD: // TODO: 000 update David's quad model after adjusting features to have linear dependencies
      {
      bool ftag2_tag_img_rot_1 = (tag->tagImgCCRotDeg == 1);
      bool ftag2_tag_img_rot_2 = (tag->tagImgCCRotDeg == 2);
      bool ftag2_tag_img_rot_3 = (tag->tagImgCCRotDeg == 3);

      tag->payload.phases.copyTo(tag->payload.phasesBiasAdj);

      features[ 0] = 1;
      features[ 1] = ftag2_tag_img_rot_1;
      features[ 2] = ftag2_tag_img_rot_2;
      features[ 3] = ftag2_tag_img_rot_3;

      for (unsigned int tag_freq = 1; tag_freq <= NUM_FREQS; tag_freq++) {
        features[ 4] = tag_freq;
        features[ 5] = ftag2_tag_img_rot_1*tag_freq;
        features[ 6] = ftag2_tag_img_rot_2*tag_freq;
        features[ 7] = ftag2_tag_img_rot_3*tag_freq;
        phaseBias = 0;
        for (unsigned int wi = 0; wi < NUM_WEIGHTS[model]; wi++) {
          phaseBias += weights[wi]*features[wi];
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
      double ftag2_txy_norm = sqrt(tag->pose.position_x*tag->pose.position_x +
          tag->pose.position_y*tag->pose.position_y)/markerWidthM;
      double ftag2_tz_norm = tag->pose.position_z/markerWidthM;
      double ftag2_pitch_height_scale = 0; // TODO: 000 compute pitch and yaw using bullet (or ideally something local without dependency), then apply cos
      double ftag2_yaw_width_scale = 0;    //  ... e.g. convert from quaternion back to rotMat, then take inverse, then find code to extra rx, ry, rz angles from rotMat

      tag->payload.phases.copyTo(tag->payload.phasesBiasAdj);

      /*
       MATLAB Model Form:
       % coeffs = biasFitPoseAll.CoefficientNames;
       % for i = 1:length(coeffs), fprintf('features[%2d] = %s;\n', i-1, strrep(coeffs{i}, ':', '*')); end;
       */
      features[ 0] = 1;
      features[ 1] = ftag2_tag_img_rot_1;
      features[ 2] = ftag2_tag_img_rot_2;
      features[ 3] = ftag2_tag_img_rot_3;
      features[ 4] = ftag2_txy_norm;
      features[ 5] = ftag2_tz_norm;
      features[ 6] = ftag2_pitch_height_scale;
      features[ 7] = ftag2_yaw_width_scale;
      features[ 9] = ftag2_tag_img_rot_1*ftag2_txy_norm;
      features[10] = ftag2_tag_img_rot_2*ftag2_txy_norm;
      features[11] = ftag2_tag_img_rot_3*ftag2_txy_norm;
      features[12] = ftag2_tag_img_rot_1*ftag2_tz_norm;
      features[13] = ftag2_tag_img_rot_2*ftag2_tz_norm;
      features[14] = ftag2_tag_img_rot_3*ftag2_tz_norm;
      features[15] = ftag2_tag_img_rot_1*ftag2_pitch_height_scale;
      features[16] = ftag2_tag_img_rot_2*ftag2_pitch_height_scale;
      features[17] = ftag2_tag_img_rot_3*ftag2_pitch_height_scale;
      features[18] = ftag2_tag_img_rot_1*ftag2_yaw_width_scale;
      features[19] = ftag2_tag_img_rot_2*ftag2_yaw_width_scale;
      features[20] = ftag2_tag_img_rot_3*ftag2_yaw_width_scale;

      for (unsigned int tag_freq = 1; tag_freq <= NUM_FREQS; tag_freq++) {
        features[ 8] = tag_freq;
        features[21] = ftag2_tag_img_rot_1*tag_freq;
        features[22] = ftag2_tag_img_rot_2*tag_freq;
        features[23] = ftag2_tag_img_rot_3*tag_freq;
        phaseBias = 0;
        for (unsigned int wi = 0; wi < NUM_WEIGHTS[model]; wi++) {
          phaseBias += weights[wi]*features[wi];
        }
        for (unsigned int tag_slice = 1; tag_slice <= NUM_SLICES; tag_slice++) {
          tag->payload.phasesBiasAdj.at<double>(tag_slice-1, tag_freq-1) -= phaseBias;
        }
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
