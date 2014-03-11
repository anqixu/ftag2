#include "decoder/FTag2Decoder.hpp"
#include "detector/FTag2Detector.hpp"


FTag2Marker FTag2Decoder::decodeTag(const cv::Mat quadImg,
    const Quad& quad,
    double markerWidthM,
    const cv::Mat cameraIntrinsic, const cv::Mat cameraDistortion,
    double quadMaxStripAvgDiff,
    double tagBorderMeanMaxThresh, double tagBorderStdMaxThresh,
    PhaseVariancePredictor& phaseVariancePredictor) {
  // Trim tag borders
  cv::Mat trimmedTagImg = trimFTag2Quad(quadImg, quadMaxStripAvgDiff);

  // Validate marker borders
  if (!validateTagBorder(trimmedTagImg, tagBorderMeanMaxThresh, tagBorderStdMaxThresh)) {
    throw std::string("tag border not sufficiently dark and/or uniform");
  }

  // Crop payload portion of marker
  cv::Mat croppedTagImg = cropFTag2Border(trimmedTagImg);

  // Initialize tag data structure
  FTag2Marker tagBuffer(trimmedTagImg.rows);

  // Extract rays, and decode frequency and phase spectra
  analyzeRays(croppedTagImg, &tagBuffer);

  // Decode signature
  if (!checkSignature(&tagBuffer)) {
    throw std::string("could not identify signature");
  }

  // Store properly-rotated image of tag
  BaseCV::rotate90(trimmedTagImg, tagBuffer.img, tagBuffer.imgRotDir/90);

  // Compute pose of tag
  tagBuffer.corners.clear();
  switch ((tagBuffer.imgRotDir/90) % 4) {
  case 1:
    tagBuffer.corners.push_back(quad.corners[1]);
    tagBuffer.corners.push_back(quad.corners[2]);
    tagBuffer.corners.push_back(quad.corners[3]);
    tagBuffer.corners.push_back(quad.corners[0]);
    break;
  case 2:
    tagBuffer.corners.push_back(quad.corners[2]);
    tagBuffer.corners.push_back(quad.corners[3]);
    tagBuffer.corners.push_back(quad.corners[0]);
    tagBuffer.corners.push_back(quad.corners[1]);
    break;
  case 3:
    tagBuffer.corners.push_back(quad.corners[3]);
    tagBuffer.corners.push_back(quad.corners[0]);
    tagBuffer.corners.push_back(quad.corners[1]);
    tagBuffer.corners.push_back(quad.corners[2]);
    break;
  default:
    tagBuffer.corners = quad.corners;
    break;
  }
  solvePose(tagBuffer.corners, markerWidthM,
      cameraIntrinsic, cameraDistortion,
      tagBuffer.pose.position_x, tagBuffer.pose.position_y, tagBuffer.pose.position_z,
      tagBuffer.pose.orientation_w, tagBuffer.pose.orientation_x, tagBuffer.pose.orientation_y,
      tagBuffer.pose.orientation_z);

  // Predict phase variances
  phaseVariancePredictor.predict(&tagBuffer);

  return tagBuffer;
};


void FTag2Decoder::analyzeRays(const cv::Mat& img, FTag2Marker* tag) {
  assert(img.channels() == 1);

  img.copyTo(tag->img);

  // Extract rays
  tag->payload.horzRays = extractHorzRays(tag->img);
  tag->payload.vertRays = extractVertRays(tag->img);

  // Compute magnitude and phase spectra for both horizontal and vertical rays
  cv::Mat flippedRays;
  cv::Mat fft;
  cv::vector<cv::Mat> fftChannels(2);

  cv::dft(tag->payload.horzRays, fft, cv::DFT_ROWS + cv::DFT_COMPLEX_OUTPUT, tag->payload.horzRays.rows);
  cv::split(fft, fftChannels);
  cv::magnitude(fftChannels[0], fftChannels[1], tag->payload.horzMagSpec);
  cv::phase(fftChannels[0], fftChannels[1], tag->payload.horzPhaseSpec, true);

  cv::flip(tag->payload.vertRays, flippedRays, 1);
  cv::dft(flippedRays, fft, cv::DFT_ROWS + cv::DFT_COMPLEX_OUTPUT, flippedRays.rows);
  cv::split(fft, fftChannels);
  cv::magnitude(fftChannels[0], fftChannels[1], tag->payload.vertMagSpec);
  cv::phase(fftChannels[0], fftChannels[1], tag->payload.vertPhaseSpec, true);

  // Extract spectra responses at relevant frequencies
  int colMax = std::min(int(1+FTag2Payload::MAX_NUM_FREQS), tag->payload.horzMagSpec.cols/2);
  tag->payload.horzMags = tag->payload.horzMagSpec(cv::Range::all(), cv::Range(1, colMax)).clone();
  tag->payload.vertMags = tag->payload.vertMagSpec(cv::Range::all(), cv::Range(1, colMax)).clone();
  tag->payload.horzPhases = tag->payload.horzPhaseSpec(cv::Range::all(), cv::Range(1, colMax)).clone();
  tag->payload.vertPhases = tag->payload.vertPhaseSpec(cv::Range::all(), cv::Range(1, colMax)).clone();
};


bool FTag2Decoder::checkSignature(FTag2Marker* tag) {
  tag->payload.hasSignature = false;

  // Extract and validate phase signature, and determine orientation
  if (tag->payload.horzMags.cols < 5 || tag->payload.vertMags.cols < 5) {
    throw std::string("Insufficient number of frequencies in tag spectra");
    return false;
  }
  const int colMax = 5;

  unsigned long long sigKey = tag->payload.SIG_KEY;

  if ((tag->payload.signature = FTag2Decoder::_extractSigBits(tag->payload.horzPhases, false, 3)) == sigKey) {
    tag->imgRotDir = 0;
    tag->payload.mags = tag->payload.horzMags(cv::Range::all(), cv::Range(0, colMax)).clone();
    tag->payload.phases = tag->payload.horzPhases(cv::Range::all(), cv::Range(0, colMax)).clone();
  } else if ((tag->payload.signature = FTag2Decoder::_extractSigBits(tag->payload.vertPhases, false, 3)) == sigKey) {
    tag->imgRotDir = 270;
    tag->payload.mags = tag->payload.vertMags(cv::Range::all(), cv::Range(0, colMax)).clone();
    tag->payload.phases = tag->payload.vertPhases(cv::Range::all(), cv::Range(0, colMax)).clone();
  } else if ((tag->payload.signature = FTag2Decoder::_extractSigBits(tag->payload.horzPhases, true, 3)) == sigKey) {
    tag->imgRotDir = 180;
    cv::flip(tag->payload.horzMags(cv::Range::all(), cv::Range(0, colMax)), tag->payload.mags, 0);
    FTag2Decoder::flipPhases(tag->payload.horzPhases(cv::Range::all(), cv::Range(0, colMax)), tag->payload.phases);
  } else if ((tag->payload.signature = FTag2Decoder::_extractSigBits(tag->payload.vertPhases, true, 3)) == sigKey) {
    tag->imgRotDir = 90;
    cv::flip(tag->payload.vertMags(cv::Range::all(), cv::Range(0, colMax)), tag->payload.mags, 0);
    FTag2Decoder::flipPhases(tag->payload.vertPhases(cv::Range::all(), cv::Range(0, colMax)), tag->payload.phases);
  } else { // No signatures matched
    return false;
  }

  tag->payload.hasSignature = true;
  return true;
};


unsigned char bin2grayLUT[8] = {0, 1, 3, 2, 6, 7, 5, 4};
unsigned char gray2binLUT[8] = {0, 1, 3, 2, 7, 6, 4, 5};


unsigned char FTag2Decoder::bin2gray(unsigned char num) {
  if (num < 8) {
    return bin2grayLUT[num];
  } else {
    return (num >> 1) ^ num;
  }
};


unsigned char FTag2Decoder::gray2bin(unsigned char num) {
  if (num < 8) {
    return gray2binLUT[num];
  } else {
    unsigned char mask;
    for (mask = num >> 1; mask != 0; mask = mask >> 1) {
      num = num ^ mask;
    }
    return num;
  }
};


char FTag2Decoder::computeXORChecksum(long long bitChunk, unsigned int numBits) {
  char checksum = 0;
  unsigned int i;
  for (i = 0; i < numBits; i++) {
    checksum = checksum ^ (bitChunk & 0b01);
    bitChunk >>= 1;
  }
  return checksum;
};


long long FTag2Decoder::_extractSigBits(const cv::Mat& phases, bool flipped, unsigned int pskSize) {
  long long signature = 0;
  int i;
  double currPhase;
  long long currBitChunk;

  const int pskMaxCount = pow(2, pskSize);
  const double PSKRange = (360.0/pskMaxCount);
  const double PSKHalfRange = PSKRange/2;
  const long long sigBitMask = (0b01 << (pskSize - 1));

  if (flipped) {
    for (i = 0; i < phases.rows; i++) {
      signature <<= 1;

      currPhase = std::floor(vc_math::wrapAngle(
          phases.at<double>(phases.rows - 1 - i, 0) + PSKHalfRange, 360.0)/PSKRange);
      currBitChunk = (pskMaxCount - int(currPhase)) % pskMaxCount;

      if ((currBitChunk & sigBitMask) == sigBitMask) {
        signature += 1;
      }
    }
  } else {
    for (i = 0; i < phases.rows; i++) {
      signature <<= 1;

      currPhase = std::floor(vc_math::wrapAngle(
          phases.at<double>(i, 0) + PSKHalfRange, 360.0)/PSKRange);
      currBitChunk = currPhase;

      if ((currBitChunk & sigBitMask) == sigBitMask) {
        signature += 1;
      }
    }
  }

  return signature;
};


void FTag2Decoder::flipPhases(const cv::Mat& phasesSrc, cv::Mat& phasesFlipped) {
  cv::flip(phasesSrc, phasesFlipped, 0);
  double* phasesFlippedPtr = (double*) phasesFlipped.data;
  for (int i = 0; i < phasesFlipped.rows * phasesFlipped.cols; i++, phasesFlippedPtr++) {
    *phasesFlippedPtr = vc_math::wrapAngle(360.0 - *phasesFlippedPtr, 360.0);
  }
};


void FTag2Decoder::flipPSK(const cv::Mat& pskSrc, cv::Mat& pskFlipped, unsigned int pskSize) {
  cv::flip(pskSrc, pskFlipped, 0);
  double* pskFlippedPtr = (double*) pskFlipped.data;
  for (int i = 0; i < pskFlipped.rows * pskFlipped.cols; i++, pskFlippedPtr++) {
    *pskFlippedPtr = (pskSize - (unsigned int) *pskFlippedPtr) % pskSize;
  }
};


// TODO: 0 test this fn
cv::Mat FTag2Decoder::decodePhases(const cv::Mat phases,
    const std::vector<double> phaseVars, const std::vector<int> bitsPerFreq,
    double nStdThresh, bool grayCode) {
  const int NUM_RAYS = phases.rows;
  const int NUM_FREQS = phases.cols;
  cv::Mat bitChunks = cv::Mat::ones(NUM_RAYS, NUM_FREQS, CV_8SC1) * -1;
  // TODO: 2 make the following more efficient
  for (int freq = 0; freq < NUM_FREQS; freq++) {
    double maxBitValue = pow(2, bitsPerFreq[freq]);
    double phaseBinDeg = 360.0/maxBitValue;
    double phaseStdBinNormed = phaseVars[freq] / phaseBinDeg;
    for (int ray = 0; ray < NUM_RAYS; ray++) {
      double phaseBinNormed = phases.at<double>(ray, freq) / phaseBinDeg + 0.5;
      if (floor(phaseBinNormed - nStdThresh*phaseStdBinNormed) == floor(phaseBinNormed + nStdThresh*phaseStdBinNormed)) {
        bitChunks.at<char>(ray, freq) = phaseBinNormed;
      }
    }
  }
  return bitChunks;
};
