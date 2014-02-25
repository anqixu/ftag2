#include "decoder/FTag2Decoder.hpp"
#include "detector/FTag2Detector.hpp"


void FTag2Decoder::analyzeRays(const cv::Mat& img, FTag2Marker* tag) {
  assert(img.channels() == 1);

  img.copyTo(tag->img);

  // Extract rays
  tag->horzRays = extractHorzRays(tag->img);
  tag->vertRays = extractVertRays(tag->img);

  // Compute magnitude and phase spectra for both horizontal and vertical rays
  cv::Mat flippedRays;
  cv::Mat fft;
  cv::vector<cv::Mat> fftChannels(2);

  cv::dft(tag->horzRays, fft, cv::DFT_ROWS + cv::DFT_COMPLEX_OUTPUT, tag->horzRays.rows);
  cv::split(fft, fftChannels);
  cv::magnitude(fftChannels[0], fftChannels[1], tag->horzMagSpec);
  cv::phase(fftChannels[0], fftChannels[1], tag->horzPhaseSpec, true);

  cv::flip(tag->vertRays, flippedRays, 1);
  cv::dft(flippedRays, fft, cv::DFT_ROWS + cv::DFT_COMPLEX_OUTPUT, flippedRays.rows);
  cv::split(fft, fftChannels);
  cv::magnitude(fftChannels[0], fftChannels[1], tag->vertMagSpec);
  cv::phase(fftChannels[0], fftChannels[1], tag->vertPhaseSpec, true);

  // Extract spectra responses at relevant frequencies
  int colMax = std::min(int(1+FTag2Marker::MAX_NUM_FREQS), tag->horzMagSpec.cols/2);
  tag->horzMags = tag->horzMagSpec(cv::Range::all(), cv::Range(1, colMax)).clone();
  tag->vertMags = tag->vertMagSpec(cv::Range::all(), cv::Range(1, colMax)).clone();
  tag->horzPhases = tag->horzPhaseSpec(cv::Range::all(), cv::Range(1, colMax)).clone();
  tag->vertPhases = tag->vertPhaseSpec(cv::Range::all(), cv::Range(1, colMax)).clone();
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


// TODO: 2 look up math on homography projection of square, and see how it's limited vs. general quad (both to obtain 3D pose, and also to pre-filter quads)

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
