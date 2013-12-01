#include "decoder/FTag2Decoder.hpp"
#include "detector/FTag2Detector.hpp"


bool pskEqual(const cv::Mat& tagPSK, const std::vector<unsigned int>& sigPSK,
    unsigned int pskSize, bool flipped) {
  assert(tagPSK.rows == int(sigPSK.size()));

  bool match = true;
  unsigned int i;
  unsigned int currSigPSK;
  if (flipped) {
    for (i = 0; i < sigPSK.size(); i++) {
      currSigPSK = (pskSize - (unsigned int) tagPSK.at<double>(sigPSK.size() - 1 - i, 0)) % pskSize;
      if (currSigPSK != sigPSK[i]) {
        match = false;
        break;
      }
    }
  } else {
    for (i = 0; i < sigPSK.size(); i++) {
      currSigPSK = (unsigned int) tagPSK.at<double>(i, 0);
      if (currSigPSK != sigPSK[i]) {
        match = false;
        break;
      }
    }
  }
  return match;
};


void flipPSK(const cv::Mat& pskSrc, cv::Mat& pskFlipped, unsigned int pskSize) {
  cv::flip(pskSrc, pskFlipped, 0);
  double* pskFlippedPtr = (double*) pskFlipped.data;
  for (int i = 0; i < pskFlipped.rows * pskFlipped.cols; i++, pskFlippedPtr++) {
    *pskFlippedPtr = (pskSize - (unsigned int) *pskFlippedPtr) % pskSize;
  }
};


FTag2Decoder::FTag2Decoder() {

};


FTag2Decoder::~FTag2Decoder() {

};


FTag2 FTag2Decoder::decodeTag(cv::Mat croppedTagImg) {
  assert(croppedTagImg.channels() == 1);

  // Extract rays
  FTag2 tag;
  croppedTagImg.copyTo(tag.img);
  tag.horzRays = extractHorzRays(croppedTagImg);
  tag.vertRays = extractVertRays(croppedTagImg);

  // Compute magnitude and phase spectra for both horizontal and vertical rays
  cv::Mat flippedRays;
  cv::Mat fft;
  cv::vector<cv::Mat> fftChannels(2);
  cv::flip(tag.horzRays, flippedRays, 1);
  cv::dft(flippedRays, fft, cv::DFT_ROWS + cv::DFT_COMPLEX_OUTPUT, flippedRays.rows);
  cv::split(fft, fftChannels);
  cv::magnitude(fftChannels[0], fftChannels[1], tag.horzMagSpec);
  cv::phase(fftChannels[0], fftChannels[1], tag.horzPhaseSpec, true);
  cv::dft(tag.vertRays, fft, cv::DFT_ROWS + cv::DFT_COMPLEX_OUTPUT, tag.vertRays.rows);
  cv::split(fft, fftChannels);
  cv::magnitude(fftChannels[0], fftChannels[1], tag.vertMagSpec);
  cv::phase(fftChannels[0], fftChannels[1], tag.vertPhaseSpec, true);

  // Extract spectra responses at relevant frequencies
  int colMax = std::min(int(1+FTag2::MAX_NUM_FREQS), tag.horzMagSpec.cols/2);
  tag.horzMags = tag.horzMagSpec(cv::Range::all(), cv::Range(1, colMax)).clone();
  tag.vertMags = tag.vertMagSpec(cv::Range::all(), cv::Range(1, colMax)).clone();
  tag.horzPhases = tag.horzPhaseSpec(cv::Range::all(), cv::Range(1, colMax)).clone();
  tag.vertPhases = tag.vertPhaseSpec(cv::Range::all(), cv::Range(1, colMax)).clone();

  // Convert phase angles into PSK IDs
  const double PSKRange = (360.0/FTag2::PSK_SIZE);
  const double PSKHalfRange = PSKRange/2;
  double* horzPhasesPtr = (double*) tag.horzPhases.data;
  double* vertPhasesPtr = (double*) tag.vertPhases.data;
  const int numPhases = tag.horzPhases.rows*tag.horzPhases.cols;
  for (int i = 0; i < numPhases; i++, horzPhasesPtr++, vertPhasesPtr++) {
    *horzPhasesPtr = std::floor(vc_math::wrapAngle(*horzPhasesPtr + PSKHalfRange, 360.0)/PSKRange);
    *vertPhasesPtr = std::floor(vc_math::wrapAngle(*vertPhasesPtr + PSKHalfRange, 360.0)/PSKRange);
  }

  // TODO: 0 need to check for validity of magnitude spectrum (see TEMP DATA PROCESSING CODE BELOW)

  // TEMP CODE: Check for signature
  if (pskEqual(tag.horzPhases, FTag2::PSK_SIG, FTag2::PSK_SIZE, false)) {
    tag.hasSignature = true;
    tag.imgRotDir = 0;
    tag.PSK = tag.horzPhases;
  } else if (pskEqual(tag.vertPhases, FTag2::PSK_SIG, FTag2::PSK_SIZE, false)) {
    tag.hasSignature = true;
    tag.imgRotDir = 270;
    tag.PSK = tag.vertPhases;
  } else if (pskEqual(tag.horzPhases, FTag2::PSK_SIG, FTag2::PSK_SIZE, true)) {
    tag.hasSignature = true;
    tag.imgRotDir = 180;
    flipPSK(tag.horzPhases, tag.PSK, FTag2::PSK_SIZE);
  } else if (pskEqual(tag.vertPhases, FTag2::PSK_SIG, FTag2::PSK_SIZE, true)) {
    tag.hasSignature = true;
    tag.imgRotDir = 90;
    flipPSK(tag.vertPhases, tag.PSK, FTag2::PSK_SIZE);
  } else {
    tag.hasSignature = false;
  }

  // TEMP DATA PROCESSING BELOW
  cv::Mat magRow;
  double rowSum;
  int i;
  //double* rowPtr; int j;
  for (i = 0; i < tag.horzMags.rows; i++) {
    magRow = tag.horzMags.row(i);
    cv::minMaxLoc(magRow, NULL, &rowSum);
    magRow = magRow/rowSum*100;

    /*
    rowPtr = (double*) magRow.data;
    for (j = 0; j < magRow.cols; j++, rowPtr++) {
      *rowPtr = std::floor(*rowPtr);
    }
    */

    magRow = tag.vertMags.row(i);
    cv::minMaxLoc(magRow, NULL, &rowSum);
    magRow = magRow/rowSum*100;

    /*
    rowPtr = (double*) magRow.data;
    for (j = 0; j < magRow.cols; j++, rowPtr++) {
      *rowPtr = std::floor(*rowPtr);
    }
    */
  }
  tag.horzMags.convertTo(tag.horzMags, CV_8U);
  tag.vertMags.convertTo(tag.vertMags, CV_8U);
  cv::threshold(tag.horzMags, tag.horzMags, 50, 1, cv::THRESH_BINARY);
  cv::threshold(tag.vertMags, tag.vertMags, 50, 1, cv::THRESH_BINARY);

  if (tag.hasSignature) {
    if (tag.PSK.rows != 6 || tag.PSK.cols < 2) {
      std::cerr << "imgRotDir: " << tag.imgRotDir << std::endl;
      std::cerr << "!!!\ntag.PSK = ..." << std::endl << cv::format(tag.PSK, "matlab") << std::endl;
      assert(tag.PSK.rows == 6);
      assert(tag.PSK.cols >= 2);
    }

    tag.ID = 0;
    for (int i = 0; i < 6; i++) {
      tag.ID = tag.ID*10 + tag.PSK.at<double>(i, 1);
    }
  }

  // TODO: 1 bring in homographic transformation info
  // TODO: 1 look up math on homography projection of square, and see how it's limited vs. general quad (both to obtain 3D pose, and also to pre-filter quads)

  return tag;
};
