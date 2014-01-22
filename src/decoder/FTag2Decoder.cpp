#include "decoder/FTag2Decoder.hpp"
#include "detector/FTag2Detector.hpp"


// TODO: 1 DEPRECATED -- REMOVE UPON COMMIT
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


char FTag2Decoder::computeXORChecksum(long long bitChunk, unsigned int numBits) {
  char checksum = 0;
  unsigned int i;
  for (i = 0; i < numBits; i++) {
    checksum = checksum ^ (bitChunk & 0b01);
    bitChunk >>= 1;
  }
  return checksum;
};


long long FTag2Decoder::_extractSigBits(const cv::Mat& tagPSK, bool flipped) {
  long long signature = 0;
  int i;
  long long currBitChunk;
  for (i = 0; i < tagPSK.rows; i++) {
    signature <<= 1;

    if (flipped) {
      currBitChunk = (8 - (unsigned int) tagPSK.at<double>(tagPSK.rows - 1 - i, 0)) % 8;
    } else {
      currBitChunk = tagPSK.at<double>(i, 0);
    }

    if ((currBitChunk & 0b0100) == 0b0100) {
      signature += 1;
    }
  }

  return signature;
};


void FTag2Decoder::extractPayloadFromTag(FTag2Marker* tag) {
  assert(tag->img.channels() == 1);

  // Extract rays
  tag->horzRays = extractHorzRays(tag->img);
  tag->vertRays = extractVertRays(tag->img);

  // Compute magnitude and phase spectra for both horizontal and vertical rays
  cv::Mat flippedRays;
  cv::Mat fft;
  cv::vector<cv::Mat> fftChannels(2);
  cv::flip(tag->horzRays, flippedRays, 1);
  cv::dft(flippedRays, fft, cv::DFT_ROWS + cv::DFT_COMPLEX_OUTPUT, flippedRays.rows);
  cv::split(fft, fftChannels);
  cv::magnitude(fftChannels[0], fftChannels[1], tag->horzMagSpec);
  cv::phase(fftChannels[0], fftChannels[1], tag->horzPhaseSpec, true);
  cv::dft(tag->vertRays, fft, cv::DFT_ROWS + cv::DFT_COMPLEX_OUTPUT, tag->vertRays.rows);
  cv::split(fft, fftChannels);
  cv::magnitude(fftChannels[0], fftChannels[1], tag->vertMagSpec);
  cv::phase(fftChannels[0], fftChannels[1], tag->vertPhaseSpec, true);

  // Extract spectra responses at relevant frequencies
  int colMax = std::min(int(1+FTag2Marker::MAX_NUM_FREQS), tag->horzMagSpec.cols/2);
  tag->horzMags = tag->horzMagSpec(cv::Range::all(), cv::Range(1, colMax)).clone();
  tag->vertMags = tag->vertMagSpec(cv::Range::all(), cv::Range(1, colMax)).clone();
  tag->horzPhases = tag->horzPhaseSpec(cv::Range::all(), cv::Range(1, colMax)).clone();
  tag->vertPhases = tag->vertPhaseSpec(cv::Range::all(), cv::Range(1, colMax)).clone();

  tag->decodePayload();
};

// TODO: 1 look up math on homography projection of square, and see how it's limited vs. general quad (both to obtain 3D pose, and also to pre-filter quads)

void FTag2Decoder::flipPSK(const cv::Mat& pskSrc, cv::Mat& pskFlipped, unsigned int pskSize) {
  cv::flip(pskSrc, pskFlipped, 0);
  double* pskFlippedPtr = (double*) pskFlipped.data;
  for (int i = 0; i < pskFlipped.rows * pskFlipped.cols; i++, pskFlippedPtr++) {
    *pskFlippedPtr = (pskSize - (unsigned int) *pskFlippedPtr) % pskSize;
  }
};
