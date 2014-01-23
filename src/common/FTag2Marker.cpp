#include "common/FTag2Marker.hpp"
#include "common/VectorAndCircularMath.hpp"
#include "decoder/FTag2Decoder.hpp"


FTag2Marker::FTag2Marker(cv::Mat tag) :
    position_x(0), position_y(0), position_z(0),
    orientation_x(0), orientation_y(0), orientation_z(0), orientation_w(0),
    hasSignature(false), imgRotDir(0), IDstring("") {
  FTag2Decoder::analyzeRays(tag, this);
};


void FTag2Marker6S5F3B::decodePayload() {
  int i;

  assert(!horzPhases.empty() && !vertPhases.empty());
  hasSignature = false;

  // Extract and validate phase signature, and determine orientation
  int colMax = std::min(5, horzMagSpec.cols/2);
  if ((signature = FTag2Decoder::_extractSigBits(horzPhases, false, 3)) == SIG_KEY) {
    imgRotDir = 0;
    mags = horzMags(cv::Range::all(), cv::Range(0, colMax)).clone();
    phases = horzPhases(cv::Range::all(), cv::Range(0, colMax)).clone();
  } else if ((signature = FTag2Decoder::_extractSigBits(vertPhases, false, 3)) == SIG_KEY) {
    imgRotDir = 270;
    mags = vertMags(cv::Range::all(), cv::Range(0, colMax)).clone();
    phases = vertPhases(cv::Range::all(), cv::Range(0, colMax)).clone();
  } else if ((signature = FTag2Decoder::_extractSigBits(horzPhases, true, 3)) == SIG_KEY) {
    imgRotDir = 180;
    cv::flip(horzMags(cv::Range::all(), cv::Range(0, colMax)), mags, 0);
    FTag2Decoder::flipPhases(horzPhases(cv::Range::all(), cv::Range(0, colMax)), phases);
  } else if ((signature = FTag2Decoder::_extractSigBits(vertPhases, true, 3)) == SIG_KEY) {
    imgRotDir = 90;
    cv::flip(vertMags(cv::Range::all(), cv::Range(0, colMax)), mags, 0);
    FTag2Decoder::flipPhases(vertPhases(cv::Range::all(), cv::Range(0, colMax)), phases);
  } else { // No signatures matched
    hasSignature = false;
    return;
  }

  // Convert all phases to PSK-ed bit chunks
  bitChunks = cv::Mat(6, 5, CV_8UC1);
  const double PSKRange = (360.0/3);
  const double PSKHalfRange = PSKRange/2;
  double* phasesPtr = (double*) phases.data;
  unsigned char* bitChunksPtr = (unsigned char*) bitChunks.data;
  for (i = 0; i < phases.rows * phases.cols; i++) {
    *bitChunksPtr = std::floor(vc_math::wrapAngle(*phasesPtr + PSKHalfRange, 360.0)/PSKRange);
  }

  // Extract CRC-12 from F=1
  unsigned char currBitChunk;
  CRC12 = 0;
  for (i = 0; i < 6; i++) {
    CRC12 <<= 2;
    currBitChunk = bitChunks.at<unsigned char>(i, 0);
    CRC12 += (currBitChunk & 0b011);
  }

  // Extract encoded XOR checksums from F=2
  for (i = 0; i < 6; i++) {
    currBitChunk = bitChunks.at<unsigned char>(i, 1);
    XORExpected.at<unsigned char>(i, 0) = ((currBitChunk & 0b0100) == 0b0100);
    XORExpected.at<unsigned char>(i, 1) = ((currBitChunk & 0b010) == 0b010);
    XORExpected.at<unsigned char>(i, 2) = ((currBitChunk & 0b01) == 0b01);
  }

  // Validate/correct per-slice payloads from F={3, 4, 5} against their XOR checksums
  unsigned char adjustedBitChunk;
  for (i = 0; i < 6; i++) {
    currBitChunk = FTag2Decoder::gray2bin(bitChunks.at<unsigned char>(i, 2));
    XORDecoded.at<unsigned char>(i, 0) = FTag2Decoder::computeXORChecksum(currBitChunk, 3);
    if (XORDecoded.at<unsigned char>(i, 0) != XORExpected.at<unsigned char>(i, 0)) {
      adjustedBitChunk = FTag2Decoder::adjustPSK(phases.at<double>(i, 2), 3);
      payloadBitChunks.at<unsigned char>(i, 0) = FTag2Decoder::gray2bin(adjustedBitChunk);
    } else {
      payloadBitChunks.at<unsigned char>(i, 0) = FTag2Decoder::gray2bin(currBitChunk);
    }

    currBitChunk = FTag2Decoder::gray2bin(bitChunks.at<unsigned char>(i, 3));
    XORDecoded.at<unsigned char>(i, 1) = FTag2Decoder::computeXORChecksum(currBitChunk, 3);
    if (XORDecoded.at<unsigned char>(i, 1) != XORExpected.at<unsigned char>(i, 1)) {
      adjustedBitChunk = FTag2Decoder::adjustPSK(phases.at<double>(i, 3), 3);
      payloadBitChunks.at<unsigned char>(i, 1) = FTag2Decoder::gray2bin(adjustedBitChunk);
    } else {
      payloadBitChunks.at<unsigned char>(i, 1) = FTag2Decoder::gray2bin(currBitChunk);
    }

    currBitChunk = FTag2Decoder::gray2bin(bitChunks.at<unsigned char>(i, 4));
    XORDecoded.at<unsigned char>(i, 2) = FTag2Decoder::computeXORChecksum(currBitChunk, 3);
    if (XORDecoded.at<unsigned char>(i, 2) != XORExpected.at<unsigned char>(i, 2)) {
      adjustedBitChunk = FTag2Decoder::adjustPSK(phases.at<double>(i, 4), 3);
      payloadBitChunks.at<unsigned char>(i, 2) = FTag2Decoder::gray2bin(adjustedBitChunk);
    } else {
      payloadBitChunks.at<unsigned char>(i, 2) = FTag2Decoder::gray2bin(currBitChunk);
    }
  }

  // Stitch together payload bits
  memset(payload, 0, 7);
  // TODO: 0 stitch together payloadBitChunks into char[]

  // TODO: 0 validate CRC

  // TODO: 0 eventually, we should check for the validity of magnitude spectrum
};


void FTag2Marker6S2F3B::decodePayload() {
  assert(!horzPhases.empty() && !vertPhases.empty());
  hasSignature = false;

  // TODO: 1
};
