#include "common/FTag2Marker.hpp"
#include "common/VectorAndCircularMath.hpp"
#include "decoder/FTag2Decoder.hpp"


void FTag2Marker6S5F3B::decodePayload() {
  assert(!horzPhases.empty() && !vertPhases.empty());
  isSuccessful = false;

  // Convert horzPhases and vertPhases into PSK bit chunks
  const double PSKRange = (360.0/3);
  const double PSKHalfRange = PSKRange/2;
  double* horzPhasesPtr = (double*) horzPhases.data;
  double* vertPhasesPtr = (double*) vertPhases.data;
  const int numPhases = horzPhases.rows*horzPhases.cols;
  for (int i = 0; i < numPhases; i++, horzPhasesPtr++, vertPhasesPtr++) {
    *horzPhasesPtr = std::floor(vc_math::wrapAngle(*horzPhasesPtr + PSKHalfRange, 360.0)/PSKRange);
    *vertPhasesPtr = std::floor(vc_math::wrapAngle(*vertPhasesPtr + PSKHalfRange, 360.0)/PSKRange);
  }

  // TODO: 0 eventually, we should check for the validity of magnitude spectrum

  // Extract and validate phase signature
  int colMax = std::min(5, horzMagSpec.cols/2);
  if ((signature = FTag2Decoder::_extractSigBits(horzPhases, false)) == SIG_KEY) {
    imgRotDir = 0;
    PSK = horzPhases(cv::Range::all(), cv::Range(1, colMax)).clone();
  } else if ((signature = FTag2Decoder::_extractSigBits(vertPhases, false)) == SIG_KEY) {
    imgRotDir = 270;
    PSK = vertPhases;
  } else if ((signature = FTag2Decoder::_extractSigBits(horzPhases, true)) == SIG_KEY) {
    imgRotDir = 0;
    FTag2Decoder::flipPSK(horzPhases(cv::Range::all(), cv::Range(1, colMax)), PSK, 8);
    PSK = horzPhases(cv::Range::all(), cv::Range(1, colMax)).clone();
  } else if ((signature = FTag2Decoder::_extractSigBits(vertPhases, true)) == SIG_KEY) {
    imgRotDir = 270;
    FTag2Decoder::flipPSK(vertPhases(cv::Range::all(), cv::Range(1, colMax)), PSK, 8);
  } else { // No signatures matched
    isSuccessful = false;
    return;
  }

  // Extract CRC-12 from F=1
  int i;
  long long currBitChunk;
  CRC12 = 0;
  for (i = 0; i < 6; i++) {
    CRC12 <<= 2;
    currBitChunk = PSK.at<double>(i, 0);
    CRC12 += (currBitChunk & 0b011);
  }

  // Extract encoded XOR checksums from F=2
  for (i = 0; i < 6; i++) {
    currBitChunk = PSK.at<double>(i, 1);
    XORExpected.at<char>(i, 0) = ((currBitChunk & 0b0100) == 0b0100);
    currBitChunk <<= 1;
    XORExpected.at<char>(i, 1) = ((currBitChunk & 0b0100) == 0b0100);
    currBitChunk <<= 1;
    XORExpected.at<char>(i, 2) = ((currBitChunk & 0b0100) == 0b0100);
  }

  // Extract per-slice payloads from F={3, 4, 5}, and also compute per-phase XOR checksums
  for (i = 0; i < 6; i++) {
    payloadBits[i] = 0;

    payloadBits[i] <<= 3;
    currBitChunk = PSK.at<double>(i, 2);
    XORDecoded.at<char>(i, 0) = FTag2Decoder::computeXORChecksum(currBitChunk, 3);
    payloadBits[i] += currBitChunk;

    payloadBits[i] <<= 3;
    currBitChunk = PSK.at<double>(i, 3);
    XORDecoded.at<char>(i, 1) = FTag2Decoder::computeXORChecksum(currBitChunk, 3);
    payloadBits[i] += currBitChunk;

    payloadBits[i] <<= 3;
    currBitChunk = PSK.at<double>(i, 4);
    XORDecoded.at<char>(i, 2) = FTag2Decoder::computeXORChecksum(currBitChunk, 3);
    payloadBits[i] += currBitChunk;
  }

  // TODO: 0 correct phases if XOR is incorrect

  // Validate XOR
  if (vc_math::countNotEqual(XORDecoded, XORExpected) > 0) {
    isSuccessful = false;
    return;
  }

  // Compute and validate CRC
  // TODO: 0 try using boost::crc

  // Combine payloads
  ID = 0;
};


void FTag2Marker6S2F3B::decodePayload() {
  assert(!horzPhases.empty() && !vertPhases.empty());
  isSuccessful = false;

  // TODO: 1
};
