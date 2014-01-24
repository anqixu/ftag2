#include "common/FTag2Marker.hpp"
#include "common/VectorAndCircularMath.hpp"
#include "decoder/FTag2Decoder.hpp"


using namespace std;


FTag2Marker::FTag2Marker(cv::Mat tag) :
    position_x(0), position_y(0), position_z(0),
    orientation_x(0), orientation_y(0), orientation_z(0), orientation_w(0),
    hasSignature(false), imgRotDir(0), payloadStr("") {
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
  hasSignature = true;

  // Convert all phases to PSK-ed bit chunks
  bitChunks = cv::Mat(6, 5, CV_8UC1);
  const double PSKRange = (360.0/8);
  const double PSKHalfRange = PSKRange/2;
  double* phasesPtr = (double*) phases.data;
  unsigned char* bitChunksPtr = (unsigned char*) bitChunks.data;
  for (i = 0; i < phases.rows * phases.cols; i++) {
    *bitChunksPtr = (unsigned char) std::floor(vc_math::wrapAngle(*phasesPtr + PSKHalfRange, 360.0)/PSKRange);
    phasesPtr++;
    bitChunksPtr++;
  }

  // Extract CRC-12 from F=1
  unsigned char currBitChunk;
  CRC12Expected = 0;
  for (i = 0; i < 6; i++) {
    CRC12Expected <<= 2;
    currBitChunk = bitChunks.at<unsigned char>(i, 0);
    CRC12Expected += (currBitChunk & 0b011);
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
      payloadBitChunks.at<unsigned char>(i, 0) = currBitChunk;
    }

    currBitChunk = FTag2Decoder::gray2bin(bitChunks.at<unsigned char>(i, 3));
    XORDecoded.at<unsigned char>(i, 1) = FTag2Decoder::computeXORChecksum(currBitChunk, 3);
    if (XORDecoded.at<unsigned char>(i, 1) != XORExpected.at<unsigned char>(i, 1)) {
      adjustedBitChunk = FTag2Decoder::adjustPSK(phases.at<double>(i, 3), 3);
      payloadBitChunks.at<unsigned char>(i, 1) = FTag2Decoder::gray2bin(adjustedBitChunk);
    } else {
      payloadBitChunks.at<unsigned char>(i, 1) = currBitChunk;
    }

    currBitChunk = FTag2Decoder::gray2bin(bitChunks.at<unsigned char>(i, 4));
    XORDecoded.at<unsigned char>(i, 2) = FTag2Decoder::computeXORChecksum(currBitChunk, 3);
    if (XORDecoded.at<unsigned char>(i, 2) != XORExpected.at<unsigned char>(i, 2)) {
      adjustedBitChunk = FTag2Decoder::adjustPSK(phases.at<double>(i, 4), 3);
      payloadBitChunks.at<unsigned char>(i, 2) = FTag2Decoder::gray2bin(adjustedBitChunk);
    } else {
      payloadBitChunks.at<unsigned char>(i, 2) = currBitChunk;
    }
  }

  // Concatenate bit chunks together into entire payload
  unsigned char* payloadBitChunksPtr = (unsigned char*) payloadBitChunks.data;
  unsigned int payloadIdx = 54;
  for (i = 0; i < payloadBitChunks.rows * payloadBitChunks.cols; i++) {
    payloadIdx--;
    payload[payloadIdx] = ((*payloadBitChunksPtr & 4) == 4);
    payloadIdx--;
    payload[payloadIdx] = ((*payloadBitChunksPtr & 2) == 2);
    payloadIdx--;
    payload[payloadIdx] = ((*payloadBitChunksPtr & 1) == 1);
    payloadBitChunksPtr++;
  }

  // Compute and validate CRC-12
  unsigned long long payloadLL = payload.to_ullong();
  unsigned char payloadBytes[7] = {
      (unsigned char) ((payloadLL >> 48) & 0x0FF),
      (unsigned char) ((payloadLL >> 40) & 0x0FF),
      (unsigned char) ((payloadLL >> 32) & 0x0FF),
      (unsigned char) ((payloadLL >> 24) & 0x0FF),
      (unsigned char) ((payloadLL >> 16) & 0x0FF),
      (unsigned char) ((payloadLL >> 8) & 0x0FF),
      (unsigned char) ((payloadLL) & 0x0FF)
  };
  CRCEngine.reset();
  CRCEngine = std::for_each(payloadBytes, payloadBytes + 7, CRCEngine);
  CRC12Decoded = CRCEngine();

  if (CRC12Expected == CRC12Decoded) {
    payloadStr = payload.to_string();
  } else {
    payloadStr = "";
  }

  /*
  cout << "XORExpected = ..." << endl << cv::format(XORExpected, "matlab") << endl << endl;
  cout << "XORDecoded = ..." << endl << cv::format(XORDecoded, "matlab") << endl << endl;
  cout << "payloadBitChunks = ..." << endl << cv::format(payloadBitChunks, "matlab") << endl << endl;
  cout << "payload: " << payload << endl;
  cout << "payloadBytes: " << std::hex <<
      "0b" << (unsigned short) payloadBytes[0] << "   " <<
      "0b" << (unsigned short) payloadBytes[1] << "   " <<
      "0b" << (unsigned short) payloadBytes[2] << "   " <<
      "0b" << (unsigned short) payloadBytes[3] << "   " <<
      "0b" << (unsigned short) payloadBytes[4] << "   " <<
      "0b" << (unsigned short) payloadBytes[5] << "   " <<
      "0b" << (unsigned short) payloadBytes[6] << std::dec << endl;
  cout << "CEC12Expected: " << std::hex << CRC12Expected << std::dec << endl;
  cout << "CEC12Decoded : " << std::hex << CRC12Decoded  << std::dec << endl;
  */

  // TODO: 2 eventually, we should check for the validity of magnitude spectrum
};


void FTag2Marker6S2F3B::decodePayload() {
  assert(!horzPhases.empty() && !vertPhases.empty());
  hasSignature = false;

  // TODO: 1
};
