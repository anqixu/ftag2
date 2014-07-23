#include "decoder/FTag2Decoder.hpp"
#include "detector/FTag2Detector.hpp"

#include <iostream>
#include <iomanip>


unsigned char bin2grayLUT[8] = {0, 1, 3, 2, 6, 7, 5, 4};
unsigned char gray2binLUT[8] = {0, 1, 3, 2, 7, 6, 4, 5};


unsigned char bin2gray(unsigned char num) {
  if (num < 8) {
    return bin2grayLUT[num];
  } else {
    return (num >> 1) ^ num;
  }
};


unsigned char gray2bin(unsigned char num) {
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


char computeXORChecksum(long long bitChunk, unsigned int numBits) {
  char checksum = 0;
  unsigned int i;
  for (i = 0; i < numBits; i++) {
    checksum = checksum ^ (bitChunk & 0b01);
    bitChunk >>= 1;
  }
  return checksum;
};


// DEPRECATED FUNCTION
void flipPSK(const cv::Mat& pskSrc, cv::Mat& pskFlipped, unsigned int pskSize) {
  cv::flip(pskSrc, pskFlipped, 0);
  double* pskFlippedPtr = (double*) pskFlipped.data;
  for (int i = 0; i < pskFlipped.rows * pskFlipped.cols; i++, pskFlippedPtr++) {
    *pskFlippedPtr = (pskSize - (unsigned int) *pskFlippedPtr) % pskSize;
  }
};


void flipPhases(const cv::Mat& phasesSrc, cv::Mat& phasesFlipped) {
  cv::flip(phasesSrc, phasesFlipped, 0);
  double* phasesFlippedPtr = (double*) phasesFlipped.data;
  for (int i = 0; i < phasesFlipped.rows * phasesFlipped.cols; i++, phasesFlippedPtr++) {
    *phasesFlippedPtr = vc_math::wrapAngle(360.0 - *phasesFlippedPtr, 360.0);
  }
};


// Extract the MSB of each slice at F=1Hz (all row entries on 2nd column)
unsigned long long extractSigBits(const cv::Mat& phases, bool flipped, unsigned int pskSize) {
  unsigned long long signature = 0;
  int i;
  double currPhase;
  unsigned long long currBitChunk;

  const unsigned int pskMaxCount = std::pow(2, pskSize);
  const double PSKRange = (360.0/pskMaxCount);
  const double PSKHalfRange = PSKRange/2;
  const unsigned long long sigBitMask = (0b01 << (pskSize - 1));

  if (phases.cols <= 1) throw std::string("Insufficient spectra content for extractSigBits; expecting 2+ columns");

  if (flipped) {
    for (i = 0; i < phases.rows; i++) {
      signature <<= 1;

      currPhase = std::floor(vc_math::wrapAngle(
          phases.at<double>(phases.rows - 1 - i, 1) + PSKHalfRange, 360.0)/PSKRange);
      currBitChunk = (pskMaxCount - (unsigned int) currPhase) % pskMaxCount;

      if ((currBitChunk & sigBitMask) == sigBitMask) {
        signature += 1;
      }
    }
  } else {
    for (i = 0; i < phases.rows; i++) {
      signature <<= 1;

      currPhase = std::floor(vc_math::wrapAngle(
          phases.at<double>(i, 1) + PSKHalfRange, 360.0)/PSKRange);
      currBitChunk = currPhase;

      if ((currBitChunk & sigBitMask) == sigBitMask) {
        signature += 1;
      }
    }
  }

  return signature;
};


bool extractPhasesAndSig(const cv::Mat& img, FTag2Marker& tag, unsigned int numSamplesPerRow, unsigned int sigPskSize) {
  assert(img.channels() == 1);
  const unsigned int MAX_NUM_FREQS = tag.payload.NUM_FREQS();

  img.copyTo(tag.tagImg);

  // Extract rays
  cv::Mat horzRays = extractHorzRays(tag.tagImg, numSamplesPerRow);
  cv::Mat vertRays = extractVertRays(tag.tagImg, numSamplesPerRow);
  if (horzRays.cols < int(MAX_NUM_FREQS*2+1) ||
      vertRays.cols < int(MAX_NUM_FREQS*2+1)) {
    throw std::string("Insufficient number of frequencies in tag spectra");
    return false;
  }

  // Compute magnitude and phase spectra for both horizontal and vertical rays
  cv::Mat flippedRays, fft;
  cv::Mat horzMagSpec, vertMagSpec, horzPhaseSpec, vertPhaseSpec;
  cv::vector<cv::Mat> fftChannels(2);

  cv::dft(horzRays, fft, cv::DFT_ROWS + cv::DFT_COMPLEX_OUTPUT, horzRays.rows);
  cv::split(fft, fftChannels);
  cv::magnitude(fftChannels[0], fftChannels[1], horzMagSpec);
  cv::phase(fftChannels[0], fftChannels[1], horzPhaseSpec, true);

  cv::flip(vertRays, flippedRays, 1);
  cv::dft(flippedRays, fft, cv::DFT_ROWS + cv::DFT_COMPLEX_OUTPUT, flippedRays.rows);
  cv::split(fft, fftChannels);
  cv::magnitude(fftChannels[0], fftChannels[1], vertMagSpec);
  cv::phase(fftChannels[0], fftChannels[1], vertPhaseSpec, true);


  // Extract and validate phase signature, and determine orientation
  const unsigned long long sigKey = tag.payload.SIG_KEY();
  const cv::Range freqSpecRange(1, MAX_NUM_FREQS+1);
  if (extractSigBits(horzPhaseSpec, false, sigPskSize) == sigKey) {
    tag.tagImgCCRotDeg = 0;
    tag.payload.mags = horzMagSpec(cv::Range::all(), freqSpecRange).clone();
    tag.payload.phases = horzPhaseSpec(cv::Range::all(), freqSpecRange).clone();
  } else if (extractSigBits(vertPhaseSpec, false, sigPskSize) == sigKey) {
    tag.tagImgCCRotDeg = 270;
    tag.payload.mags = vertMagSpec(cv::Range::all(), freqSpecRange).clone();
    tag.payload.phases = vertPhaseSpec(cv::Range::all(), freqSpecRange).clone();

  } else if (extractSigBits(horzPhaseSpec, true, sigPskSize) == sigKey) {
    tag.tagImgCCRotDeg = 180;
    cv::flip(horzMagSpec(cv::Range::all(), freqSpecRange), tag.payload.mags, 0);
    flipPhases(horzPhaseSpec(cv::Range::all(), freqSpecRange), tag.payload.phases);
  } else if (extractSigBits(vertPhaseSpec, true, 3) == sigKey) {
    tag.tagImgCCRotDeg = 90;
    cv::flip(vertMagSpec(cv::Range::all(), freqSpecRange), tag.payload.mags, 0);
    flipPhases(vertPhaseSpec(cv::Range::all(), freqSpecRange), tag.payload.phases);
  } else { // No signatures matched
    return false;
  }

  return true;
};


FTag2Marker FTag2Decoder::decodeQuad(const cv::Mat quadImg,
    const Quad& quad,
    int tagType,
    double markerWidthM,
    unsigned int numSamplesPerRow,
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
  FTag2Marker tagBuffer(trimmedTagImg.rows, tagType);

  // Extract rays, decode frequency and phase spectra, and validate signature
  if (!extractPhasesAndSig(croppedTagImg, tagBuffer, numSamplesPerRow, tagBuffer.payload.SIG_PSK_SIZE())) { // NOTE: last argument (3) is the PSK size for the signature frequency
    throw std::string("failed to validate phase signature");
  } else {
    // Update image of tag based on detected orientation
    BaseCV::rotate90(trimmedTagImg, tagBuffer.tagImg, tagBuffer.tagImgCCRotDeg/90);
  }

  // Compute pose of tag
  //tagBuffer.tagCorners.clear(); // redundant
  switch ((tagBuffer.tagImgCCRotDeg/90) % 4) {
  case 1:
    tagBuffer.tagCorners.push_back(quad.corners[1]);
    tagBuffer.tagCorners.push_back(quad.corners[2]);
    tagBuffer.tagCorners.push_back(quad.corners[3]);
    tagBuffer.tagCorners.push_back(quad.corners[0]);
    break;
  case 2:
    tagBuffer.tagCorners.push_back(quad.corners[2]);
    tagBuffer.tagCorners.push_back(quad.corners[3]);
    tagBuffer.tagCorners.push_back(quad.corners[0]);
    tagBuffer.tagCorners.push_back(quad.corners[1]);
    break;
  case 3:
    tagBuffer.tagCorners.push_back(quad.corners[3]);
    tagBuffer.tagCorners.push_back(quad.corners[0]);
    tagBuffer.tagCorners.push_back(quad.corners[1]);
    tagBuffer.tagCorners.push_back(quad.corners[2]);
    break;
  default:
    tagBuffer.tagCorners = quad.corners;
    break;
  }
  solvePose(tagBuffer.tagCorners, markerWidthM,
      cameraIntrinsic, cameraDistortion,
      tagBuffer.pose.position_x, tagBuffer.pose.position_y, tagBuffer.pose.position_z,
      tagBuffer.pose.orientation_w, tagBuffer.pose.orientation_x, tagBuffer.pose.orientation_y,
      tagBuffer.pose.orientation_z);

  // Predict phase variances
  phaseVariancePredictor.predict(&tagBuffer);

  return tagBuffer;
};


void FTag2Decoder::decodePayload(FTag2Payload& tag, double nStdThresh) {
  // WARNING: this function only applies to FTag2MarkerV2
  const cv::Mat& phases = tag.phases;
  const std::vector<double>& phaseVars = tag.phaseVariances;
  const int NUM_RAYS = phases.rows;
  const int NUM_FREQS = phases.cols;
  assert(NUM_FREQS == 5);
  const std::vector<int> bitsPerFreq{3, 3, 3, 2, 2};

  // 0. Reset decoded contents
  tag.bitChunksStr = "";
  tag.numDecodedPhases = 0;
  tag.hasValidXORs = false;
  tag.numDecodedSections = 0;

  // 1. Convert phases to bit chunks
  cv::Mat bitChunks = cv::Mat::ones(NUM_RAYS, NUM_FREQS, CV_8SC1) * -1;

  for (int freq = 0; freq < NUM_FREQS; freq++) {
    double maxBitValue = pow(2, bitsPerFreq[freq]);
    double phaseBinDeg = 360.0/maxBitValue;
    double phaseStdBinNormed = sqrt(phaseVars[freq]) / phaseBinDeg;
    for (int ray = 0; ray < NUM_RAYS; ray++) {
        double phaseBinNormed = vc_math::wrapAngle(phases.at<double>(ray, freq), 360) / phaseBinDeg + 0.5;
        if( freq == 0 ){
            std::cout << "(freq , ray) = " << "( " << freq << ", " << ray << " )" << std::endl;
            std::cout << "PhaseBinNormed = " << phaseBinNormed << std::endl;
            std::cout << "stdThresh = " << nStdThresh << std::endl;
            std::cout << "PhaseStdBinNormed = " << phaseStdBinNormed << std::endl;
        }
      if (floor(phaseBinNormed - nStdThresh*phaseStdBinNormed) == floor(phaseBinNormed + nStdThresh*phaseStdBinNormed)) {
        bitChunks.at<char>(ray, freq) = (unsigned char) phaseBinNormed % (unsigned char) maxBitValue;

        if (freq == 0) {
        int temp = (unsigned char) phaseBinNormed % (unsigned char) maxBitValue;
std::cout << "(ray, freq)" << "(" << ray << ", " << freq << ") = " << temp << "(" << (int) bitChunks.at<char>(ray, freq) << "), phaseBinNormed: " << phaseBinNormed << ", maxBitValue: " << maxBitValue << std::endl;
        }
      }
    }
  }

  // 2. String-ify bit chunks
  std::ostringstream bitChunksStr;
  unsigned int numDecodedPhases = 0;
  char* bitChunksPtr = (char*) bitChunks.data;
  for (int i = 0; i < NUM_RAYS*NUM_FREQS; i++, bitChunksPtr++) {
    if (i % NUM_FREQS == 0 && i > 0) {
      bitChunksStr << "_";
    }
    if (*bitChunksPtr < 0) {
      bitChunksStr << "?";
    } else {
      bitChunksStr << (unsigned short) *bitChunksPtr;
      numDecodedPhases += 1;
    }
  }
  std::cout << cv::format(bitChunks,"matlab") << std::endl;
  tag.bitChunksStr = bitChunksStr.str();
  std::cout << bitChunksStr.str() << std::endl;
  tag.numDecodedPhases = numDecodedPhases;

  // 3. Validate XORs in FTag2MarkerV2 payload structure
  cv::Mat decodedSections = cv::Mat::ones(NUM_RAYS, 2, CV_8SC1) * -1; // -1: missing; -2: xor failed
  for (int ray = 0; ray < NUM_RAYS; ray++) {
    char sigXORBits = bitChunks.at<char>(ray, 0);

    // If XOR chunk is not valid, then cannot validate XORs at all
    if (sigXORBits < 0) {
      return;
    }

    // Decode 2-3Hz payload
    char bitChunk2Hz = bitChunks.at<char>(ray, 1);
    char bitChunk3Hz = bitChunks.at<char>(ray, 2);
    if (bitChunk2Hz >= 0 && bitChunk3Hz >= 0) {
      unsigned char bitChunk23Hz = ((bitChunk2Hz << 3) | bitChunk3Hz) & 0x3F;
      unsigned char greyChunk23Hz = bin2gray(bitChunk23Hz);
      char decodedXOR23Hz = computeXORChecksum(greyChunk23Hz, 6);
      char expectedXOR23Hz = ((sigXORBits & 0x02) == 0x02);
      if (decodedXOR23Hz == expectedXOR23Hz) {
        decodedSections.at<char>(ray, 0) = bitChunk23Hz;
      } else {
        decodedSections.at<char>(ray, 0) = -2;
      }
    }

    // Decode 4-5Hz payload
    char bitChunk4Hz = bitChunks.at<char>(ray, 3);
    char bitChunk5Hz = bitChunks.at<char>(ray, 4);
    if (bitChunk4Hz >= 0 && bitChunk5Hz >= 0) {
      unsigned char bitChunk45Hz = ((bitChunk4Hz << 2) | bitChunk5Hz) & 0x0F;
      unsigned char greyChunk45Hz = bin2gray(bitChunk45Hz);
      char decodedXOR45Hz = computeXORChecksum(greyChunk45Hz, 4);
      char expectedXOR45Hz = ((sigXORBits & 0x01) == 0x01);
      if (decodedXOR45Hz == expectedXOR45Hz) {
        decodedSections.at<char>(ray, 1) = bitChunk45Hz;
      } else {
        decodedSections.at<char>(ray, 1) = -2;
      }
    }
  }

  // 4. String-ify payload sections
  tag.hasValidXORs = true;
  std::ostringstream decodedSectionsStr;
  unsigned int numDecodedSections = 0;
  char* decodedSectionsPtr = (char*) decodedSections.data;
  for (int i = 0; i < NUM_RAYS; i++, decodedSectionsPtr++) {
    if (i > 0) {
      decodedSectionsStr << "_";
    }

    if (*decodedSectionsPtr == -2) {
      decodedSectionsStr << "XX";
    } else if (*decodedSectionsPtr < 0) {
      decodedSectionsStr << "??";
    } else {
      decodedSectionsStr << std::setfill('0') << std::hex << std::uppercase << \
          std::setw(2) << (unsigned short) *decodedSectionsPtr;
      numDecodedSections += 1;
    }

    decodedSectionsStr << ".";
    decodedSectionsPtr++;

    if (*decodedSectionsPtr == -2) {
      decodedSectionsStr << "X";
      tag.hasValidXORs = false;
    } else if (*decodedSectionsPtr < 0) {
      decodedSectionsStr << "?";
    } else {
      decodedSectionsStr << std::hex << std::uppercase << std::setw(1) << \
          (unsigned short) *decodedSectionsPtr;
      numDecodedSections += 1;
    }
  }
  tag.decodedPayloadStr = decodedSectionsStr.str();
  tag.numDecodedSections = numDecodedSections;
};


int FTag2Decoder::davinqiDist(const FTag2Payload& tag1, const FTag2Payload& tag2) {
	int davinqi_dist = 0;
//	std::cout << tag1.bitChunksStr << std::endl;
//	std::cout << tag2.bitChunksStr << std::endl;
	std::string::const_iterator it1 = tag1.bitChunksStr.begin();
	std::string::const_iterator it2 = tag2.bitChunksStr.begin();
	while( it1 != tag1.bitChunksStr.end() && it2 != tag2.bitChunksStr.end() )
	{
		if ( *it1 >= '0' && *it1 < '8' && *it2 >= '0' && *it2 < '8' )
		{
			if ( *it1 != *it2 )
				davinqi_dist++;
		}
		it1++; it2++;
	}
	return davinqi_dist;
}
