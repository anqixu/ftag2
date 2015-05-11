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


void filterMagnitudePoly(cv::Mat mags, double gainNeg, double gainPos, double powNeg, double powPos) {
  double* currMagRow;
  double mag1Hz;
  double normMag;
  std::vector<double> lowerThresh(5, 1.0);
  std::vector<double> upperThresh(5, 1.0);

  for (int f = 1; f < mags.cols; f++) { // f == 1 represents 2Hz
    lowerThresh[f] = 1.0 - gainNeg*pow(f, powNeg);
    upperThresh[f] = 1.0 + gainPos*pow(f, powPos);
  }

  for (int i = 0; i < mags.rows; i++) {
    currMagRow = (double*) mags.ptr<double>(i);
    mag1Hz = *currMagRow;
    if (mag1Hz == 0) {
      throw std::string("failed polynomial magnitude filter (1Hz mag == 0)");
    }
    currMagRow++;

    for (int j = 1; j < mags.cols; j++, currMagRow++) {
      normMag = *currMagRow/mag1Hz;
      if (normMag < lowerThresh[j] || normMag > upperThresh[j]) {
        std::ostringstream oss;
        oss << "failed to pass through polynomial magnitude filter (" << j+1 <<
            "Hz: " << normMag << " out of bounds [" << lowerThresh[j] << ", " <<
            upperThresh[j] << ")";
        throw oss.str();
      }
    }
  }
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


// Extract the LSB of each slice at F=1Hz
unsigned long long extractSigBits(const cv::Mat& phases, bool flipped, unsigned int pskSize) {
  unsigned long long signature = 0;
  int i;
  double currPhase;
  unsigned long long currBitChunk;

  const unsigned int pskMaxCount = std::pow(2, pskSize);
  const double PSKRange = (360.0/pskMaxCount);
  const double PSKHalfRange = PSKRange/2;
  const unsigned long long sigBitMask = 0b01;

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


void extractPhasesAndSigWithMagFilter(const cv::Mat& img, FTag2Marker& tag,
    unsigned int numSamplesPerRow, unsigned int sigPskSize,
    double magFilGainNeg, double magFilGainPos, double magFilPowNeg, double magFilPowPos) {
  assert(numSamplesPerRow > 0);
  assert(img.channels() == 1);
  const unsigned int MAX_NUM_FREQS = tag.payload.NUM_FREQS();

  img.copyTo(tag.tagImg);

  // Extract rays
  cv::Mat horzRays = extractHorzRays(tag.tagImg, numSamplesPerRow);
  cv::Mat vertRays = extractVertRays(tag.tagImg, numSamplesPerRow);
  if (horzRays.cols < int(MAX_NUM_FREQS*2+1) ||
      vertRays.cols < int(MAX_NUM_FREQS*2+1)) {
    throw std::string("Insufficient number of frequencies in tag spectra");
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

  // Check for phase signature and apply magnitude filter to all 4 quad orientations
  //
  // 1. identify all 90-rotated orientations containing the phase signature
  // 2. for each candidate orientation, compute the normalized standard
  //    deviation in magnitudes across S slices for each given freq, and average
  //    these normalized standard deviations across frequencies; i.e.
  //    avg_freq( std_slice(mag)/avg_slice(mag) )
  // 3. accept the orientation with the smallest frequency-averaged across-slice
  //    normalized standard deviations in magnitude, i.e. the orientation
  //    whose magnitude matrix is most consistent across slices, while
  //    accounting for frequency-dependent drop-offs
  const unsigned long long sigKey = tag.payload.SIG_KEY();
  const cv::Range freqSpecRange(1, MAX_NUM_FREQS+1);
  const double INF = std::numeric_limits<double>::infinity();
  std::string sigCheckErr[4];
  double sliceMagNormStd[4];
  cv::Scalar tempMean, tempStdev;
  cv::Mat tempMag;
  bool foundSig[4];
  for (int rot90 = 0; rot90 < 4; rot90++) {
    sliceMagNormStd[rot90] = INF;
    sigCheckErr[rot90] = "";

    if (rot90 == 1) {
      foundSig[rot90] = (extractSigBits(vertPhaseSpec, true, sigPskSize) == sigKey);
      cv::flip(vertMagSpec(cv::Range::all(), freqSpecRange), tempMag, 0);
    } else if (rot90 == 2) {
      foundSig[rot90] = (extractSigBits(horzPhaseSpec, true, sigPskSize) == sigKey);
      cv::flip(horzMagSpec(cv::Range::all(), freqSpecRange), tempMag, 0);
    } else if (rot90 == 3) {
      foundSig[rot90] = (extractSigBits(vertPhaseSpec, false, sigPskSize) == sigKey);
      tempMag = vertMagSpec(cv::Range::all(), freqSpecRange).clone(); // NOTE: clone() important due to subsequent in-place ops
    } else { // rot90 == 0
      foundSig[rot90] = (extractSigBits(horzPhaseSpec, false, sigPskSize) == sigKey);
      tempMag = horzMagSpec(cv::Range::all(), freqSpecRange).clone(); // NOTE: clone() important due to subsequent in-place ops
    }

    if (foundSig[rot90]) {
      try {
        filterMagnitudePoly(tempMag, magFilGainNeg, magFilGainPos, magFilPowNeg, magFilPowPos);

        sliceMagNormStd[rot90] = 0;
        for (unsigned int freq = 1; freq <= MAX_NUM_FREQS; freq++) {
          cv::meanStdDev(tempMag(cv::Range::all(), cv::Range(freq-1, freq)), tempMean, tempStdev);
          if (tempMean[0] > 0) {
            sliceMagNormStd[rot90] += tempStdev[0]/tempMean[0];
          }
        }
        sliceMagNormStd[rot90] /= MAX_NUM_FREQS;

      } catch (const std::string& err) {
        sigCheckErr[rot90] = err;
      }
    } else {
      sigCheckErr[rot90] = "no phase signature";
    }
  }

  // Identify orientation that passed magnitude+phase signature,
  // and has the smallest magnitude variance (i.e. least likely to be due to
  // random signal)
  tag.tagImgCCRotDeg = -1;
  double tempMinNormStd = INF;
  for (int rot90 = 0; rot90 < 4; rot90++) {
    if (sliceMagNormStd[rot90] < tempMinNormStd) {
      tempMinNormStd = sliceMagNormStd[rot90];
      tag.tagImgCCRotDeg = rot90*90;
    }
  }

  if (tag.tagImgCCRotDeg == 0) {
    tag.payload.mags = horzMagSpec(cv::Range::all(), freqSpecRange).clone();
    tag.payload.phases = horzPhaseSpec(cv::Range::all(), freqSpecRange).clone();
  } else if (tag.tagImgCCRotDeg == 90) {
    cv::flip(vertMagSpec(cv::Range::all(), freqSpecRange), tag.payload.mags, 0);
    flipPhases(vertPhaseSpec(cv::Range::all(), freqSpecRange), tag.payload.phases);
  } else if (tag.tagImgCCRotDeg == 180) {
    cv::flip(horzMagSpec(cv::Range::all(), freqSpecRange), tag.payload.mags, 0);
    flipPhases(horzPhaseSpec(cv::Range::all(), freqSpecRange), tag.payload.phases);
  } else if (tag.tagImgCCRotDeg == 270) {
    tag.payload.mags = vertMagSpec(cv::Range::all(), freqSpecRange).clone();
    tag.payload.phases = vertPhaseSpec(cv::Range::all(), freqSpecRange).clone();
  } else {
    // Verbose error
    /*
    std::ostringstream oss;
    oss << "phase+mag sig filter failed:" << std::endl <<
        "  0-rot: " << sigCheckErr[0] << std::endl <<
        " 90-rot: " << sigCheckErr[1] << std::endl <<
        "180-rot: " << sigCheckErr[2] << std::endl <<
        "270-rot: " << sigCheckErr[3] << std::endl;
    throw oss.str();
    */

    // Sparse error
    ///*
    for (int rot90 = 0; rot90 < 4; rot90++) {
      if (foundSig[rot90]) throw sigCheckErr[rot90];
    }
    throw sigCheckErr[0];
    //*/
  }
};


FTag2Marker decodeQuad(const cv::Mat quadImg,
    const Quad& quad,
    int tagType,
    double markerWidthM,
    unsigned int numSamplesPerRow,
    const cv::Mat cameraIntrinsic, const cv::Mat cameraDistortion,
    double quadMaxStripAvgDiff,
    double tagBorderMeanMaxThresh, double tagBorderStdMaxThresh,
    double magFilGainNeg, double magFilGainPos,
    double magFilPowNeg, double magFilPowPos,
    PhaseVariancePredictor& phaseVariancePredictor) {
  // Trim tag borders
  cv::Mat trimmedTagImg = trimFTag2Quad(quadImg, quadMaxStripAvgDiff);

  // Validate marker borders
  // NOTE: function will throw std::string error if failed
  validateTagBorder(trimmedTagImg, tagBorderMeanMaxThresh, tagBorderStdMaxThresh);

  // Initialize tag data structure
  FTag2Marker tagBuffer(tagType, trimmedTagImg.rows);

  // Extract rays, decode frequency and phase spectra, and validate signature
  std::vector<unsigned int> bitsPerFreq = tagBuffer.payload.BITS_PER_FREQ();
  if (bitsPerFreq.size() < 1) {
    throw std::string("INTERNAL ERROR: could not identify tag's bits per freq due to unexpected type");
  }
  unsigned int sigPSKSize = bitsPerFreq[0];
  // NOTE: function will throw std::string error if failed
  extractPhasesAndSigWithMagFilter(trimmedTagImg, tagBuffer,
      numSamplesPerRow, sigPSKSize,
      magFilGainNeg, magFilGainPos, magFilPowNeg, magFilPowPos);

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


void decodePayload(FTag2Payload& tag, double nStdThresh) {
  const cv::Mat& phases = tag.phases;
  const std::vector<double>& phaseVars = tag.phaseVariances;
  const int NUM_RAYS = phases.rows;
  const int NUM_FREQS = phases.cols;
  const std::vector<unsigned int> bitsPerFreq = tag.BITS_PER_FREQ();

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
      if (floor(phaseBinNormed - nStdThresh*phaseStdBinNormed) == floor(phaseBinNormed + nStdThresh*phaseStdBinNormed)) {
        bitChunks.at<char>(ray, freq) = (unsigned char) phaseBinNormed % (unsigned char) maxBitValue;
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
  tag.bitChunksStr = bitChunksStr.str();
  tag.numDecodedPhases = numDecodedPhases;

  // 3. Convert bit chunks to type-specific payload strings
  if (tag.type == FTag2Payload::FTAG2_6S5F3B) {
    tag.hasValidXORs = true;
    tag.decodedPayloadStr = tag.bitChunksStr;
    tag.numDecodedSections = tag.numDecodedPhases;


  } else if (tag.type == FTag2Payload::FTAG2_6S5F33322B) { // Special type: has defined XORs
#ifdef DEPRECATED_CODE
    // 3.1 Validate XORs in FTag2MarkerV2 payload structure
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

    // 3.2 String-ify payload sections
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
#else
    // TODO: 0 update the decoding logic for 6s5f33322b
    tag.hasValidXORs = true;
    tag.decodedPayloadStr = tag.bitChunksStr;
    tag.numDecodedSections = tag.numDecodedPhases;


#endif

  } else if (tag.type == FTag2Payload::FTAG2_6S5F33222B) {
    tag.hasValidXORs = true;
    tag.decodedPayloadStr = tag.bitChunksStr;
    tag.numDecodedSections = tag.numDecodedPhases;


  } else if (tag.type == FTag2Payload::FTAG2_6S5F22111B) {
    tag.hasValidXORs = true;
    tag.decodedPayloadStr = tag.bitChunksStr;
    tag.numDecodedSections = tag.numDecodedPhases;


  } else if (tag.type == FTag2Payload::FTAG2_6S2F21B) {
    tag.hasValidXORs = true;
    tag.decodedPayloadStr = tag.bitChunksStr;
    tag.numDecodedSections = tag.numDecodedPhases;


  } else if (tag.type == FTag2Payload::FTAG2_6S2F22B) {
    tag.hasValidXORs = true;
    tag.decodedPayloadStr = tag.bitChunksStr;
    tag.numDecodedSections = tag.numDecodedPhases;


  } else if (tag.type == FTag2Payload::FTAG2_6S3F211B) {
    tag.hasValidXORs = true;
    tag.decodedPayloadStr = tag.bitChunksStr;
    tag.numDecodedSections = tag.numDecodedPhases;


  } else { // Unexpected type
    std::ostringstream oss;
    oss << "Unknown tag type: " << tag.type;
    tag.decodedPayloadStr = oss.str();
    tag.numDecodedSections = 0;
  }
};
