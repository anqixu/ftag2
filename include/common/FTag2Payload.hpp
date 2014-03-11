/*
 * FTag2Payload.hpp
 *
 *  Created on: Mar 6, 2014
 *      Author: dacocp
 */

#ifndef FTAG2PAYLOAD_HPP_
#define FTAG2PAYLOAD_HPP_

#include "common/BaseCV.hpp"
#include <opencv2/core/core.hpp>

struct FTag2Payload {
	constexpr static unsigned long long SIG_KEY = 0b00100011;
	constexpr static unsigned int MAX_NUM_FREQS = 5;

	std::vector<double> phaseVariances;

	bool hasSignature;
	bool hasValidXORs;

	std::string payloadOct;
	std::string payloadBin;
	std::string xorBin; // == same data as XORExpected

	cv::Mat XORExpected;
	cv::Mat XORDecoded;
	cv::Mat payloadChunks;

	// TEMP VARS
	unsigned long long signature;

	cv::Mat horzRays;
	cv::Mat vertRays;
	cv::Mat horzMagSpec;
	cv::Mat vertMagSpec;
	cv::Mat horzPhaseSpec;
	cv::Mat vertPhaseSpec;

	cv::Mat horzMags;
	cv::Mat vertMags;
	cv::Mat horzPhases;
	cv::Mat vertPhases;

	cv::Mat mags;
	cv::Mat phases;
	cv::Mat bitChunks;

	FTag2Payload (): phaseVariances(), hasSignature(false), hasValidXORs(false),
			payloadOct(""), payloadBin(""), xorBin(""), signature(0) {
		for (int i = 0; i < 5; i++) { phaseVariances.push_back(0); }
		    payloadChunks = cv::Mat::ones(6, 5, CV_8SC1) * -1;
	};

	virtual ~FTag2Payload() {};
	bool withinPhaseRange(const FTag2Payload& marker);

	double sumOfStds() {
		double sum = 0.0;
		for ( double d : phaseVariances )
			sum += sqrt(d);
		return sum;
	}


	virtual void decodeSignature() {};
};



#endif /* FTAG2PAYLOAD_HPP_ */
