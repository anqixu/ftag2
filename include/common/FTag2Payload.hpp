#ifndef FTAG2PAYLOAD_HPP_
#define FTAG2PAYLOAD_HPP_

#include "common/BaseCV.hpp"

struct FTag2Payload {
	constexpr static unsigned long long SIG_KEY = 0b00100011;
	constexpr static unsigned int MAX_NUM_FREQS = 5;

	std::vector<double> phaseVariances;

	bool hasSignature;

	std::string payloadOct;
	std::string payloadBin;

	cv::Mat payloadChunks;

	// TODO: 2 remove TEMP VARS
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
	cv::Mat bitChunks; // TEMP: 2 bitChunks == graycoded phases; payloadChunks = de-graycoded phases

	FTag2Payload (): phaseVariances(), hasSignature(false),
			payloadOct(""), payloadBin(""), signature(0) {
		for (int i = 0; i < 5; i++) { phaseVariances.push_back(0); }
		    payloadChunks = cv::Mat::ones(6, 5, CV_8SC1) * -1;
	};
	~FTag2Payload() {};

	FTag2Payload(const FTag2Payload& other) :
	  phaseVariances(other.phaseVariances),
	  hasSignature(other.hasSignature),
    payloadOct(other.payloadOct),
    payloadBin(other.payloadBin),
    payloadChunks(other.payloadChunks.clone()),
    signature(other.signature),
    horzRays(other.horzRays.clone()),
    vertRays(other.vertRays.clone()),
    horzMagSpec(other.horzMagSpec.clone()),
    vertMagSpec(other.vertMagSpec.clone()),
    horzPhaseSpec(other.horzPhaseSpec.clone()),
    vertPhaseSpec(other.vertPhaseSpec.clone()),
    horzMags(other.horzMags.clone()),
    vertMags(other.vertMags.clone()),
    horzPhases(other.horzPhases.clone()),
    vertPhases(other.vertPhases.clone()),
    mags(other.mags.clone()),
    phases(other.phases.clone()),
    bitChunks(other.bitChunks.clone()) {};

	void operator=(const FTag2Payload& other) {
	  phaseVariances = other.phaseVariances;
    hasSignature = other.hasSignature;
    payloadOct = other.payloadOct;
    payloadBin = other.payloadBin;
    payloadChunks = other.payloadChunks.clone();
    signature = other.signature;
    horzRays = other.horzRays.clone();
    vertRays = other.vertRays.clone();
    horzMagSpec = other.horzMagSpec.clone();
    vertMagSpec = other.vertMagSpec.clone();
    horzPhaseSpec = other.horzPhaseSpec.clone();
    vertPhaseSpec = other.vertPhaseSpec.clone();
    horzMags = other.horzMags.clone();
    vertMags = other.vertMags.clone();
    horzPhases = other.horzPhases.clone();
    vertPhases = other.vertPhases.clone();
    mags = other.mags.clone();
    phases = other.phases.clone();
    bitChunks = other.bitChunks.clone();
	};

	bool withinPhaseRange(FTag2Payload& marker);

	double sumOfStds() {
		double sum = 0.0;
		for ( double d : phaseVariances )
			sum += sqrt(d);
		return sum;
	}


	virtual void decodeSignature() {};
};



#endif /* FTAG2PAYLOAD_HPP_ */
