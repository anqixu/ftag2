/*
 * PayloadFilter.hpp
 *
 *  Created on: Mar 5, 2014
 *      Author: dacocp
 */

#ifndef PAYLOADFILTER_HPP_
#define PAYLOADFILTER_HPP_

#include "common/FTag2Payload.hpp"

using namespace std;

class PayloadFilter {
private:
	FTag2Payload payload;

public:
	PayloadFilter() {};
	virtual ~PayloadFilter(){};
	void step();
	void step(FTag2Payload tag);
};





#endif /* PAYLOADFILTER_HPP_ */
