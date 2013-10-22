#include "common/Timer.hpp"


double timeSinceEpoch(boost::posix_time::ptime t) {
  boost::posix_time::ptime epoch(boost::gregorian::date(1970, 1, 1));
  return (t - epoch).total_microseconds() / 1.0e6;
};
