#ifndef PROFILER_HPP_
#define PROFILER_HPP_


#include <boost/date_time/posix_time/posix_time.hpp>
#include <mutex>
#include <utility>
#include <string>
#include <fstream>


/**
 * NOTE: there is an intrinsic ~2us duration between tic() and toc() function calls
 */
class Profiler {
public:
  Profiler() : count(0), totalTime(0), totalTimeSqrd(0),
      ticTime(boost::posix_time::not_a_date_time) {
  };

  ~Profiler() {
  };

  void reset() {
    timeMutex.lock();
    count = 0;
    totalTime = 0;
    totalTimeSqrd = 0;
    ticTime = boost::posix_time::not_a_date_time;
    timeMutex.unlock();
  };

  void tic() {
    timeMutex.lock();
    ticTime = boost::posix_time::microsec_clock::universal_time();
    timeMutex.unlock();
  };

  void toc() throw (const std::string&) {
    if (ticTime != boost::posix_time::not_a_date_time) {
      timeMutex.lock();
      double td = double((boost::posix_time::microsec_clock::universal_time() - ticTime).total_microseconds()) / 1.0e6;
      count += 1;
      totalTime += td;
      totalTimeSqrd += td*td;
      ticTime = boost::posix_time::not_a_date_time;
      timeMutex.unlock();
    } else {
      throw std::string("WARNING: Profiler::tic() not called");
    }
  };

  inline void try_toc() {
    try { toc(); } catch (const std::string& err) {};
  };

  unsigned long long getCount() { return count; };

  std::pair<double, double> getAvgAndStdTime() {
    if (count <= 0) {
      //throw std::string("WARNING: Profiler::tic() / Profiler::toc() not called");
      return std::pair<double, double>(-1, -1);
    }
    timeMutex.lock();
    double avgTime = totalTime/count;
    double stdTime = totalTimeSqrd/count - avgTime*avgTime;
    std::pair<double, double> result = std::make_pair(avgTime, stdTime);
    timeMutex.unlock();
    return result;
  };

  std::string getStatsString() {
    if (count > 0) {
      std::ostringstream oss;
      oss.width(3);
      oss << count << " entries - ";
      std::pair<double, double> avgAndStd = getAvgAndStdTime();
      oss.unsetf(std::ios::floatfield);
      oss.precision(3);
      oss << avgAndStd.first*1e3 << " +/- " << avgAndStd.second*1e3 << " ms";
      return oss.str();
    } else {
      return "no entries";
    }
  };

protected:
  unsigned long long count;
  double totalTime;
  double totalTimeSqrd;
  boost::posix_time::ptime ticTime;
  std::mutex timeMutex;
};


#endif /* PROFILER_HPP_ */
