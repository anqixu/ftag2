#ifndef FTAG2_HPP_
#define FTAG2_HPP_


#include <opencv2/core/core.hpp>


struct FTag2 {
  unsigned int ID;

  std::vector<cv::Mat> rays;
  cv::Mat img;
};


#endif /* FTAG2_HPP_ */
