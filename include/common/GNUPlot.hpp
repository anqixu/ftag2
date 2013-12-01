#ifndef GNUPLOT_HPP_
#define GNUPLOT_HPP_


#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>


namespace gp {


#define MAX_NUM_GNUPLOTS 3

extern FILE* gnuplotPipe[MAX_NUM_GNUPLOTS];
extern bool gnuplotPipeInit;


void initGNUPlot();

void termGNUPlot();

void plot(const std::vector<cv::Point2d>& pts, unsigned int figI = 0,
    std::string title = "");

void bar(const std::vector<double>& pts, unsigned int figI = 0,
    std::string title = "");

};


#endif /* GNUPLOT_HPP_ */
