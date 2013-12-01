#include "common/GNUPlot.hpp"


namespace gp {


FILE* gnuplotPipe[MAX_NUM_GNUPLOTS];
bool gnuplotPipeInit = false;


void initGNUPlot() {
  if (!gnuplotPipeInit) {
    for (int i = 0; i < MAX_NUM_GNUPLOTS; i++) {
      gnuplotPipe[i] = popen("gnuplot -persistent", "w");
    }
    gnuplotPipeInit = true;
  }
};


void termGNUPlot() {
  if (gnuplotPipeInit) {
    for (int i = 0; i < MAX_NUM_GNUPLOTS; i++) {
      pclose(gnuplotPipe[i]);
    }
    gnuplotPipeInit = false;
  }
};


void plot(const std::vector<cv::Point2d>& pts, unsigned int figI,
    std::string title) {
  if (figI >= MAX_NUM_GNUPLOTS) {
    std::cerr << "WARNING: cannot plot on Fig. " << figI << "; only " <<
        MAX_NUM_GNUPLOTS << " were allocated" << std::endl;
    return;
  }
  initGNUPlot();

  std::ostringstream filename;
  filename << "/tmp/gnuplot.data" << figI;
  FILE* temp = fopen(filename.str().c_str(), "w");
  for (const cv::Point2d& pt: pts) {
    std::fprintf(temp, "%.15lf %.15lf\n", pt.x, pt.y);
  }
  fclose(temp);

  FILE* fig = gnuplotPipe[figI];
  std::fprintf(fig, "clear \n");
  std::fprintf(fig, "set autoscale \n");
  std::fprintf(fig, "set xtic auto \n");
  std::fprintf(fig, "set ytic auto \n");
  std::fprintf(fig, "plot '%s' \n", filename.str().c_str());
  std::fprintf(fig, "unset label \n");
  if (title.size() > 0) {
    std::fprintf(fig, "set title '%s' \n", title.c_str());
  } else {
    std::fprintf(fig, "set title 'Fig. %d' \n", figI+1);
  }
  std::fflush(fig);
};


void bar(const std::vector<double>& pts, unsigned int figI,
    std::string title) {
  if (figI >= MAX_NUM_GNUPLOTS) {
    std::cerr << "WARNING: cannot plot on Fig. " << figI << "; only " <<
        MAX_NUM_GNUPLOTS << " were allocated" << std::endl;
    return;
  }
  initGNUPlot();

  std::ostringstream filename;
  filename << "/tmp/gnuplot.data" << figI;
  FILE* temp = fopen(filename.str().c_str(), "w");
  unsigned int i;
  double x, xIncr = 1.0/(pts.size()-1);
  for (i = 0, x = 0; i < pts.size(); i++, x += xIncr) {
    std::fprintf(temp, "%.15lf %.15lf\n", x, pts[i]);
  }
  fclose(temp);

  FILE* fig = gnuplotPipe[figI];
  std::fprintf(fig, "clear \n");
  std::fprintf(fig, "set autoscale \n");
  std::fprintf(fig, "set xtic auto \n");
  std::fprintf(fig, "set ytic auto \n");
  std::fprintf(fig, "set xrange [0:1] \n");
  std::fprintf(fig, "set yrange [0:255] \n");
  std::fprintf(fig, "plot '%s' with impulses \n", filename.str().c_str());
  std::fprintf(fig, "unset label\n");
  if (title.size() > 0) {
    std::fprintf(fig, "set title '%s' \n", title.c_str());
  } else {
    std::fprintf(fig, "set title 'Fig. %d' \n", figI+1);
  }
  std::fflush(fig);
};


};
