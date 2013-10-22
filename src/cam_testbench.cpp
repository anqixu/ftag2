#include "detector/FTag2Detector.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "common/Profiler.hpp"
#include <chrono>


//#define SAVE_IMAGES_FROM edgeImg
#define ENABLE_PROFILER


using namespace std;
using namespace std::chrono;
using namespace cv;


int main(int argc, char** argv) {
  try {
    // Initialize parameters
    int sobelThreshHigh = 100;
    int sobelThreshLow = 30;
    int sobelBlurWidth = 3;
    //int threshold = 50, linbinratio = 100, angbinratio = 100;

    // Initialize internal variables
    bool alive = true;
    cv::Mat bgrImg, grayImg, blurredImg, edgeImg, linesImg;
    std::vector<cv::Vec4i> lines;
#ifdef SAVE_IMAGES_FROM
    int imgid = 0;
    char* filename = (char*) calloc(1000, sizeof(char));
#endif
#ifdef ENABLE_PROFILER
    Profiler durationProf, rateProf;
    time_point<system_clock> lastProfTime = system_clock::now();
    time_point<system_clock> currTime;
    duration<double> profTD;
#endif

    // Open camera
    VideoCapture cam(0);
    if (!cam.isOpened()) {
      cerr << "OpenCV did not detect any cameras." << endl;
      return EXIT_FAILURE;
    }


    // Manage OpenCV windows
    //namedWindow("source", CV_GUI_EXPANDED);

    //namedWindow("edge", CV_GUI_EXPANDED);
    //createTrackbar("sobelThreshLow", "edge", &sobelThreshLow, 255);
    //createTrackbar("sobelThreshHigh", "edge", &sobelThreshHigh, 255);

    //namedWindow("lines", CV_GUI_EXPANDED);
    //createTrackbar("threshold", "lines", &threshold, 255);
    //createTrackbar("linbinratio", "lines", &linbinratio, 100);
    //createTrackbar("angbinratio", "lines", &angbinratio, 100);


    // Main camera pipeline
    alive = true;
    while (alive) {
#ifdef ENABLE_PROFILER
      rateProf.try_toc();
      rateProf.tic();
      durationProf.tic();
#endif

      // Obtain grayscale image
      cam >> bgrImg;
      cvtColor(bgrImg, grayImg, CV_BGR2GRAY);
      //imshow("source", grayImg);

      // Compute edge
      // TODO: tune params: blur==3 removes most unwanted edges but fails when tag is moving; blur==5 gets more spurious edges in general but detects tag boundary when moving
      blur(grayImg, blurredImg, Size(sobelBlurWidth, sobelBlurWidth));
      Canny(blurredImg, edgeImg, sobelThreshLow, sobelThreshHigh, sobelBlurWidth); // args: in, out, low_thresh, high_thresh, gauss_blur
      imshow("edge", edgeImg);

      /*
      // Detect lines
      cv::HoughLinesP(edgeImg, lines, max(linbinratio/10.0, 1.0), max(angbinratio/100.0*CV_PI/180, 1/100.0*CV_PI/180), threshold, 10*1.5, 10*1.5/3);
      //cv::HoughLines(edgeImg, lines, max(linbinratio/10.0, 1.0), max(angbinratio/100.0*CV_PI/180, 1/100.0*CV_PI/180), threshold);
      cvtColor(edgeImg, linesImg, CV_GRAY2BGR);
      for (unsigned int i = 0; i < lines.size(); i++) {
        line(linesImg, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(0,255,0), 2, 8);
      }
      imshow("lines", linesImg);
      */

#ifdef SAVE_IMAGES_FROM
      sprintf(filename, "img%05d.jpg", imgid++);
      imwrite(filename, SAVE_IMAGES_FROM);
      cout << "Wrote to " << filename << endl;
#endif

#ifdef ENABLE_PROFILER
      durationProf.toc();

      currTime = system_clock::now();
      profTD = currTime - lastProfTime;
      if (profTD.count() > 1) {
        cout << "Pipeline Duration: " << durationProf.getStatsString() << endl;
        cout << "Pipeline Rate: " << rateProf.getStatsString() << endl;
        lastProfTime = currTime;
      }
#endif

      // Process displays
      if (waitKey(30) >= 0) { break; }
    }
  } catch (const Exception& err) {
    cout << "CV Exception: " << err.what() << endl;
  }

  return EXIT_SUCCESS;
};
