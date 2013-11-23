#ifndef BASECV_HPP_
#define BASECV_HPP_


#include "common/VectorAndCircularMath.hpp"


// ---------- MODE FILTER OPTIONS ----------
// This should be disabled in order not to break down the removeIslands algorithm
//#define MODE_FILTER_ZERO_BORDERS

// This should be disabled because border blurring uses smaller mode windows
// and does not produce continuous results, therefore makes the result not
// qualitatively different from not blurring the borders
//#define MODE_FILTER_BLUR_BORDERS

// ---------- STANDALONE NODE IMSHOW FLAGS ----------
// Debug imshow flag
//#define MODE_FILTER_SHOW_FILTER_DIFF

// Debug imshow flag
//#define APPLY_SOBEL_VIEW_MAG

// Debug imshow flag
//#define APPLY_SOBEL_VIEW_OUTPUT

// Debug imshow flag
//#define REMOVE_ISLANDS_SHOW_MERGED_DISJOINT_SETS

// ---------- PARAMETER PROFILE FLAGS ----------
// Accuracy profile flag
//#define SOLVE_LINE_RANSAC_PROFILE

// Accuracy profile flag
//#define CKMEANS_PROFILE

// Accuracy profile flag
//#define MIXED_KMEANS_PROFILE


namespace blob {
typedef std::vector<cv::Point2f> Contour;
typedef std::vector< Contour > ContourVector;
};


extern const cv::Point2f INVALID_POINT;


/**
 * Displays double-typed matrix after rescaling from [0, max] to [0, 255]
 *
 * NOTE: negative values in the original matrix will be saturated to 0
 */
inline void imshow64F(const std::string& winName, const cv::Mat& im) {
  if (im.type() == CV_64FC1) {
    double maxValue;
    cv::Mat im8U;
    cv::minMaxLoc(im, NULL, &maxValue);
    im.convertTo(im8U, CV_8UC1, 255.0/maxValue, 0);
    cv::imshow(winName, im8U);
  } else {
    cv::imshow(winName, im);
  }
};


/**
 * Draws lines parametrized in the Hough space (radius rho from (0, 0), and
 * theta angle from x axis) over provided image
 */
inline void drawLines(cv::Mat img, const std::vector<cv::Point2d> rhoThetas) {
  cv::Point pa, pb;
  double angleHemi;
  for (const cv::Point2d& rhoTheta: rhoThetas) {
    angleHemi = vc_math::wrapAngle(rhoTheta.y, vc_math::pi);
    if (angleHemi > vc_math::half_pi) { angleHemi = vc_math::pi - angleHemi; }
    if (angleHemi > vc_math::half_pi/2) {
      pa.x = -img.cols;
      pa.y = round((rhoTheta.x - cos(rhoTheta.y)*pa.x)/sin(rhoTheta.y));
      pb.x = 2*img.cols;
      pb.y = round((rhoTheta.x - cos(rhoTheta.y)*pb.x)/sin(rhoTheta.y));
    } else {
      pa.y = -img.rows;
      pa.x = round((rhoTheta.x - sin(rhoTheta.y)*pa.y)/cos(rhoTheta.y));
      pb.y = 2*img.rows;
      pb.x = round((rhoTheta.x - sin(rhoTheta.y)*pb.y)/cos(rhoTheta.y));
    }
    cv::line(img, pa, pb, CV_RGB(0, 255, 255), 3);
    cv::line(img, pa, pb, CV_RGB(0, 0, 255), 1);
  }
};


/**
 * Draws line segments (of the form [endA.x, endA.y, endB.x, endB.y] over
 * provided image
 */
inline void drawLineSegments(cv::Mat img, const std::vector<cv::Vec4i> lineSegments) {
  for (const cv::Vec4i& endpts: lineSegments) {
    cv::line(img, cv::Point2i(endpts[0], endpts[1]), cv::Point2i(endpts[2], endpts[3]), CV_RGB(255, 255, 0), 3);
  }
  for (const cv::Vec4i& endpts: lineSegments) {
    cv::line(img, cv::Point2i(endpts[0], endpts[1]), cv::Point2i(endpts[2], endpts[3]), CV_RGB(255, 0, 0), 1);
  }
  for (const cv::Vec4i& endpts: lineSegments) {
    cv::circle(img, cv::Point2i(endpts[0], endpts[1]), 2, CV_RGB(0, 0, 255));
    cv::circle(img, cv::Point2i(endpts[2], endpts[3]), 2, CV_RGB(255, 0, 128));
  }
};


/**
 * This class contains a collection of generic computer vision algorithms,
 * which are implemented as static functions to ensure variable isolation
 * and prevent accidental variable cross-contamination. Nevertheless, since
 * some of the algorithms require potentially large amounts of scratch memory,
 * they are stored in a separate structure (for convenience / singleton-caching),
 * although it means that classes using these caches should take care in
 * preventing access prior to initialization.
 */
class BaseCV {
public:
  /**
   * A seemingly faster implementation of cv::countNonZero, for single-channel
   * unsigned char (8UC1) matrices
   */
  static inline int countNonZero(const cv::Mat& mtx) {
    int count = 0, currY, currX;
    const unsigned char* row;
    for (currY = 0; currY < mtx.rows; currY++) {
      row = mtx.ptr<unsigned char>(currY);
      for (currX = 0; currX < mtx.cols; currX++) {
        if (row[currX] != 0) count++;
      }
    }
    return count;
  };

  /**
   * Applies binary mode (a.k.a. box) filter on binarized image in-place.
   *
   * WARNING: this function expects 0/1 values, so will not work properly with 0/X values!
   *
   * \param binLabelBuffer: input/output 0/1 matrix containing binary labels
   *   to be filtered
   * \param binLabelMask: output matrix containing matrix mask where pixels
   *   have been binarized (since part of border cannot be filtered
   *   due to insufficient box size)
   * \param boxWidthRatio: Box width, computed as the ratio of min(imWidth, imHeight)
   */
  static void applyModeFilter(cv::Mat& binLabelBuffer,
      cv::Mat& binLabelMask,
      double boxWidthRatio = MODE_FILTER_BOX_WIDTH_RATIO);
  constexpr static double MODE_FILTER_BOX_WIDTH_RATIO = 0.05;

  /**
   * Computes disjoint sets for binary-labeled image
   *
   * NOTE: the output disjoint set manifests 1-step parent invariance, i.e.
   *       parentIDs[parentIDs[imageID_ij]] == parentIDs[imageID_ij],
   *       although not necessarily imageID_ij == parentIDs[imageID_ij]
   *
   * NOTE: 8-connected disjoint set may turn out to not be visually "disjoint",
   *       e.g. given a checkerboard-style binary image, there will be 2 "disjoint"
   *       8-connected sets.
   *
   * \param binLabelImage: input binary image
   * \param imageIDs: output matrix containing indices to parentIDs for each pixel
   * \param parentIDs: output matrix containing all disjoint set IDs
   * \param setLabels: output matrix containing set label of each pixel
   * \param eightConnected: if true, then assume set is 8-connected; otherwise
   *   assume that set is 4-connected
   */
  static void computeDisjointSets(
      const cv::Mat& binLabelImage,
      cv::Mat& imageIDs,
      std::vector<unsigned int>& parentIDs,
      std::vector<unsigned char>& setLabels,
      bool eightConnected);

  /**
   * Removes empty-sized labels and updates new contiguous labels for a
   * given disjoint set structure
   *
   * \param imageIDs: input/output matrix containing indices to parentIDs for each pixel
   * \param parentIDs: input/output matrix containing all disjoint set IDs
   * \param setLabels: input/output matrix containing set label of each pixel
   */
  static void condenseDisjointSetLabels(
      cv::Mat& imageIDs,
      std::vector<unsigned int>& parentIDs,
      std::vector<unsigned char>& setLabels,
      unsigned int nStepParentInvariance = 1);

  /**
   * Determines whether 1-step parent invariance is violated
   *
   * \param imageIDs: matrix containing indices to parentIDs for each pixel
   * \param parentIDs: matrix containing all disjoint set IDs
   * \param setLabels: matrix containing set label of each pixel
   */
  static void debugDisjointSetLabels(
      const cv::Mat& imageIDs,
      const std::vector<unsigned int>& parentIDs,
      const std::vector<unsigned char>& setLabels) throw (const std::string&);

  /**
   * Computes disjoint sets, i.e. islands, of connected pixels that all share
   * the same label; then flip the labels for all sets that are not connected
   * to the boundary.
   *
   * binLabelImage will be modified in-place.
   *
   * NOTE: the output disjoint set manifests 1-step parent invariance, i.e.
   *       parentIDs[parentIDs[imageID_ij]] == parentIDs[imageID_ij],
   *       although not necessarily imageID_ij == parentIDs[imageID_ij]
   *
   * NOTE: imageIDs, parentIDs, setLabels will be populated internally
   * NOTE: this function preserves 0-step ID invariance, i.e. setID_ij == parentID[setID_ij]
   */
  static void removeIslands(
      cv::Mat& binLabelImage,
      cv::Mat& imageIDsOutputBuffer,
      std::vector<unsigned int>& parentIDsOutputBuffer,
      std::vector<unsigned char>& setLabelsOutputBuffer);

  /**
   * Merges all disjoint sets whose pixel size (area ratio) is below a
   * certain threshold with their neighbors. Note that this threshold is
   * matched with the original size of each set, so there may be occurrences
   * of multi-merges, where the intermediate-merged set size exceeds minSetRatio.
   *
   * binLabelImage, imageIDs, and parentIDs will be modified in-place.
   *
   * NOTE: the output disjoint set manifests 1-step parent invariance, i.e.
   *       parentIDs[parentIDs[imageID_ij]] == parentIDs[imageID_ij],
   *       although not necessarily imageID_ij == parentIDs[imageID_ij]
   *
   * NOTE: setLabels do not need to be updated since imageIDs and parentIDs
   *       of merged sets will be different, hence the set labels for classes
   *       with previously-merged parent IDs will no longer matter.
   *       Nevertheless, it is crucial to access setLabels via 1-step parent,
   *       rather than with image ID!
   *
   * WARNING: imageIDs, parentIDs, and setLabels MUST be pre-populated properly,
   *          either by calling computeDisjointSets or removeIslands
   */
  static void removeSmallPeninsulas(
      cv::Mat& binLabelImage,
      cv::Mat& imageIDs,
      std::vector<unsigned int>& parentIDs,
      const std::vector<unsigned char>& setLabels,
      double minSetRatio = REMOVE_SMALL_PENINSULAS_MIN_SET_RATIO);
  constexpr static double REMOVE_SMALL_PENINSULAS_MIN_SET_RATIO = 0.025; // Removes peninsulas with sizes smaller than 1/6 width * 1/6 height

  /**
   * Computes the set of pixels 4-/8-connected to a target pixel and which
   * shares the same class label as the target pixel, in a binarized label image.
   *
   * \param binLabelImage: binarized labeled image
   * \param targetPixel: starting pixel
   * \param connectedSet: output buffer containing set of connected pixels
   * \param eightConnected: whether to compute 4-connected or 8-connected
   *   neighbours
   */
  static void findConnectedPixels(
      const cv::Mat& binLabelImage,
      const cv::Point targetPixel,
      std::vector<cv::Point>& connectedSet,
      bool eightConnected = true);

  /** NOTE: binImage should be a binary image within the range [0, 1] */
  /**
   * Applies the Sobel edge detection operator and identifies edge pixels.
   *
   * \param binImage: input binary image within the range [0, 1]
   * \param mask: input pixel selection mask where Sobel operator will be applied on
   * \param edgePoints: output vector containing (x, y) coordinates of edge pixels
   * \param edgelBuffer: optional output matrix containing edgel/non-edgel pixels;
   *   see fillEdgelBuffer
   * \param fillEdgelBuffer: edgeBuffer will only be populated if
   *   fillEdgelBuffer == true
   */
  static void applySobel(
      const cv::Mat& binImage,
      const cv::Mat& mask,
      std::vector<cv::Point>& edgePoints,
      cv::Mat& edgelBuffer,
      bool fillEdgelBuffer = false);
  constexpr static short SOBEL_MAG_SQRD_THRESHOLD = 4; // Valid range: [0, 20]
  // NOTE: SOBEL_MAG_SQRD_THRESHOLD is computed specifically for a binary image,
  //       where the criteria for an edge is either:
  //       - detect sufficiently strong gradient (most cases equal to edge of grayscale image)
  //       or
  //       - a U-shaped 3x3 window (which will generate a 3x3 Sobel magnitude of 4)

  /**
   * Computes a linear fit probabilistically using the RANdom SAmpling Consensus
   * (RANSAC) algorithm.
   *
   * \param points: input set of 2-D candidate points
   * \param line: output buffer containing fitted line; format: [dx, dy, x0, y0]
   * \param baseLineFitCount: minimum number of points needed to fit a line (>= 2)
   * \param maxNumIters: maximum number of RANSAC iterations
   * \param goodFitDist: Euclidean distance threshold criterion used to
   *   identify points that are sufficiently close to candidate line
   * \param candidateFitRatio: minimum ratio of sufficiently-close points needed
   *   to accept a candidate line fit; values between [0, 1]
   * \param termFitRatio: minimum ratio of points needed to accept a refined
   *   line fit and terminate RANSAC; values between [0, 1]
   */
  static double solveLineRANSAC(
      const std::vector<cv::Point>& points,
      cv::Vec4f& line,
      double goodFitDist,
      unsigned int baseLineFitCount = RANSAC_BASE_LINE_FIT_COUNT,
      unsigned int maxNumIters = RANSAC_MAX_NUM_ITERS,
      double candidateFitRatio = RANSAC_CAND_FIT_RATIO,
      double termFitRatio = RANSAC_TERM_FIT_RATIO);
  constexpr static unsigned int RANSAC_BASE_LINE_FIT_COUNT = 8;
  constexpr static unsigned int RANSAC_MAX_NUM_ITERS = 100;
  constexpr static double RANSAC_CAND_FIT_RATIO = 0.25;
  constexpr static double RANSAC_TERM_FIT_RATIO = 0.65;
  constexpr static double RANSAC_LINE_FIT_DIST_TO_IMWIDTH_RATIO = 0.1;
  constexpr static double RANSAC_LINE_FIT_RADIUS_EPS = 0.01;
  constexpr static double RANSAC_LINE_FIT_ANGLE_EPS = 0.01;


  /**
   * Computes maximum likelihood estimation of largest-size blob contour.
   */
  blob::Contour findMLELargestBlobContour(cv::Mat& src);


  /**
   * Computes maximum likelihood estimation of most eccentric blob contour.
   */
  blob::Contour findMLECircularBlobContour(cv::Mat& src);


  /**
   * Computes maximum likelihood estimation circular blob center.
   */
  cv::Point2f findMLECircularBlobCenter(cv::Mat& src, blob::Contour& bestContourBuffer);


  /**
   * Labels grayscale image based on acceptance threshold.
   *
   * \param src: source grayscale image; CV_8UC1, values with [0, 255]
   * \param dst: destination labelled image
   * \param thresh: grayscale threshold, value within [0, 255]
   */
  void applyGrayscaleThresholdClassifier(const cv::Mat& src, cv::Mat& dst, char thresh);


  /**
   * Labels hue image based on hue range acceptance.
   *
   * \param src: source hue image; CV_8UC1, values with [0, 180)
   * \param dst: destination labelled image
   * \param minHueDeg: minimum hue value, in degrees
   * \param maxHueDeg: maximum hue value, in degrees
   * \param medianBlurWidthRatio: post-label median filter for smoothing
   */
  static void applyHueRangeClassifier(const cv::Mat& src, cv::Mat& dst,
      int minHueDeg, int maxHueDeg, double medianBlurWidthRatio);


  /**
   * Computes center of bounding circle of contour.
   */
  static cv::Point2f findBoundingCircleCenter(const blob::Contour& ctr);

  /**
   * Renders a heading arrow over image
   */
  static void drawArrow(cv::Mat& bgrBuffer, double headingDeg,
      cv::Scalar edgeColor, cv::Scalar fillColor, double alphaRatio = 1.0);


#ifdef SOLVE_LINE_RANSAC_PROFILE
  static std::vector<unsigned int> ransac_iters;
  static std::vector<double> ransac_first_fit_ratio;
  static std::vector<double> ransac_second_fit_ratio;
  static std::vector<double> ransac_first_fit_avg_dist;
  static std::vector<double> ransac_second_fit_avg_dist;
  static unsigned int ransac_term_via_iters;
  static unsigned int ransac_term_via_fit;
#endif
};


#endif /* BASECV_HPP_ */
