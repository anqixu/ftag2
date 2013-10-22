#ifndef VECTORANDCIRCULARMATH_HPP_
#define VECTORANDCIRCULARMATH_HPP_


#include "common/CircularStatistics.hpp"
#include <boost/math/constants/constants.hpp>
#include <opencv2/opencv.hpp>
#include <utility>
#include <vector>
#include <cmath>


namespace vc_math {


//constexpr double degree = boost::math::constants::pi<double>()/180.0;
//constexpr double radian = 180.0/boost::math::constants::pi<double>();
//constexpr double two_pi = boost::math::constants::pi<double>()*2;
constexpr double degree = 3.1415926535897932384626433832795/180.0;
constexpr double radian = 180.0/3.1415926535897932384626433832795;
constexpr double two_pi = 3.1415926535897932384626433832795*2;
constexpr double half_pi = 3.1415926535897932384626433832795/2;
constexpr double INVALID_ANGLE = 361.0;


/**
 * Generates a pair of random samples from the unit Normal distribution,
 * using the Box-Mueller method.
 */
inline std::pair<double, double> randn() {
  double U = double(rand())/RAND_MAX;
  double V = double(rand())/RAND_MAX;
  double sqrtMinusTwolnU = sqrt(-2*log(U));
  double TwoPiV = two_pi * V;
  return std::pair<double, double>(sqrtMinusTwolnU * cos(TwoPiV), sqrtMinusTwolnU * sin(TwoPiV));
};

/**
 * Computes angular (and general modulo-'range') magnitude
 */
inline double angularMag(
    double a,
    double b,
    double range = 360.0) {
  double d = b - a + range/2;
  d = (d > 0) ? d - floor(d/range)*range - range/2 : d - (floor(d/range) + 1)*range + range/2;
  return d;
};

/**
 * Computes angular (and general modulo-'range') distance
 */
inline double angularDist(double a, double b, double range = 360.0) { return fabs(angularMag(a, b, range)); };

/**
 * Computes Eulidean distance between 2 2-D points
 */
inline double dist(double x1, double y1, double x2, double y2) {
  return sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));
};

/**
 * Wraps angle in degrees to [0, 360) range
 */
inline constexpr double wrapAngle(double angleDeg) {
  return (angleDeg - floor(angleDeg/360.0)*360.0);
};

/**
 * Count number of entries in matrix that are not equal
 *
 * NOTE: if Mat represents a colored image, then it will count different
 * color channels separately.
 */
inline unsigned long long countNotEqual(const cv::Mat& a, const cv::Mat& b) {
  if (a.size != b.size || a.channels() != b.channels()) {
    return std::max(a.rows * a.cols * a.channels(), b.rows * b.cols * b.channels());
  }

  cv::Mat not_equal;
  cv::compare(a, b, not_equal, cv::CMP_NE);
  cv::Scalar sum_not_equal = sum(sum(not_equal) / 255);
  return sum_not_equal[0] + sum_not_equal[1] + sum_not_equal[2] + sum_not_equal[3];
};

inline unsigned long long countNotEqual(const std::vector<cv::Point>& a,
    const std::vector<cv::Point>& b) {
  unsigned long long count = abs(a.size() - b.size());
  size_t minSize = std::min(a.size(), b.size());
  std::vector<cv::Point>::const_iterator itA = a.begin();
  std::vector<cv::Point>::const_iterator itB = b.begin();
  for (unsigned int i = 0; i < minSize; i++, itA++, itB++) {
    if (*itA != *itB) {
      count += 1;
    }
  }
  return count;
};

/**
 * Computes the Euclidean distance between a 2-D point and a 2-D line
 */
inline double distPointLine(
    const cv::Point& point,
    const cv::Vec4f& currFit) {
  // http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/
  double vx = currFit[0], vy = currFit[1], x1 = currFit[2], y1 = currFit[3];
  if (vx == 0 && vy == 0) {
    return 0; // P1 = P2!
  }
  double x3 = point.x, y3 = point.y;
  double u = ((x3-x1)*vx + (y3-y1)*vy)/(vx*vx+vy*vy);
  double dx = x1 + u*vx - x3, dy = y1 + u*vy - y3;
  return sqrt(dx*dx + dy*dy);
};

/**
 * Finds closest point to given line in given set of points.
 */
inline cv::Point findClosestPointToLine(
    const std::vector<cv::Point>& points,
    const cv::Vec4f& line) {
  // Identify the 8-connected set of edgels closest to the previous line fit
  std::vector<cv::Point>::const_iterator itPoints = points.begin();
  std::vector<cv::Point>::const_iterator itPointsEnd = points.end();
  double closestDist = distPointLine(*itPoints, line);
  cv::Point closestPoint = *itPoints;
  double currDist;
  for (itPoints++; itPoints != itPointsEnd; itPoints++) {
    currDist = distPointLine(*itPoints, line);
    if (currDist < closestDist) {
      closestDist = currDist;
      closestPoint = *itPoints;
    }
  }
  return closestPoint;
};

 /**
  * Computes 3D point on plane closest to specified 3D point
  *
  * Justification:
  * The equation of a plane is: n * v = d, where n is the normal vector,
  * v is an arbitrary 3D point, and d is a constant offset.
  *
  * Given query point v', then the closest point on the above plane is:
  * v* = v' + a * n, for some unknown factor a
  *
  * Also, we know that n * v* = d, by definition of the plane.
  *
  * Thus, n * (v' + a * n) = d,
  * i.e. a = ( d - n * v' ) / ( n * n ),
  * which allows us to then solve for v* by substituting in a
  *
  * (side-note: n * n is guaranteed to be always non-zero for plane to exist)
  */
inline cv::Vec3d findClosestPointOnPlane(const cv::Vec3d& vp,
    const cv::Vec3d& n, double d) {
   double a = (d - n.dot(vp)) / n.dot(n);
   cv::Vec3d vs = vp + a*n;
   return vs;
};


/**
 * Computes 2D point on line closest to specified 2D point
 *
 * Justification:
 * The equation of a line is: vla + a * vd,
 * where vd = vlb - vla, for some constant a
 *
 * Let vdd be orthogonal to vd. The resulting point vr satisfied both:
 * vr = vla + a * vd, for some constant a
 * vr = vp - d * vdd, for some constant d
 *
 * Hence, vd * a + vdd * b = vp - vla. We can now solve a and d using the normal equation.
 */
inline cv::Vec2d findClosestPointOnLine2(const cv::Vec2d& vp,
    const cv::Vec2d& vla, const cv::Vec2d& vlb) {
  if (vla == vlb) { return vp; }
  cv::Vec2d vd = vlb - vla;

  double det = vd[0]*vd[0] + vd[1]*vd[1];
  double a = (vd[0] * (vp[0] - vla[0]) + vd[1] * (vp[1] - vla[1])) / det;
  //double d = (vd[1] * (vp[0] - vla[0]) - vd[0] * (vp[1] - vla[1])) / det;
  cv::Vec2d result = vla + a * vd;

  return result;
};
/**
 * same as above, but takes in normal form of line: (n1, n2) * (x, y) = nd
 */
inline cv::Vec2d findClosestPointOnLine2(const cv::Vec2d& vp,
    double n1, double n2, double nd) {
  // Make sure normal vector of line has non-zero norm
  double det = n1*n1 + n2*n2;
  if (det == 0) { return vp; }

  // Compute point on line
  double vlx = 0, vly = 0;
  if (n1 == 0) { vly = nd/n2; }
  else { vlx = nd/n1; }

  // Compute vr = vl + a * vd, where vd is orthogonal to n
  double a = (n2 * (vp[0] - vlx) - n1 * (vp[1] - vly)) / det;

  return cv::Vec2d(vlx + a*n2, vly - a*n1);
};


/**
 * Computes 3D point on line closest to specified 3D point
 *
 * Justification:
 * The equation of a line is: vla + a * vld,
 * where vld = vlb - vla, for some constant a
 *
 * Denote the resulting point on the line as vr, then the vector vn = (vr->vp) is
 * orthogonal to vld, as well as to a 3rd orthogonal vector vo.
 * vo can be computed as vld x vnp, where vnp = (vl->vp) for any point vl on the line.
 *
 * Then, we can link the 2 points vla and vp together, as:
 * vnp = vp - vla = a * vld + d * vn
 *
 * There are 2 unknowns and 3 equations here, so we can solve this using the
 * normal equation.
 */
inline cv::Vec3d findClosestPointOnLine3(const cv::Vec3d& vp,
    const cv::Vec3d& vla, const cv::Vec3d& vlb) {
  cv::Vec3d result;
  if (vla == vlb) { return vp; }
  cv::Vec3d vld = vlb - vla;
  cv::Vec3d vnp = vp - vla;
  if (vp == vla) { vnp = vp - vlb; }
  cv::Vec3d vo = vld.cross(vnp);
  cv::Vec3d vn = vld.cross(vo);

  cv::Mat A(3, 2, CV_64FC1);
  cv::Mat b(3, 1, CV_64FC1);
  for (unsigned int i = 0; i < 3; i++) {
    A.at<double>(i, 0) = vld[i];
    A.at<double>(i, 1) = vn[i];
    b.at<double>(i, 0) = vnp[i];
  }
  cv::Mat sln = (A.t()*A).inv()*(A.t()*b);
  //double a = sln.at<double>(0, 0);
  double d = sln.at<double>(1, 0);
  result = vp - d*vn;

  return result;
};


/**
 * Find the closest point to a number of planes, where each plane is represented
 * by their normal vector n, and constant offset d, which are stored into
 * matrices A = [n1; n2; ...], and b = [d1; d2; ...].
 *
 * The query point vp is only used in degenerate cases; see below.
 *
 * For 1 single plane, this function returns the closest point on plane to
 * the query point vp.
 *
 * Otherwise, we check if (A'*A) is invertible, and if it is, then we can use
 * the normal equation to solve for the closest point.
 *
 * If (A'*A) is not invertible, then we choose the first 2 planes that have
 * different normals, and find their intersecting line. We then compute
 * the closest point on line to the query point.
 */
cv::Vec3d findClosestPointToPlanes3(const cv::Vec3d& vp,
    const cv::Mat& A, const cv::Mat& b);


/**
 * Find the closest point to a number of lines, where each line is represented
 * by their normal vector n, and constant offset d, which are stored into
 * matrices A = [n1; n2; ...], and b = [d1; d2; ...].
 *
 * The query point vp is only used in degenerate cases; see below.
 *
 * For 1 single line, this function returns the closest point on line to
 * the query point vp.
 *
 * Otherwise, we check if the lines are parallel via det(A) != 0. If they
 * intersect, then we can use the normal equation to solve for the closest point.
 *
 * If all lines are parallel, then we average over the offsets b, to get an
 * average line. We then return the closest point on this average line to the
 * query point.
 */
cv::Vec2d findClosestPointToLines2(const cv::Vec2d& vp,
    const cv::Mat& A, const cv::Mat& b);


/**
 * Computes x intercept (in pixels, intersection with bottom of image) and
 * slope (in degrees, 0' = top of image) given line in image
 *
 * \param line: straight line; format: [dx, dy, x0, y0]
 * \param imWidth: image width, in pixels
 * \param imHeight: image height, in pixels
 */
std::pair<double, double> computeXInterceptAndSlopeFromLine(
    cv::Vec4f line,
    unsigned int imWidth,
    unsigned int imHeight);


/**
 * Computes heading angle (in degrees) given line in overhead image, and
 * given preferred heading directionality.
 *
 * \param line: straight line; format: [dx, dy, x0, y0]
 * \param imWidth: image width, in pixels
 * \param imHeight: image height, in pixels
 * \param preferredDirDeg: preferred direction of heading, in degrees (0 = top of image, 90 = right of image)
 */
double computeHeadingFromOverheadLine(
    cv::Vec4f line,
    unsigned int imWidth,
    unsigned int imHeight,
    double preferredDirDeg);

/**
 * Computes point closest to border in the direction of heading,
 * centered at image's center.
 *
 * \param heading: desired heading, in degrees; 0 = North/top of image; 90 = East/right of image
 * \param imWidth: image width
 * \param imHeight: image height
 * \param marginWidth: width of margin around image borders to avoid when computing intersection
 */
cv::Point computeHeadingBorderIntersection(
    double heading,
    unsigned int imWidth,
    unsigned int imHeight,
    unsigned int marginWidth);

/**
 * Computes and returns closest point within set of points
 */
inline cv::Point findClosestPoint(
    const std::vector<cv::Point>& points,
    const cv::Point targetPoint) {
  std::vector<cv::Point>::const_iterator itPoints = points.begin();
  std::vector<cv::Point>::const_iterator itPointsEnd = points.end();
  cv::Point result = *itPoints;
  double bestDistSqrd =
      (itPoints->x - targetPoint.x)*(itPoints->x - targetPoint.x) +
      (itPoints->y - targetPoint.y)*(itPoints->y - targetPoint.y);
  double currDistSqrd;
  for (itPoints++; itPoints != itPointsEnd; itPoints++) {
    currDistSqrd =
        (itPoints->x - targetPoint.x)*(itPoints->x - targetPoint.x) +
        (itPoints->y - targetPoint.y)*(itPoints->y - targetPoint.y);
    if (currDistSqrd < bestDistSqrd) {
      result = *itPoints;
      bestDistSqrd = currDistSqrd;
    }
  }

  return result;
};


inline double computeScaleFactor(const cv::Size& from, const cv::Size& to) {
  if (from == cv::Size() || to == cv::Size()) {
    return 1.0;
  } else {
    return std::min(double(to.width) / double(from.width),
        double(to.height) / double(from.height));
  }
};

};


#endif /* VECTORANDCIRCULARMATH_HPP_ */
