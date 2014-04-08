#ifndef VECTORANDCIRCULARMATH_HPP_
#define VECTORANDCIRCULARMATH_HPP_


#include <boost/math/constants/constants.hpp>
#include <opencv2/opencv.hpp>
#include <utility>
#include <vector>
#include <array>
#include <cmath>
#include <cassert>


inline bool operator==(const cv::Vec4i& lhs, const cv::Vec4i& rhs) {
  return ((lhs[0] == rhs[0]) &&
          (lhs[1] == rhs[1]) &&
          (lhs[2] == rhs[2]) &&
          (lhs[3] == rhs[3]));
};


inline bool operator<(const cv::Vec4i& lhs, const cv::Vec4i& rhs) {
  if (lhs[0] < rhs[0]) return true;
  else if (lhs[0] > rhs[0]) return false;
  if (lhs[1] < rhs[1]) return true;
  else if (lhs[1] > rhs[1]) return false;
  if (lhs[2] < rhs[2]) return true;
  else if (lhs[2] > rhs[2]) return false;
  if (lhs[3] < rhs[3]) return true;
  return false;
};
inline bool lessThanVec4i(const cv::Vec4i& lhs, const cv::Vec4i& rhs) {
  return (lhs < rhs);
};
inline bool lessThanArray5i(const std::array<int, 5>& lhs, const std::array<int, 5>& rhs) {
  return (lhs < rhs);
};


namespace vc_math {


//constexpr double degree = boost::math::constants::pi<double>()/180.0;
//constexpr double radian = 180.0/boost::math::constants::pi<double>();
//constexpr double two_pi = boost::math::constants::pi<double>()*2;
constexpr double degree = 3.1415926535897932384626433832795/180.0;
constexpr double radian = 180.0/3.1415926535897932384626433832795;
constexpr double pi = 3.1415926535897932384626433832795;
constexpr double two_pi = 3.1415926535897932384626433832795*2;
constexpr double half_pi = 3.1415926535897932384626433832795/2;
constexpr double inv_sqrt_2pi = 0.3989422804014327;
constexpr double INVALID_ANGLE = 361.0;


inline double normal_pdf(double x, double m, double s) {
  double a = (x - m) / s;
  return (inv_sqrt_2pi / s) * exp(-0.5 * a * a);
};


inline double log_normal_pdf(double x, double m, double s) {
  double a = (x - m) / s;
  return log(inv_sqrt_2pi) - log(s) -0.5 * a * a;
};


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
 * Computes angular (and general modulo-'range') magnitude, signed
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
inline double dist(const cv::Point2d& xy1, const cv::Point2d& xy2) {
  return sqrt((xy1.x-xy2.x)*(xy1.x-xy2.x)+(xy1.y-xy2.y)*(xy1.y-xy2.y));
};
inline double dist(const cv::Point2f& xy1, const cv::Point2f& xy2) {
  return sqrt((xy1.x-xy2.x)*(xy1.x-xy2.x)+(xy1.y-xy2.y)*(xy1.y-xy2.y));
};
inline double dist(const cv::Point2i& xy1, const cv::Point2i& xy2) {
  return sqrt((xy1.x-xy2.x)*(xy1.x-xy2.x)+(xy1.y-xy2.y)*(xy1.y-xy2.y));
};
inline double dist(const cv::Vec4i& xy12) {
  return sqrt((xy12[0]-xy12[2])*(xy12[0]-xy12[2])+(xy12[1]-xy12[3])*(xy12[1]-xy12[3]));
};

/**
 * Wraps angle in degrees to [0, maxAngle) range
 */
inline constexpr double wrapAngle(double angleDeg, double maxAngle = 360.0) {
  return (angleDeg - floor(angleDeg/maxAngle)*maxAngle);
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
 * Check if the line segments (a<->b) and (c<->d) intersect with each other
 *
 * From: http://gamedev.stackexchange.com/questions/26004/how-to-detect-2d-line-on-line-collision
 *
 * WARNING: algorithm returns true for special case where the line segments
 *          are co-linear and do not overlap
 */
inline bool isIntersecting(const cv::Point2f& a, const cv::Point2f& b,
    const cv::Point2f& c, const cv::Point2f& d) {
  float denominator = ((b.x - a.x) * (d.y - c.y)) - ((b.y - a.y) * (d.x - c.x));
  float numerator1 = ((a.y - c.y) * (d.x - c.x)) - ((a.x - c.x) * (d.y - c.y));
  float numerator2 = ((a.y - c.y) * (b.x - a.x)) - ((a.x - c.x) * (b.y - a.y));

  if (denominator == 0) return numerator1 == 0 && numerator2 == 0;

  float r = numerator1 / denominator;
  float s = numerator2 / denominator;

  return (r >= 0 && r <= 1) && (s >= 0 && s <= 1);
};


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

/**
 * Computes (minimum) scale factor between 2 sizes (i.e. prefer letterbox over
 * cropping)
 */
inline double computeScaleFactor(const cv::Size& from, const cv::Size& to) {
  if (from == cv::Size() || to == cv::Size()) {
    return 1.0;
  } else {
    return std::min(double(to.width) / double(from.width),
        double(to.height) / double(from.height));
  }
};

/**
 * Computes orientation of line segment (of the form [x1, y1, x2, y2])
 */
inline double orientation(const cv::Vec4i& seg) {
  return std::atan2(seg[3] - seg[1], seg[2] - seg[0]);
};
inline double orientation(const cv::Point2d& ptA, const cv::Point2d& ptB) {
  return std::atan2(ptB.y - ptA.y, ptB.x - ptA.x);
};

/**
 * Sort values in increasing order
 * NOTE: faster than recursive implementations and specifically
 *       cv::sort(vec4iA, vec4iB, CV_SORT_EVERY_COLUMN | CV_SORT_ASCENDING)
 */
inline cv::Vec4i sort(cv::Vec4i v) {
  cv::Vec4i s(v);
  if (v[0] > v[1]) {
    s[0] = v[1]; s[1] = v[0];
  }
  if (v[2] > v[3]) {
    s[2] = v[3]; s[3] = v[2];
  }

  if (s[0] > s[2]) {
    v[0] = s[2];
    if (s[3] > s[1]) {
      v[1] = s[0]; v[2] = s[1]; v[3] = s[3];
    } else if (s[3] > s[0]) {
      v[1] = s[0]; v[2] = s[3]; v[3] = s[1];
    } else {
      v[1] = s[3]; v[2] = s[0]; v[3] = s[1];
    }
  } else {
    v[0] = s[0];
    if (s[1] > s[3]) {
      v[1] = s[2]; v[2] = s[3]; v[3] = s[1];
    } else if (s[1] > s[2]) {
      v[1] = s[2]; v[2] = s[1]; v[3] = s[3];
    } else {
      v[1] = s[1]; v[2] = s[2]; v[3] = s[3];
    }
  }
  return v;
};

/**
 * Cycles vector such that smallest value is listed first, but the (cyclic)
 * ordering of values are preserved
 */
inline cv::Vec4i minCyclicOrder(cv::Vec4i v) {
  if ((v[0] <= v[1]) && (v[0] <= v[2]) && (v[0] <= v[3])) {
    return v;
  } else if ((v[1] <= v[0]) && (v[1] <= v[2]) && (v[1] <= v[3])) {
      return cv::Vec4i(v[1], v[2], v[3], v[0]);
  } else if ((v[2] <= v[0]) && (v[2] <= v[1]) && (v[2] <= v[3])) {
    return cv::Vec4i(v[2], v[3], v[0], v[1]);
  } else { // if ((v[3] <= v[0]) && (v[3] <= v[1]) && (v[3] <= v[2])) {
    return cv::Vec4i(v[3], v[0], v[1], v[2]);
  }
};

/**
 * Sorts and removes duplicate entries in-place
 */
inline void unique(std::vector<cv::Vec4i>& v) {
  std::sort(v.begin(), v.end(), lessThanVec4i);

  std::vector<cv::Vec4i> u;
  for (const cv::Vec4i& d: v) {
    if (u.empty()) { u.push_back(d); }
    else if (u.back() < d) { u.push_back(d); }
  }
  v.swap(u);
};
inline void unique(std::vector< std::array<int, 5> >& v) {
  std::sort(v.begin(), v.end(), lessThanArray5i);

  std::vector< std::array<int, 5> > u;
  for (const std::array<int, 5>& d: v) {
    if (u.empty()) { u.push_back(d); }
    else if (u.back() < d) { u.push_back(d); }
  }
  v.swap(u);
};

/**
 * Returns dot product of 2 line segments (of the form [x1, y1, x2, y2])
 */
inline double dot(const cv::Vec4i& A, const cv::Vec4i& B) {
  return (A[2]-A[0])*(B[2]-B[0]) + (A[3]-A[1])*(B[3]-B[1]);
};

/**
 * Returns dot product of 2 line segments
 */
inline double dot(const cv::Point2f& endA1, const cv::Point2f& endA2,
    const cv::Point2f& endB1, const cv::Point2f& endB2) {
  return (endA2.x-endA1.x)*(endB2.x-endB1.x) + (endA2.y-endA1.y)*(endB2.y-endB1.y);
};

/**
 * Converts a rotation matrix into a quaternion
 *
 * From: http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
 */
inline void rotMat2quat(const cv::Mat rotMat,
    double& w, double& x, double& y, double& z) {
  assert(rotMat.rows == 3 && rotMat.cols == 3 && rotMat.isContinuous() &&
      rotMat.elemSize() == sizeof(double));

  double tr = rotMat.at<double>(0, 0) + rotMat.at<double>(1, 1) + rotMat.at<double>(2, 2);
  double s;

  if (tr > 0) {
    s = 0.5/sqrt(tr + 1.0);
    w = 0.25/s;
    x = (rotMat.at<double>(2, 1) - rotMat.at<double>(1, 2))*s;
    y = (rotMat.at<double>(0, 2) - rotMat.at<double>(2, 0))*s;
    z = (rotMat.at<double>(1, 0) - rotMat.at<double>(0, 1))*s;
  } else if (rotMat.at<double>(0, 0) > rotMat.at<double>(1, 1) &&
      rotMat.at<double>(0, 0) > rotMat.at<double>(2, 2)) {
    s = 2.0*sqrt(1.0 + 2*rotMat.at<double>(0, 0) - tr);
    w = (rotMat.at<double>(2, 1) - rotMat.at<double>(1, 2))/s;
    x = 0.25*s;
    y = (rotMat.at<double>(0, 1) + rotMat.at<double>(1, 0))/s;
    z = (rotMat.at<double>(0, 2) + rotMat.at<double>(2, 0))/s;
  } else if (rotMat.at<double>(1, 1) > rotMat.at<double>(2, 2)) {
    s = 2.0*sqrt(1.0 + 2*rotMat.at<double>(1, 1) - tr);
    w = (rotMat.at<double>(0, 2) - rotMat.at<double>(2, 0))/s;
    x = (rotMat.at<double>(0, 1) + rotMat.at<double>(1, 0))/s;
    y = 0.25*s;
    z = (rotMat.at<double>(1, 2) + rotMat.at<double>(2, 1))/s;
  } else {
    s = 2.0*sqrt(1.0 + 2*rotMat.at<double>(2, 2) - tr);
    w = (rotMat.at<double>(1, 0) - rotMat.at<double>(0, 1))/s;
    x = (rotMat.at<double>(0, 2) + rotMat.at<double>(2, 0))/s;
    y = (rotMat.at<double>(1, 2) + rotMat.at<double>(2, 1))/s;
    z = 0.25*s;
  }
};


inline cv::Mat quat2RotMat(double w, double x, double y, double z) {
  double distSqrd = x*x+y*y+z*z+w*w;
  if (distSqrd == 0.0) { return cv::Mat::zeros(3, 3, CV_64FC1); }
  double s = 2.0/distSqrd;
  double xs = x * s,   ys = y * s,   zs = z * s;
  double wx = w * xs,  wy = w * ys,  wz = w * zs;
  double xx = x * xs,  xy = x * ys,  xz = x * zs;
  double yy = y * ys,  yz = y * zs,  zz = z * zs;
  cv::Mat result(3, 3, CV_64FC1);
  double* data = (double*) result.data;
  *data = 1.0 - (yy + zz); data++;
  *data = xy - wz; data++;
  *data = xz + wy; data++;
  *data = xy + wz; data++;
  *data = 1.0 - (xx + zz); data++;
  *data = yz - wx; data++;
  *data = xz - wy; data++;
  *data = yz + wx; data++;
  *data = 1.0 - (xx + yy);
  return result;
};


inline cv::Mat str2mat(const std::string& s, int rows,
    int type = CV_64F, int channels = 1) {
  std::string input = s;
  auto it = std::remove_if(std::begin(input), std::end(input),
      [](char c) { return (c == ',' || c == ';' || c == ':'); });
  input.erase(it, std::end(input));

  cv::Mat mat(0, 0, type);
  std::istringstream iss(input);
  double currNum;
  while (!iss.eof()) {
    iss >> currNum;
    mat.push_back(currNum);
  }
  return mat.reshape(channels, rows);
};


/**
 * Check if 2 (convex) polygons overlap, using the dividing axis algorithm:
 * - if 2 convex polygons do not intersect, then there exists a line that passes between them
 * - such a line only exists if formed by one of the polygons' sides
 *
 * Note that two polygons sharing an edge, or whose one's endpoint intersects
 * the other's edge, are considered to be overlapping.
 */
inline bool checkPolygonOverlap(const std::vector<cv::Point2f>& cornersA, const std::vector<cv::Point2f>& cornersB) {
  // Compute angles perpendicular to each of the polygons' sides
  std::vector<double> projectionAngles;
  unsigned int i, j;
  cv::Point2f vec;
  for (i = 0; i < cornersA.size(); i++) {
    j = (i == 0) ? cornersA.size() - 1 : i - 1;
    vec = cornersA[i] - cornersA[j];
    if (vec.x == 0 && vec.y == 0) continue;
    projectionAngles.push_back(atan2(vec.x, vec.y)); // NOTE: (x, y) arguments swapped to compute perpendicular angle
  }
  for (i = 0; i < cornersB.size(); i++) {
    j = (i == 0) ? cornersB.size() - 1 : i - 1;
    vec = cornersB[i] - cornersB[j];
    if (vec.x == 0 && vec.y == 0) continue;
    projectionAngles.push_back(atan2(vec.x, vec.y));
  }

  // Scan for dividing axis line
  bool overlap = true;
  for (const double& angle: projectionAngles) {
    double projAMin = std::numeric_limits<double>::infinity();
    double projAMax = -std::numeric_limits<double>::infinity();
    double projBMin = std::numeric_limits<double>::infinity();
    double projBMax = -std::numeric_limits<double>::infinity();
    double cosAngle = cos(angle);
    double sinAngle = sin(angle);
    for (const cv::Point2f& pt: cornersA) {
      double proj = cosAngle * pt.x - sinAngle * pt.y;
      if (proj < projAMin) projAMin = proj;
      if (proj > projAMax) projAMax = proj;
    }
    for (const cv::Point2f& pt: cornersB) {
      double proj = cosAngle * pt.x - sinAngle * pt.y;
      if (proj < projBMin) projBMin = proj;
      if (proj > projBMax) projBMax = proj;
    }
    if ((projAMax < projBMin) || (projAMin > projBMax)) {
      overlap = false;
      break;
    }
  }

  return overlap;
};


};


#endif /* VECTORANDCIRCULARMATH_HPP_ */
