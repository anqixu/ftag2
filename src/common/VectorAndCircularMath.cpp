#include "common/VectorAndCircularMath.hpp"


using namespace std;
using namespace boost::math::constants;


namespace vc_math {


std::pair<double, double> computeXInterceptAndSlopeFromLine(
    cv::Vec4f line,
    unsigned int imWidth,
    unsigned int imHeight) {
  double currX, currY, a;
  double vx = line[0], vy = line[1], x0 = line[2], y0 = line[3];

  // Compute intersection with bottom border
  if (vy != 0) {
    currY = imHeight - 1;
    a = (currY - y0)/vy;
    currX = x0 + a*vx;
  } else {
    currX = std::numeric_limits<double>::max();
  }

  double slope = atan2(vx, -vy) * radian;
  if (slope > 90.0) { slope -= 180.0; }
  else if (slope <= -90.0) { slope += 180.0; }

  return std::pair<double, double>(currX, slope);
};


double computeHeadingFromOverheadLine(
    cv::Vec4f line,
    unsigned int imWidth,
    unsigned int imHeight,
    double preferredDirDeg) {
  double currX, currY, a, currHeading, bestHeading = 0;
  double currHeadingDist, bestHeadingDist = std::numeric_limits<double>::max();
  double vx = line[0], vy = line[1], x0 = line[2], y0 = line[3];
  double cx = imWidth/2.0 - 0.5;
  double cy = imHeight/2.0 - 0.5;

  // Compute intersection with top and bottom borders
  if (vy != 0) {
    currY = 0;
    a = (currY - y0)/vy;
    currX = x0 + a*vx;
    if (currX >= 0 && currX < imWidth) {
      currHeading = atan2(currX - cx, cy - currY) * radian;
      currHeadingDist = angularDist(currHeading, preferredDirDeg);
      if (currHeadingDist < bestHeadingDist) {
        bestHeadingDist = currHeadingDist;
        bestHeading = currHeading;
      }
    }

    currY = imHeight - 1;
    a = (currY - y0)/vy;
    currX = x0 + a*vx;
    if (currX >= 0 && currX < imWidth) {
      currHeading = atan2(currX - cx, cy - currY) * radian;
      currHeadingDist = angularDist(currHeading, preferredDirDeg);
      if (currHeadingDist < bestHeadingDist) {
        bestHeadingDist = currHeadingDist;
        bestHeading = currHeading;
      }
    }
  } else {
    currHeading = 0.0;
    currHeadingDist = angularDist(currHeading, preferredDirDeg);
    if (currHeadingDist < bestHeadingDist) {
      bestHeadingDist = currHeadingDist;
      bestHeading = currHeading;
    }

    currHeading = 180.0;
    currHeadingDist = angularDist(currHeading, preferredDirDeg);
    if (currHeadingDist < bestHeadingDist) {
      bestHeadingDist = currHeadingDist;
      bestHeading = currHeading;
    }
  }

  // Compute intersection with left and right borders
  if (vx != 0) {
    currX = 0;
    a = (currX - x0)/vx;
    currY = y0 + a*vy;
    if (currY >= 0 && currY < imHeight) {
      currHeading = atan2(currX - cx, cy - currY) * radian;
      currHeadingDist = angularDist(currHeading, preferredDirDeg);
      if (currHeadingDist < bestHeadingDist) {
        bestHeadingDist = currHeadingDist;
        bestHeading = currHeading;
      }
    }

    currX = imWidth - 1;
    a = (currX - x0)/vx;
    currY = y0 + a*vy;
    if (currY >= 0 && currY < imHeight) {
      currHeading = atan2(currX - cx, cy - currY) * radian;
      currHeadingDist = angularDist(currHeading, preferredDirDeg);
      if (currHeadingDist < bestHeadingDist) {
        bestHeadingDist = currHeadingDist;
        bestHeading = currHeading;
      }
    }
  } else {
    currHeading = 90.0;
    currHeadingDist = angularDist(currHeading, preferredDirDeg);
    if (currHeadingDist < bestHeadingDist) {
      bestHeadingDist = currHeadingDist;
      bestHeading = currHeading;
    }

    currHeading = 270.0;
    currHeadingDist = angularDist(currHeading, preferredDirDeg);
    if (currHeadingDist < bestHeadingDist) {
      bestHeadingDist = currHeadingDist;
      bestHeading = currHeading;
    }
  }

  return bestHeading;
};


cv::Point computeHeadingBorderIntersection(
    double heading,
    unsigned int imWidth,
    unsigned int imHeight,
    unsigned int marginWidth) {
  double cx = imWidth/2.0 - 0.5, cy = imHeight/2.0 - 0.5;
  cv::Point result(round(cx), round(cy));
  double vx = sin(heading * degree), vy = -cos(heading * degree);
  double currY, currX, a;
  double headingMod = wrapAngle(heading);

  // Compute intersections for right angles
  if (headingMod == 0.0) {
    result.y = round(marginWidth);
    return result;
  } else if (headingMod == 90.0) {
    result.x = round(imWidth - 1 - marginWidth);
    return result;
  } else if (headingMod == 180.0) {
    result.y = round(imHeight - 1 - marginWidth);
    return result;
  } else if (headingMod == 270.0) {
    result.x = round(marginWidth);
    return result;
  }

  // Compute intersection with top or bottom border
  currY = (headingMod < 90.0 || headingMod > 270.0) ?
      marginWidth : imHeight - 1 - marginWidth;
  a = (currY - cy)/vy;
  currX = cx + a*vx;
  if (currX >= 0 && currX < imWidth) {
    result.x = round(currX);
    result.y = round(currY);
    return result;
  }

  // Compute intersection with left or right border
  currX = (headingMod > 180.0) ?
      marginWidth : imWidth - 1 - marginWidth;
  a = (currX - cx)/vx;
  currY = cy + a*vy;
  if (currY >= 0 && currY < imHeight) {
    result.x = round(currX);
    result.y = round(currY);
    return result;
  }

  // Coding error
  CV_Assert(0);
  return result;
};


cv::Vec3d findClosestPointToPlanes3(const cv::Vec3d& vp,
    const cv::Mat& A, const cv::Mat& b) {
  if (A.rows < 1 || A.rows != b.rows) { return vp; }

  // For 1 single plane, this function returns the closest point on plane to
  // the query point vp.
  if (A.rows == 1) {
    cv::Vec3d n(A.at<double>(0, 0), A.at<double>(0, 1), A.at<double>(0, 2));
    double d = b.at<double>(0, 0);
    return findClosestPointOnPlane(vp, n, d);
  }

  // Otherwise, we check if (A'*A) is invertible, and if it is, then we can use
  // the normal equation to solve for the closest point.
  cv::Vec3d result;
  cv::Mat As = A.t()*A;
  cv::Mat As_inv;
  double det = invert(As, As_inv);
  if (det != 0) {
    cv::Mat sln = As_inv*(A.t()*b);
    result[0] = sln.at<double>(0, 0);
    result[1] = sln.at<double>(1, 0);
    result[2] = sln.at<double>(2, 0);
    return result;
  }

  // If (A'*A) is not invertible, then we choose the first 2 planes that have
  // different normals, and find their intersecting line. We then compute
  // the closest point on line to the query point.
  cv::Vec3d vna, vnb, vna_norm, vnb_norm;
  cv::Mat Ap(2, 2, CV_64FC1);
  cv::Mat bp(2, 1, CV_64FC1);
  bool foundPair = false;
  double vna_norm_factor, vnb_norm_factor;
  for (int i = 0; i < A.rows - 1; i++) {
    vna = cv::Vec3d(A.at<double>(i, 0), A.at<double>(i, 1), A.at<double>(i, 2));
    vna_norm = vna / sqrt(vna.dot(vna));
    bp.at<double>(0, 0) = b.at<double>(i, 0);

    for (int j = i + 1; j < A.rows; j++) {
      vnb = cv::Vec3d(A.at<double>(j, 0), A.at<double>(j, 1), A.at<double>(j, 2));
      vnb_norm = vnb / sqrt(vnb.dot(vnb));
      bp.at<double>(1, 0) = b.at<double>(j, 0);
      if (vna_norm != vnb_norm && vna_norm != -vnb_norm) {
        foundPair = true; break;
      }
    }
    if (foundPair) { break; }
  }
  if (!foundPair) {
    // All planes are parallel; so we can compute closest distance from average
    // plane and point (the average plane is found by averaging the offsets)
    vna = cv::Vec3d(A.at<double>(0, 0), A.at<double>(0, 1), A.at<double>(0, 2));
    vna_norm_factor = sqrt(vna.dot(vna));
    if (vna_norm_factor == 0) return vp; // Failed since one of the planes is ill-defined
    vna_norm = vna / vna_norm_factor;
    double bAvg = b.at<double>(0, 0) / vna_norm_factor;
    for (int j = 1; j < A.rows; j++) {
      vnb = cv::Vec3d(A.at<double>(j, 0), A.at<double>(j, 1), A.at<double>(j, 2));
      vnb_norm_factor = sqrt(vnb.dot(vnb));
      if (vnb_norm_factor == 0) return vp; // Failed since one of the planes is ill-defined
      vnb_norm = vnb / vnb_norm_factor;
      if (vna_norm == -vnb_norm) {
        bAvg -= b.at<double>(j, 0) / vnb_norm_factor;
      } else { // expect vna_norm == vnb_norm
        bAvg += b.at<double>(j, 0) / vnb_norm_factor;
      }
    }
    bAvg /= A.rows;

    return findClosestPointOnPlane(vp, vna_norm, bAvg);
  }

  cv::Vec3d vld = vna.cross(vnb); // vector of line
  cv::Vec3d vl0(0, 0, 0); // point on line
  if (vna[0] != 0) { // set vl0.x == 0, and solve for vl0.y and vl0.z
    Ap.at<double>(0, 0) = vna[1]; Ap.at<double>(0, 1) = vna[2];
    Ap.at<double>(1, 0) = vnb[1]; Ap.at<double>(1, 1) = vnb[2];
    cv::Mat sln = (A.t()*A).inv()*(A.t()*b);
    vl0[1] = sln.at<double>(0, 0); vl0[2] = sln.at<double>(1, 0);
  } else if (vna[1] != 0) {
    Ap.at<double>(0, 0) = vna[0]; Ap.at<double>(0, 1) = vna[2];
    Ap.at<double>(1, 0) = vnb[0]; Ap.at<double>(1, 1) = vnb[2];
    cv::Mat sln = (A.t()*A).inv()*(A.t()*b);
    vl0[0] = sln.at<double>(0, 0); vl0[2] = sln.at<double>(1, 0);
  } else {
    Ap.at<double>(0, 0) = vna[0]; Ap.at<double>(0, 1) = vna[1];
    Ap.at<double>(1, 0) = vnb[0]; Ap.at<double>(1, 1) = vnb[1];
    cv::Mat sln = (A.t()*A).inv()*(A.t()*b);
    vl0[0] = sln.at<double>(0, 0); vl0[1] = sln.at<double>(1, 0);
  }
  return findClosestPointOnLine3(vp, vl0, vl0 + vld);
};


cv::Vec2d findClosestPointToLines2(const cv::Vec2d& vp,
    const cv::Mat& A, const cv::Mat& b) {
  if (A.rows < 1 || A.rows != b.rows) { return vp; }

  // For 1 single line, this function returns the closest point on line to
  // the query point vp.
  if (A.rows == 1) {
    return findClosestPointOnLine2(vp,
        A.at<double>(0, 0),
        A.at<double>(0, 1),
        b.at<double>(0, 0));
  }

  // Otherwise, we check if the lines are parallel via det(A) != 0. If they
  // intersect, then we can use the normal equation to solve for the closest point.
  cv::Mat As = A.t()*A;
  cv::Mat As_inv;
  double det = invert(As, As_inv);
  if (det != 0) {
    cv::Mat sln = As_inv*(A.t()*b);
    return cv::Vec2d(sln.at<double>(0, 0), sln.at<double>(1, 0));
  }

  // If all lines are parallel, then we average over the offsets b, to get an
  // average line. We then return the closest point on this average line to the
  // query point.
  cv::Vec2d na(A.at<double>(0, 0), A.at<double>(0, 1));
  cv::Vec2d nb, na_norm, nb_norm;
  double na_norm_factor, nb_norm_factor;
  na_norm_factor = sqrt(na.dot(na));
  if (na_norm_factor == 0) return vp; // Failed since one of the lines is ill-defined
  na_norm = na / na_norm_factor;
  double bAvg = b.at<double>(0, 0) / na_norm_factor;
  for (int j = 1; j < A.rows; j++) {
    nb = cv::Vec2d(A.at<double>(j, 0), A.at<double>(j, 1));
    nb_norm_factor = sqrt(nb.dot(nb));
    if (nb_norm_factor == 0) return vp; // Failed since one of the lines is ill-defined
    nb_norm = nb / nb_norm_factor;
    if (na_norm == -nb_norm) {
      bAvg -= b.at<double>(j, 0) / nb_norm_factor;
    } else { // expect na_norm == nb_norm
      bAvg += b.at<double>(j, 0) / nb_norm_factor;
    }
  }
  bAvg /= A.rows;
  return findClosestPointOnLine2(vp, na_norm[0], na_norm[1], bAvg);
};


};
