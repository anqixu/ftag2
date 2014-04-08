#ifndef CIRCULARSTATISTICS_HPP_
#define CIRCULARSTATISTICS_HPP_


#include <boost/math/constants/constants.hpp>
#include <boost/tr1/cmath.hpp>
#include <utility>
#include <cmath>


namespace vc_math {

/**
 * Samples from a von Mises probability density distribution
 *
 * NOTE: it turns out that exp() and cos() are much more costly than cyl_bessel_i()
 *
 * \param muRad: von Mises distribution mean angle, in radians
 * \param kappa: von Mises distribution concentration parameter, in 1/radians^2
 * \param queryRad: query angular value, in radians
 */
template <typename T>
T circ_vmpdf(T muRad, T kappa, T queryRad) {
  return exp(kappa * cos(queryRad - muRad)) /
      2 / boost::math::constants::pi<T>() /
      boost::math::tr1::cyl_bessel_i<int, T>(0, kappa);
};


/**
 * Estimates mean (mu) and concentration (kappa) parameters for
 * von Mises distribution from angular data
 */
template <typename T, typename Iterable>
std::pair<T, T> circ_vmpar(Iterable anglesRad) {
  // Compute sum of sine and cosines of angles, which maps angles into x and y components of their unit vectors
  T sum_sin_angles = 0.0;
  T sum_cos_angles = 0.0;

  if (anglesRad.size() == 0) {
    return std::make_pair(0, 0);
  } else if (anglesRad.size() == 1) {
    return std::make_pair(anglesRad.front(), 0);
  }

  typename Iterable::const_iterator it = anglesRad.begin();
  typename Iterable::const_iterator itEnd = anglesRad.end();
  size_t N = 0;
  for (; it != itEnd; it++) {
    sum_sin_angles += sin(*it);
    sum_cos_angles += cos(*it);
    N += 1;
  }

  // Compute total and average lengths of summed angular vector
  T sum_angle_vector_length = sqrt(sum_sin_angles*sum_sin_angles + sum_cos_angles*sum_cos_angles);
  T avg_angle_vector_length = sum_angle_vector_length / N;

  // Estimate kappa
  // Reference: Statistical analysis of circular data, Fisher, equation p. 88
  T kappa = 0;
  if (avg_angle_vector_length < 0.53) {
    kappa = 2*avg_angle_vector_length +
        pow(avg_angle_vector_length, 3) +
        5.0 / 6 * pow(avg_angle_vector_length, 5);
  } else if (avg_angle_vector_length >= 0.53 && avg_angle_vector_length < 0.85) {
    kappa = -0.4 +
        1.39 * avg_angle_vector_length +
        0.43 / (1 - avg_angle_vector_length);
  } else if (avg_angle_vector_length >= 1.0) {
    kappa = std::numeric_limits<double>::infinity();
  } else {
    kappa = 1.0 /
        (pow(avg_angle_vector_length, 3) -
        4 * pow(avg_angle_vector_length, 2) +
        3 * avg_angle_vector_length);
  }

  // Estimate mu
  T mu = atan2(sum_sin_angles, sum_cos_angles);

  // Set mu = 0 below kappa threshold
  // NOTE: kappa threshold parameter chosen such that the expected pdf sample
  //       will deviate from mean over all possible samples at approx. ~1e-15
  if (kappa < 1e-14) {
    mu = 0;
  }

#ifdef IMPLEMENT_CIRC_VMPAR_CONFINT
  % compute ingredients for conf. lim.
  % obtain length
  confint = 0.05;
  c2 = chi2inv((1-confint),1); // http://www.boost.org/doc/libs/1_47_0/libs/math/doc/sf_and_dist/html/math_toolkit/dist/dist_ref/dists/inverse_chi_squared_dist.html

  % check for resultant vector length and select appropriate formula
  % Reference: Statistical analysis of circular data, N. I. Fisher
  if avg_angle_vector_length < .9 && avg_angle_vector_length > sqrt(c2/2/n)
    t = sqrt((2*n*(2*sum_angle_vector_length^2-n*c2))/(4*n-c2));  % equ. 26.24
  elseif avg_angle_vector_length >= .9
    t = sqrt(n^2-(n^2-sum_angle_vector_length^2)*exp(c2/n));      % equ. 26.25
  else
    t = 0;
    t = NaN;
    warning('Requirements for confidence levels not met.');
  end

  % apply final transform
  t = acos(t./sum_angle_vector_length);

  ul = mu + t;
  ll = mu - t;
#endif

  return std::make_pair(mu, kappa);
};


template<typename T, typename Iterable>
T computeAngularMean(Iterable anglesRad) {
  if (anglesRad.size() == 0) {
    return 0.0;
  } else if (anglesRad.size() == 1) {
    return anglesRad.front();
  }

  T sumSin = 0.0;
  T sumCos = 0.0;
  typename Iterable::const_iterator it = anglesRad.begin();
  typename Iterable::const_iterator itEnd = anglesRad.end();

  for (; it != itEnd; it++) {
    sumSin += sin(*it);
    sumCos += cos(*it);
  }
  sumSin /= anglesRad.size();
  sumCos /= anglesRad.size();
  if (fabs(sumSin) < 2.2204e-15) { sumSin = 0.0; } // 2.2204e-15 == 10 * 2^-52 =
  if (fabs(sumCos) < 2.2204e-15) { sumCos = 0.0; } //   = 10 * IEEE754 double floating-point precision increment
  return atan2(sumSin, sumCos);
};


/**
 * WARNING: this does NOT guarantee to yield highest likelihood at either mu,
 *          hence it is not a measure of similarity!
 *
 * Samples 2 von Mises probability density distributions from [-pi, pi] and sum
 * their products, as a discrete approximation to the area underneath the product
 * of 2 von Mises pdfs.
 *
 * \param mu1: mean of first von Mises distribution, in radians
 * \param kappa1: concentration of first von Mises distribution, in 1/radians^2
 * \param mu2: mean of second von Mises distribution, in radians
 * \param kappa2: concentration of second von Mises distribution, in 1/radians^2
 */
/*
template <typename T>
T circ_vmprodarea(T mu1, T kappa1, T mu2, T kappa2, unsigned long numSamples = 2000) {
  T denom1 = 2 * pi<T>() * boost::math::tr1::cyl_bessel_i<int, T>(0, kappa1);
  T denom2 = 2 * pi<T>() * boost::math::tr1::cyl_bessel_i<int, T>(0, kappa2);
  T incr = 2 * pi<T>() /  (numSamples + 1);
  T accum = 0.0;
  for (T x = -pi<T>(); x < pi<T>(); x += incr) {
    accum += exp(kappa1 * cos(x - mu1) + kappa2 * cos(x - mu2));
  }
  accum = accum / denom1 / denom2 / numSamples;
  return accum;
};
*/

};


#endif /* CIRCULARSTATISTICS_HPP_ */
