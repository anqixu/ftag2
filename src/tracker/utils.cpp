/*
 * utils.cpp
 *
 *  Created on: 2013-11-20
 *      Author: dacocp
 */
#include "tracker/utils.hpp"

double normal_pdf(double x, double m, double s)
{
    static const double inv_sqrt_2pi = 0.3989422804014327;
    double a = (x - m) / s;

    return inv_sqrt_2pi / s * exp(-0.5f * a * a);
}


double log_normal_pdf(double x, double m, double s)
{
    static const double inv_sqrt_2pi = 0.3989422804014327;
    double a = (x - m) / s;

//    return inv_sqrt_2pi / s * exp(-0.5f * a * a);
    return log(inv_sqrt_2pi) - log(s) -0.5f * a * a;
}
