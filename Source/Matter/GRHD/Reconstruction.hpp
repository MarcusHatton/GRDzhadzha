/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef GRHD_RECONSTRUCTION_HPP_
#define GRHD_RECONSTRUCTION_HPP_

#include "DimensionDefinitions.hpp"
#include "GRHDVars.hpp"
#include <algorithm>
#include <cmath>

namespace GRHD
{
template <class data_t> data_t minmod(data_t left, data_t right)
{
    if (left * right <= 0.0)
        return 0.0;
    return left > 0.0 ? std::min(left, right) : std::max(left, right);
}

template <class data_t>
data_t minmod(data_t first, data_t second, data_t third)
{
    return minmod(first, minmod(second, third));
}

//! Monotonized central primitive slope for uniform 1D grids.
template <class data_t>
data_t compute_limited_slope(data_t left, data_t center, data_t right,
                             double theta = 1.5)
{
    return minmod(data_t(theta) * (center - left),
                  data_t(0.5) * (right - left),
                  data_t(theta) * (right - center));
}

template <class data_t>
Primitive<data_t> compute_limited_slope(const Primitive<data_t> &left,
                                        const Primitive<data_t> &center,
                                        const Primitive<data_t> &right,
                                        double theta = 1.5)
{
    Primitive<data_t> slope;
    slope.rho = compute_limited_slope(left.rho, center.rho, right.rho, theta);
    slope.eps = compute_limited_slope(left.eps, center.eps, right.eps, theta);
    slope.pressure = compute_limited_slope(left.pressure, center.pressure,
                                           right.pressure, theta);
    FOR(i)
    {
        slope.velocity_U[i] = compute_limited_slope(
            left.velocity_U[i], center.velocity_U[i], right.velocity_U[i],
            theta);
    }
    return slope;
}

template <class data_t>
Primitive<data_t> add_scaled(const Primitive<data_t> &primitive,
                             const Primitive<data_t> &slope, data_t scale)
{
    Primitive<data_t> out;
    out.rho = primitive.rho + scale * slope.rho;
    out.eps = primitive.eps + scale * slope.eps;
    out.pressure = primitive.pressure + scale * slope.pressure;
    FOR(i) { out.velocity_U[i] = primitive.velocity_U[i] + scale * slope.velocity_U[i]; }
    return out;
}
} // namespace GRHD

#endif /* GRHD_RECONSTRUCTION_HPP_ */
