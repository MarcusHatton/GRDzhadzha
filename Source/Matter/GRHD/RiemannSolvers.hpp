/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo root directory.
 */

#ifndef GRHD_RIEMANNSOLVERS_HPP_
#define GRHD_RIEMANNSOLVERS_HPP_

#include "DimensionDefinitions.hpp"
#include "GRHDVars.hpp"
#include "Valencia.hpp"
#include <algorithm>
#include <cmath>

namespace GRHD
{
template <class data_t> struct WaveSpeeds
{
    data_t lower;
    data_t upper;
};

template <class data_t, class eos_t>
WaveSpeeds<data_t>
compute_characteristic_speeds(const Primitive<data_t> &primitive,
                              const eos_t &eos,
                              const Tensor<2, data_t> &spatial_metric_LL,
                              const Tensor<2, data_t> &spatial_metric_UU,
                              int direction)
{
    const data_t v_squared =
        compute_velocity_squared(primitive.velocity_U, spatial_metric_LL);
    const data_t sound_speed_squared = eos.compute_sound_speed_squared(
        primitive.rho, primitive.eps, primitive.pressure);
    const data_t sound_speed = std::sqrt(sound_speed_squared);
    const data_t velocity = primitive.velocity_U[direction];
    const data_t denominator = 1.0 - v_squared * sound_speed_squared;
    const data_t radicand =
        (1.0 - v_squared) *
        (spatial_metric_UU[direction][direction] *
             (1.0 - v_squared * sound_speed_squared) -
         velocity * velocity * (1.0 - sound_speed_squared));
    const data_t root = sound_speed * std::sqrt(std::max(data_t(0.0), radicand));

    WaveSpeeds<data_t> speeds;
    speeds.lower =
        (velocity * (1.0 - sound_speed_squared) - root) / denominator;
    speeds.upper =
        (velocity * (1.0 - sound_speed_squared) + root) / denominator;
    return speeds;
}

template <class data_t, class eos_t>
WaveSpeeds<data_t>
compute_characteristic_speeds(const Primitive<data_t> &primitive,
                              const eos_t &eos,
                              const Tensor<2, data_t> &spatial_metric_LL,
                              const Tensor<2, data_t> &spatial_metric_UU,
                              int direction, const data_t &lapse,
                              const Tensor<1, data_t> &shift_U)
{
    WaveSpeeds<data_t> speeds = compute_characteristic_speeds(
        primitive, eos, spatial_metric_LL, spatial_metric_UU, direction);
    speeds.lower = lapse * speeds.lower - shift_U[direction];
    speeds.upper = lapse * speeds.upper - shift_U[direction];
    return speeds;
}

template <class data_t, class eos_t>
Flux<data_t> compute_hlle_flux(
    const Primitive<data_t> &left_primitive,
    const Primitive<data_t> &right_primitive,
    const Conserved<data_t> &left_conserved,
    const Conserved<data_t> &right_conserved, const eos_t &eos,
    const Tensor<2, data_t> &spatial_metric_LL,
    const Tensor<2, data_t> &spatial_metric_UU, int direction,
    const data_t &lapse, const Tensor<1, data_t> &shift_U)
{
    const Flux<data_t> left_flux =
        compute_flux(left_primitive, left_conserved, direction, lapse, shift_U);
    const Flux<data_t> right_flux = compute_flux(
        right_primitive, right_conserved, direction, lapse, shift_U);
    const WaveSpeeds<data_t> left_speeds = compute_characteristic_speeds(
        left_primitive, eos, spatial_metric_LL, spatial_metric_UU, direction,
        lapse, shift_U);
    const WaveSpeeds<data_t> right_speeds = compute_characteristic_speeds(
        right_primitive, eos, spatial_metric_LL, spatial_metric_UU, direction,
        lapse, shift_U);

    const data_t s_lower =
        std::min(data_t(0.0), std::min(left_speeds.lower, right_speeds.lower));
    const data_t s_upper =
        std::max(data_t(0.0), std::max(left_speeds.upper, right_speeds.upper));

    if (s_lower >= 0.0)
        return left_flux;
    if (s_upper <= 0.0)
        return right_flux;

    Flux<data_t> flux;
    const data_t inverse_speed_width = 1.0 / (s_upper - s_lower);
    flux.D = (s_upper * left_flux.D - s_lower * right_flux.D +
              s_lower * s_upper * (right_conserved.D - left_conserved.D)) *
             inverse_speed_width;
    flux.tau =
        (s_upper * left_flux.tau - s_lower * right_flux.tau +
         s_lower * s_upper * (right_conserved.tau - left_conserved.tau)) *
        inverse_speed_width;
    FOR(i)
    {
        flux.S_L[i] =
            (s_upper * left_flux.S_L[i] - s_lower * right_flux.S_L[i] +
             s_lower * s_upper *
                 (right_conserved.S_L[i] - left_conserved.S_L[i])) *
            inverse_speed_width;
    }
    return flux;
}

template <class data_t, class eos_t>
Flux<data_t> compute_hlle_flux(
    const Primitive<data_t> &left_primitive,
    const Primitive<data_t> &right_primitive,
    const Conserved<data_t> &left_conserved,
    const Conserved<data_t> &right_conserved, const eos_t &eos,
    const Tensor<2, data_t> &spatial_metric_LL,
    const Tensor<2, data_t> &spatial_metric_UU, int direction)
{
    Tensor<1, data_t> zero_shift;
    FOR(i) { zero_shift[i] = 0.0; }
    return compute_hlle_flux(left_primitive, right_primitive, left_conserved,
                             right_conserved, eos, spatial_metric_LL,
                             spatial_metric_UU, direction, data_t(1.0),
                             zero_shift);
}
} // namespace GRHD

#endif /* GRHD_RIEMANNSOLVERS_HPP_ */
