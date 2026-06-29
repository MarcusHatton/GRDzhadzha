/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef GRHD_VALENCIA_HPP_
#define GRHD_VALENCIA_HPP_

#include "DimensionDefinitions.hpp"
#include "GRHDVars.hpp"
#include "TensorAlgebra.hpp"
#include <algorithm>
#include <cmath>

namespace GRHD
{
struct RecoveryOptions
{
    int max_iterations = 50;
    double tolerance = 1.0e-12;
    double pressure_floor = 1.0e-12;
    double max_velocity_squared = 1.0 - 1.0e-12;
};

template <class data_t> struct RecoveryResult
{
    Primitive<data_t> primitive;
    bool success = false;
    int iterations = 0;
    data_t residual = 0.0;
};

template <class data_t>
data_t compute_velocity_squared(const Tensor<1, data_t> &velocity_U,
                                const Tensor<2, data_t> &spatial_metric_LL)
{
    data_t v_squared = 0.0;
    FOR(i, j)
    {
        v_squared +=
            spatial_metric_LL[i][j] * velocity_U[i] * velocity_U[j];
    }
    return v_squared;
}

template <class data_t>
data_t compute_lorentz_factor(const Tensor<1, data_t> &velocity_U,
                              const Tensor<2, data_t> &spatial_metric_LL)
{
    const data_t v_squared =
        compute_velocity_squared(velocity_U, spatial_metric_LL);
    return 1.0 / sqrt(1.0 - v_squared);
}

template <class data_t, class eos_t>
Conserved<data_t>
compute_conserved(const Primitive<data_t> &primitive, const eos_t &eos,
                  const Tensor<2, data_t> &spatial_metric_LL)
{
    Conserved<data_t> conserved;
    const Tensor<1, data_t> velocity_L =
        TensorAlgebra::lower_all(primitive.velocity_U, spatial_metric_LL);
    const data_t W =
        compute_lorentz_factor(primitive.velocity_U, spatial_metric_LL);
    const data_t h = eos.compute_specific_enthalpy(
        primitive.rho, primitive.eps, primitive.pressure);
    const data_t rho_h_W2 = primitive.rho * h * W * W;

    conserved.D = primitive.rho * W;
    conserved.tau = rho_h_W2 - primitive.pressure - conserved.D;
    FOR(i) { conserved.S_L[i] = rho_h_W2 * velocity_L[i]; }
    return conserved;
}

namespace detail
{
inline double square(double x) { return x * x; }

inline double pressure_residual(double pressure, double D, double tau,
                                double S_squared, double gamma,
                                double max_velocity_squared)
{
    const double energy_plus_pressure = tau + D + pressure;
    double v_squared =
        S_squared / square(energy_plus_pressure);
    if (v_squared > max_velocity_squared)
        v_squared = max_velocity_squared;
    const double W = 1.0 / std::sqrt(1.0 - v_squared);
    return D * W + gamma * pressure * W * W / (gamma - 1.0) -
           energy_plus_pressure;
}

inline double pressure_residual_derivative(double pressure, double D,
                                           double tau, double S_squared,
                                           double gamma,
                                           double max_velocity_squared)
{
    const double energy_plus_pressure = tau + D + pressure;
    double v_squared =
        S_squared / square(energy_plus_pressure);
    if (v_squared > max_velocity_squared)
        v_squared = max_velocity_squared;
    const double W = 1.0 / std::sqrt(1.0 - v_squared);
    const double dW_dp = -W * W * W * v_squared / energy_plus_pressure;
    const double dW2_dp = 2.0 * W * dW_dp;
    return D * dW_dp +
           gamma * (W * W + pressure * dW2_dp) / (gamma - 1.0) - 1.0;
}
} // namespace detail

//! Recover primitive variables from Valencia conserved variables.
//!
//! This scalar Newton/bisection solve is intended as the first robust baseline.
//! BoxLoops callers should use it in non-SIMD mode until a vectorized recovery
//! path is added.
template <class eos_t>
RecoveryResult<double>
recover_primitive(const Conserved<double> &conserved, const eos_t &eos,
                  const Tensor<2, double> &spatial_metric_UU,
                  const RecoveryOptions &options = RecoveryOptions())
{
    RecoveryResult<double> result;
    if (conserved.D <= 0.0)
        return result;

    const double gamma = eos.adiabatic_index();
    const double S_squared = TensorAlgebra::compute_dot_product(
        conserved.S_L, conserved.S_L, spatial_metric_UU);

    double lower = options.pressure_floor;
    double f_lower = detail::pressure_residual(
        lower, conserved.D, conserved.tau, S_squared, gamma,
        options.max_velocity_squared);

    if (f_lower >= 0.0 && std::abs(f_lower) > options.tolerance)
        return result;


    double upper = lower;
    double f_upper = f_lower;
    if (f_lower < 0.0)
    {
        upper = std::max(1.0, std::abs(conserved.tau) + conserved.D);
        f_upper = detail::pressure_residual(
            upper, conserved.D, conserved.tau, S_squared, gamma,
            options.max_velocity_squared);
        int bracket_iterations = 0;
        while (f_upper <= 0.0 &&
               bracket_iterations < options.max_iterations)
        {
            upper *= 2.0;
            f_upper = detail::pressure_residual(
                upper, conserved.D, conserved.tau, S_squared, gamma,
                options.max_velocity_squared);
            ++bracket_iterations;
        }

        if (f_upper <= 0.0)
            return result;
    }

    double pressure =
        (std::abs(f_lower) <= options.tolerance)
            ? lower
            : std::min(std::max(conserved.tau * (gamma - 1.0), lower), upper);
    double residual = detail::pressure_residual(
        pressure, conserved.D, conserved.tau, S_squared, gamma,
        options.max_velocity_squared);

    for (int iter = 0; iter < options.max_iterations; ++iter)
    {
        result.iterations = iter + 1;
        if (std::abs(residual) < options.tolerance)
            break;

        if (f_lower < 0.0)
        {
            if (residual > 0.0)
                upper = pressure;
            else
                lower = pressure;
        }

        const double derivative = detail::pressure_residual_derivative(
            pressure, conserved.D, conserved.tau, S_squared, gamma,
            options.max_velocity_squared);
        double next_pressure = pressure - residual / derivative;
        if (!std::isfinite(next_pressure) || next_pressure <= lower ||
            next_pressure >= upper)
        {
            next_pressure = 0.5 * (lower + upper);
        }

        pressure = next_pressure;
        residual = detail::pressure_residual(
            pressure, conserved.D, conserved.tau, S_squared, gamma,
            options.max_velocity_squared);
    }

    result.residual = residual;
    if (std::abs(residual) >= options.tolerance)
        return result;

    const double energy_plus_pressure = conserved.tau + conserved.D + pressure;
    const double v_squared = S_squared / detail::square(energy_plus_pressure);
    if (v_squared >= options.max_velocity_squared)
        return result;

    const double W = 1.0 / std::sqrt(1.0 - v_squared);
    result.primitive.rho = conserved.D / W;
    result.primitive.pressure = pressure;
    result.primitive.eps =
        pressure / ((gamma - 1.0) * result.primitive.rho);

    FOR(i)
    {
        result.primitive.velocity_U[i] = 0.0;
        FOR(j)
        {
            result.primitive.velocity_U[i] +=
                spatial_metric_UU[i][j] * conserved.S_L[j] /
                energy_plus_pressure;
        }
    }

    result.success = true;
    return result;
}

template <class data_t>
Flux<data_t> compute_flux(const Primitive<data_t> &primitive,
                          const Conserved<data_t> &conserved,
                          int direction, const data_t &lapse,
                          const Tensor<1, data_t> &shift_U)
{
    Flux<data_t> flux;
    const data_t transport_velocity =
        lapse * primitive.velocity_U[direction] - shift_U[direction];
    flux.D = conserved.D * transport_velocity;
    flux.tau = conserved.tau * transport_velocity +
               lapse * primitive.pressure * primitive.velocity_U[direction];
    FOR(j)
    {
        flux.S_L[j] = conserved.S_L[j] * transport_velocity;
        if (j == direction)
            flux.S_L[j] += lapse * primitive.pressure;
    }
    return flux;
}

template <class data_t>
Flux<data_t> compute_flux(const Primitive<data_t> &primitive,
                          const Conserved<data_t> &conserved, int direction)
{
    Tensor<1, data_t> zero_shift;
    FOR(i) { zero_shift[i] = 0.0; }
    return compute_flux(primitive, conserved, direction, data_t(1.0),
                        zero_shift);
}
} // namespace GRHD

#endif /* GRHD_VALENCIA_HPP_ */
