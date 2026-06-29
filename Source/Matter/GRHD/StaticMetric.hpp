/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef GRHD_STATIC_METRIC_HPP_
#define GRHD_STATIC_METRIC_HPP_

#include "DimensionDefinitions.hpp"
#include "FiniteVolume.hpp"
#include "TensorAlgebra.hpp"
#include "Valencia.hpp"
#include <array>
#include <cmath>
#include <stdexcept>

namespace GRHD
{
template <class data_t> struct CellGeometryData
{
    Tensor<2, data_t> spatial_metric_LL;
    Tensor<2, data_t> spatial_metric_UU;
    data_t lapse;
    Tensor<1, data_t> shift_U;
    Tensor<2, data_t> extrinsic_curvature_LL;

    CellGeometryData() : lapse(0.0)
    {
        FOR(i)
        {
            shift_U[i] = 0.0;
            FOR(j)
            {
                spatial_metric_LL[i][j] = 0.0;
                spatial_metric_UU[i][j] = 0.0;
                extrinsic_curvature_LL[i][j] = 0.0;
            }
        }
    }
};

using CellGeometry = CellGeometryData<double>;

inline CellGeometry make_flat_cell_geometry()
{
    CellGeometry geometry;
    FOR(i)
    {
        geometry.spatial_metric_LL[i][i] = 1.0;
        geometry.spatial_metric_UU[i][i] = 1.0;
    }
    geometry.lapse = 1.0;
    FOR(i) { geometry.shift_U[i] = 0.0; }
    return geometry;
}

inline double schwarzschild_areal_f(double mass, double radius)
{
    return 1.0 - 2.0 * mass / radius;
}

inline CellGeometry make_schwarzschild_radial_line_geometry(double mass,
                                                           double radius)
{
    const double f = schwarzschild_areal_f(mass, radius);
    if (f <= 0.0)
    {
        throw std::runtime_error(
            "Schwarzschild radial-line geometry is inside the horizon");
    }

    CellGeometry geometry = make_flat_cell_geometry();
    geometry.spatial_metric_LL[0][0] = 1.0 / f;
    geometry.spatial_metric_UU[0][0] = f;
    geometry.lapse = std::sqrt(f);
    return geometry;
}

template <class data_t> struct StaticMetricDerivatives
{
    Tensor<1, data_t> lapse;
    Tensor<1, data_t> log_sqrt_det_spatial_metric;
    std::array<Tensor<2, data_t>, CH_SPACEDIM> spatial_metric_LL;
    Tensor<2, data_t> shift_U;

    StaticMetricDerivatives()
    {
        FOR(direction)
        {
            lapse[direction] = 0.0;
            log_sqrt_det_spatial_metric[direction] = 0.0;
            FOR(i)
            {
                shift_U[i][direction] = 0.0;
                FOR(j)
                {
                    spatial_metric_LL[direction][i][j] = 0.0;
                }
            }
        }
    }
};

inline StaticMetricDerivatives<double>
make_schwarzschild_radial_line_derivatives(double mass, double radius)
{
    const double f = schwarzschild_areal_f(mass, radius);
    if (f <= 0.0)
    {
        throw std::runtime_error(
            "Schwarzschild radial-line derivatives are inside the horizon");
    }

    StaticMetricDerivatives<double> derivatives;
    derivatives.lapse[0] = mass / (radius * radius * std::sqrt(f));
    derivatives.log_sqrt_det_spatial_metric[0] =
        -mass / (radius * radius * f);
    derivatives.spatial_metric_LL[0][0][0] =
        -2.0 * mass / (radius * radius * f * f);
    return derivatives;
}

inline double radius_from_position(const Tensor<1, double> &position)
{
    double radius_squared = 0.0;
    FOR(i) { radius_squared += position[i] * position[i]; }
    return std::sqrt(radius_squared);
}

inline CellGeometry make_schwarzschild_isotropic_geometry(
    double mass, const Tensor<1, double> &position)
{
    const double radius = radius_from_position(position);
    if (radius <= 0.0 || (mass > 0.0 && radius <= 0.5 * mass))
    {
        throw std::runtime_error(
            "Schwarzschild isotropic geometry is inside the horizon");
    }

    const double q = mass / (2.0 * radius);
    const double psi = 1.0 + q;
    const double conformal_factor = std::pow(psi, 4);

    CellGeometry geometry = make_flat_cell_geometry();
    FOR(i)
    {
        geometry.spatial_metric_LL[i][i] = conformal_factor;
        geometry.spatial_metric_UU[i][i] = 1.0 / conformal_factor;
    }
    geometry.lapse = (1.0 - q) / (1.0 + q);
    return geometry;
}

inline StaticMetricDerivatives<double> make_schwarzschild_isotropic_derivatives(
    double mass, const Tensor<1, double> &position)
{
    const double radius = radius_from_position(position);
    if (radius <= 0.0 || (mass > 0.0 && radius <= 0.5 * mass))
    {
        throw std::runtime_error(
            "Schwarzschild isotropic derivatives are inside the horizon");
    }

    StaticMetricDerivatives<double> derivatives;
    if (mass == 0.0)
    {
        return derivatives;
    }

    const double q = mass / (2.0 * radius);
    const double psi = 1.0 + q;
    const double d_lapse_dr = mass /
                              (radius * radius * (1.0 + q) * (1.0 + q));
    const double d_metric_dr = -2.0 * mass * std::pow(psi, 3) /
                               (radius * radius);
    const double d_log_sqrt_det_dr = -3.0 * mass / (radius * radius * psi);

    FOR(direction)
    {
        const double radial_unit = position[direction] / radius;
        derivatives.lapse[direction] = d_lapse_dr * radial_unit;
        derivatives.log_sqrt_det_spatial_metric[direction] =
            d_log_sqrt_det_dr * radial_unit;
        FOR(metric_dir)
        {
            derivatives.spatial_metric_LL[direction][metric_dir][metric_dir] =
                d_metric_dr * radial_unit;
        }
    }
    return derivatives;
}

template <class data_t, class eos_t>
Tensor<2, data_t> compute_spatial_stress_UU(
    const Primitive<data_t> &primitive, const eos_t &eos,
    const CellGeometryData<data_t> &geometry)
{
    const data_t W = compute_lorentz_factor(primitive.velocity_U,
                                            geometry.spatial_metric_LL);
    const data_t h = eos.compute_specific_enthalpy(
        primitive.rho, primitive.eps, primitive.pressure);
    const data_t rho_h_W2 = primitive.rho * h * W * W;

    Tensor<2, data_t> stress_UU = data_t(0.0);
    FOR(i, j)
    {
        stress_UU[i][j] = rho_h_W2 * primitive.velocity_U[i] *
                          primitive.velocity_U[j] +
                          primitive.pressure * geometry.spatial_metric_UU[i][j];
    }
    return stress_UU;
}

template <class data_t, class eos_t>
Conserved<data_t> compute_static_metric_source_terms(
    const Primitive<data_t> &primitive, const Conserved<data_t> &conserved,
    const eos_t &eos, const CellGeometryData<data_t> &geometry,
    const StaticMetricDerivatives<data_t> &derivatives)
{
    Conserved<data_t> source = zero_conserved<data_t>();
    const auto stress_UU = compute_spatial_stress_UU(primitive, eos,
                                                     geometry);
    const data_t energy_density = conserved.D + conserved.tau;

    Tensor<1, data_t> momentum_U;
    FOR(i) { momentum_U[i] = 0.0; }
    FOR(i, j)
    {
        momentum_U[i] += geometry.spatial_metric_UU[i][j] * conserved.S_L[j];
    }

    std::array<Flux<data_t>, CH_SPACEDIM> cell_fluxes;
    FOR(direction)
    {
        cell_fluxes[direction] = compute_flux(
            primitive, conserved, direction, geometry.lapse, geometry.shift_U);
    }

    FOR(direction)
    {
        source.D -= cell_fluxes[direction].D *
                    derivatives.log_sqrt_det_spatial_metric[direction];
        source.tau -= cell_fluxes[direction].tau *
                      derivatives.log_sqrt_det_spatial_metric[direction];
        source.tau -= momentum_U[direction] * derivatives.lapse[direction];
    }

    FOR(i, j)
    {
        source.tau += geometry.lapse * stress_UU[i][j] *
                      geometry.extrinsic_curvature_LL[i][j];
    }

    FOR(momentum_dir)
    {
        data_t momentum_source = -energy_density * derivatives.lapse[momentum_dir];
        FOR(i, j)
        {
            momentum_source += data_t(0.5) * geometry.lapse * stress_UU[i][j] *
                               derivatives.spatial_metric_LL[momentum_dir][i][j];
        }
        FOR(shift_dir)
        {
            momentum_source += conserved.S_L[shift_dir] *
                               derivatives.shift_U[shift_dir][momentum_dir];
        }
        FOR(direction)
        {
            momentum_source -= cell_fluxes[direction].S_L[momentum_dir] *
                               derivatives.log_sqrt_det_spatial_metric[direction];
        }
        source.S_L[momentum_dir] += momentum_source;
    }
    return source;
}

template <class data_t>
Conserved<data_t> compute_metric_volume_time_source_terms(
    const Conserved<data_t> &conserved,
    const data_t &d_log_sqrt_det_spatial_metric_dt)
{
    Conserved<data_t> source = zero_conserved<data_t>();
    source.D -= conserved.D * d_log_sqrt_det_spatial_metric_dt;
    source.tau -= conserved.tau * d_log_sqrt_det_spatial_metric_dt;
    FOR(i)
    {
        source.S_L[i] -=
            conserved.S_L[i] * d_log_sqrt_det_spatial_metric_dt;
    }
    return source;
}

inline double spatial_metric_log_sqrt_det(const CellGeometry &geometry)
{
    const double determinant = TensorAlgebra::compute_determinant_sym(
        geometry.spatial_metric_LL);
    if (determinant <= 0.0 || !std::isfinite(determinant))
    {
        return 0.0;
    }
    return 0.5 * std::log(determinant);
}
} // namespace GRHD

#endif /* GRHD_STATIC_METRIC_HPP_ */
