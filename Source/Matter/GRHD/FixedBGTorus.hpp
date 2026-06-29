/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef GRHD_FIXED_BG_TORUS_HPP_
#define GRHD_FIXED_BG_TORUS_HPP_

#include "Coordinates.hpp"
#include "FixedBGFArrayBoxFiniteVolume.hpp"
#include "TensorAlgebra.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace GRHD
{
struct FixedBGTorusConfig
{
    double total_mass = 1.0;
    double inner_radius = 0.0;
    double pressure_max_radius = 0.0;
    double rho_peak = 1.0;
    double max_velocity_squared = 1.0 - 1.0e-12;
    double radial_derivative_fraction = 1.0e-3;
    double radial_derivative_min_step = 1.0e-4;
    double angular_momentum_min_factor = 0.25;
    double angular_momentum_max_factor = 4.0;
    int angular_momentum_samples = 120;
    int angular_momentum_bisection_iterations = 80;
    double angular_momentum_tolerance = 1.0e-13;
};

struct FixedBGTorusParameters
{
    double ell = 0.0;
    double surface_potential = 0.0;
    double polytropic_constant = 0.0;
    double max_enthalpy = 1.0;
};

inline Tensor<1, double> fixed_bg_torus_azimuthal_basis(double x,
                                                        double y)
{
    Tensor<1, double> basis;
    FOR(dir) { basis[dir] = 0.0; }
    basis[0] = -y;
    basis[1] = x;
    return basis;
}

inline double fixed_bg_torus_azimuthal_metric(
    double x, double y, const CellGeometry &geometry)
{
    const Tensor<1, double> basis =
        fixed_bg_torus_azimuthal_basis(x, y);
    double g_phiphi = 0.0;
    FOR(i, j)
    {
        g_phiphi += geometry.spatial_metric_LL[i][j] * basis[i] * basis[j];
    }
    return g_phiphi;
}

inline double fixed_bg_torus_covariant_azimuthal_shift(
    double x, double y, const CellGeometry &geometry)
{
    const Tensor<1, double> basis =
        fixed_bg_torus_azimuthal_basis(x, y);
    Tensor<1, double> shift_L;
    FOR(i) { shift_L[i] = 0.0; }
    FOR(i, j)
    {
        shift_L[i] += geometry.spatial_metric_LL[i][j] *
                      geometry.shift_U[j];
    }

    double beta_phi = 0.0;
    FOR(i) { beta_phi += shift_L[i] * basis[i]; }
    return beta_phi;
}

inline double
fixed_bg_torus_shift_norm_squared(const CellGeometry &geometry)
{
    return TensorAlgebra::compute_dot_product(
        geometry.shift_U, geometry.shift_U, geometry.spatial_metric_LL);
}

inline double fixed_bg_torus_coordinate_angular_velocity(
    double x, double y, double ell, const CellGeometry &geometry)
{
    const double g_phiphi =
        fixed_bg_torus_azimuthal_metric(x, y, geometry);
    const double g_tphi =
        fixed_bg_torus_covariant_azimuthal_shift(x, y, geometry);
    const double g_tt = -geometry.lapse * geometry.lapse +
                        fixed_bg_torus_shift_norm_squared(geometry);
    const double denominator = ell * g_tphi + g_phiphi;
    if (std::abs(denominator) < 1.0e-300)
        return std::numeric_limits<double>::quiet_NaN();
    return -(ell * g_tt + g_tphi) / denominator;
}

inline Tensor<1, double>
fixed_bg_torus_eulerian_velocity_from_angular_velocity(
    double x, double y, double omega, const CellGeometry &geometry)
{
    const Tensor<1, double> basis =
        fixed_bg_torus_azimuthal_basis(x, y);
    Tensor<1, double> velocity_U;
    FOR(i) { velocity_U[i] = 0.0; }
    FOR(i)
    {
        velocity_U[i] =
            (omega * basis[i] + geometry.shift_U[i]) / geometry.lapse;
    }
    return velocity_U;
}

inline double fixed_bg_torus_circular_specific_angular_momentum(
    double mass, double areal_radius)
{
    if (areal_radius <= 3.0 * mass)
        throw std::runtime_error("circular orbit radius is too small");
    return std::sqrt(mass * areal_radius) /
           (1.0 - 2.0 * mass / areal_radius);
}

inline Coordinates<double> fixed_bg_torus_coordinates_at(double x, double y,
                                                         double z)
{
    std::array<double, CH_SPACEDIM> center;
    FOR(dir) { center[dir] = 0.0; }
    Coordinates<double> coords(IntVect(D_DECL(0, 0, 0)), 1.0, center);
    coords.x = x;
    coords.y = y;
    coords.z = z;
    return coords;
}

template <class background_t>
CellGeometry fixed_bg_torus_geometry_at(double x, double y, double z,
                                        const background_t &background)
{
    return make_cell_geometry_from_background(
        background, fixed_bg_torus_coordinates_at(x, y, z));
}

inline double fixed_bg_torus_potential_from_geometry(
    double x, double y, double ell, const CellGeometry &geometry,
    const FixedBGTorusConfig &config)
{
    const double coordinate_omega =
        fixed_bg_torus_coordinate_angular_velocity(x, y, ell, geometry);
    if (!std::isfinite(coordinate_omega))
        return std::numeric_limits<double>::infinity();

    const Tensor<1, double> velocity_U =
        fixed_bg_torus_eulerian_velocity_from_angular_velocity(
            x, y, coordinate_omega, geometry);
    const double velocity_squared =
        compute_velocity_squared(velocity_U, geometry.spatial_metric_LL);
    if (velocity_squared >= config.max_velocity_squared ||
        velocity_squared >= 1.0)
        return std::numeric_limits<double>::infinity();

    const double g_phiphi =
        fixed_bg_torus_azimuthal_metric(x, y, geometry);
    const double g_tphi =
        fixed_bg_torus_covariant_azimuthal_shift(x, y, geometry);
    const double g_tt = -geometry.lapse * geometry.lapse +
                        fixed_bg_torus_shift_norm_squared(geometry);
    const double normalization =
        -g_tt - 2.0 * coordinate_omega * g_tphi -
        coordinate_omega * coordinate_omega * g_phiphi;
    if (normalization <= 0.0)
        return std::numeric_limits<double>::infinity();

    const double minus_u_t =
        -(g_tt + coordinate_omega * g_tphi) / std::sqrt(normalization);
    if (minus_u_t <= 0.0)
        return std::numeric_limits<double>::infinity();
    return std::log(minus_u_t);
}

template <class background_t>
double fixed_bg_torus_potential(double x, double y, double z, double ell,
                                const background_t &background,
                                const FixedBGTorusConfig &config)
{
    const auto geometry =
        fixed_bg_torus_geometry_at(x, y, z, background);
    return fixed_bg_torus_potential_from_geometry(x, y, ell, geometry,
                                                  config);
}

template <class background_t>
double fixed_bg_torus_radial_potential_derivative(
    double radius, double ell, const background_t &background,
    const FixedBGTorusConfig &config)
{
    const double dr =
        std::max(config.radial_derivative_fraction * radius,
                 config.radial_derivative_min_step);
    const double r_minus = std::max(radius - dr, 1.0e-6);
    const double w_plus = fixed_bg_torus_potential(
        radius + dr, 0.0, 0.0, ell, background, config);
    const double w_minus = fixed_bg_torus_potential(
        r_minus, 0.0, 0.0, ell, background, config);
    if (!std::isfinite(w_plus) || !std::isfinite(w_minus))
        return std::numeric_limits<double>::quiet_NaN();
    return (w_plus - w_minus) / (radius + dr - r_minus);
}

template <class background_t>
double fixed_bg_torus_pressure_max_specific_angular_momentum(
    const background_t &background, const FixedBGTorusConfig &config)
{
    const double guess = fixed_bg_torus_circular_specific_angular_momentum(
        config.total_mass, config.pressure_max_radius);

    double previous_ell = config.angular_momentum_min_factor * guess;
    double previous_derivative =
        fixed_bg_torus_radial_potential_derivative(
            config.pressure_max_radius, previous_ell, background, config);

    for (int sample = 1; sample <= config.angular_momentum_samples;
         ++sample)
    {
        const double fraction =
            static_cast<double>(sample) / config.angular_momentum_samples;
        const double ell =
            guess * (config.angular_momentum_min_factor +
                     fraction * (config.angular_momentum_max_factor -
                                 config.angular_momentum_min_factor));
        const double derivative =
            fixed_bg_torus_radial_potential_derivative(
                config.pressure_max_radius, ell, background, config);
        if (!std::isfinite(derivative))
            continue;
        if (std::isfinite(previous_derivative) &&
            previous_derivative * derivative <= 0.0)
        {
            double lower = previous_ell;
            double upper = ell;
            double f_lower = previous_derivative;
            for (int iter = 0;
                 iter < config.angular_momentum_bisection_iterations;
                 ++iter)
            {
                const double mid = 0.5 * (lower + upper);
                const double f_mid =
                    fixed_bg_torus_radial_potential_derivative(
                        config.pressure_max_radius, mid, background,
                        config);
                if (!std::isfinite(f_mid))
                    break;
                if (std::abs(f_mid) <
                    config.angular_momentum_tolerance)
                    return mid;
                if (f_lower * f_mid <= 0.0)
                {
                    upper = mid;
                }
                else
                {
                    lower = mid;
                    f_lower = f_mid;
                }
            }
            return 0.5 * (lower + upper);
        }
        previous_ell = ell;
        previous_derivative = derivative;
    }

    return guess;
}

inline double fixed_bg_torus_specific_enthalpy_from_potential(
    double potential, const FixedBGTorusParameters &torus)
{
    if (!std::isfinite(potential))
        return 1.0;
    return std::exp(torus.surface_potential - potential);
}

template <class eos_t, class background_t>
FixedBGTorusParameters make_fixed_bg_torus_parameters(
    const FixedBGTorusConfig &config, const eos_t &eos,
    const background_t &background, const Box &domain_box, double dx,
    const std::array<double, CH_SPACEDIM> &center)
{
    FixedBGTorusParameters torus;
    torus.ell =
        fixed_bg_torus_pressure_max_specific_angular_momentum(background,
                                                              config);
    torus.surface_potential = fixed_bg_torus_potential(
        config.inner_radius, 0.0, 0.0, torus.ell, background, config);
    if (!std::isfinite(torus.surface_potential))
        throw std::runtime_error("torus surface potential is invalid");

    BoxIterator bit(domain_box);
    for (bit.begin(); bit.ok(); ++bit)
    {
        const IntVect iv = bit();
        const Coordinates<double> coords(iv, dx, center);
        const double radius =
            std::sqrt(coords.x * coords.x + coords.y * coords.y);
        if (radius < config.inner_radius)
            continue;
        const double potential = fixed_bg_torus_potential(
            coords.x, coords.y, coords.z, torus.ell, background, config);
        torus.max_enthalpy =
            std::max(torus.max_enthalpy,
                     fixed_bg_torus_specific_enthalpy_from_potential(
                         potential, torus));
    }

    if (torus.max_enthalpy <= 1.0)
        throw std::runtime_error("torus has no pressure-supported region");

    const double gamma = eos.adiabatic_index();
    torus.polytropic_constant =
        (torus.max_enthalpy - 1.0) * (gamma - 1.0) /
        (gamma * std::pow(config.rho_peak, gamma - 1.0));
    return torus;
}

template <class eos_t>
Primitive<double> fixed_bg_torus_primitive(
    double x, double y, double z, const eos_t &eos,
    const CellGeometry &geometry, const FixedBGTorusParameters &torus,
    const FixedBGTorusConfig &config,
    const AtmosphereOptions &atmosphere = AtmosphereOptions())
{
    Primitive<double> primitive =
        make_atmosphere_primitive<double>(eos, atmosphere);

    const double radius = std::sqrt(x * x + y * y);
    if (radius < config.inner_radius)
        return primitive;

    const double potential = fixed_bg_torus_potential_from_geometry(
        x, y, torus.ell, geometry, config);
    const double h =
        fixed_bg_torus_specific_enthalpy_from_potential(potential, torus);
    if (h <= 1.0)
        return primitive;

    const double gamma = eos.adiabatic_index();
    primitive.rho =
        std::pow((h - 1.0) * (gamma - 1.0) /
                     (gamma * torus.polytropic_constant),
                 1.0 / (gamma - 1.0));
    primitive.pressure =
        torus.polytropic_constant * std::pow(primitive.rho, gamma);
    primitive.eps =
        primitive.pressure / ((gamma - 1.0) * primitive.rho);

    const double coordinate_omega =
        fixed_bg_torus_coordinate_angular_velocity(x, y, torus.ell,
                                                   geometry);
    primitive.velocity_U =
        fixed_bg_torus_eulerian_velocity_from_angular_velocity(
            x, y, coordinate_omega, geometry);

    enforce_primitive_floors(primitive, eos, geometry.spatial_metric_LL,
                             atmosphere);
    return primitive;
}

template <class eos_t, class background_factory_t>
void fill_fixed_bg_torus_level_data(
    LevelData<FArrayBox> &state,
    const background_factory_t &background_factory, const eos_t &eos,
    double time, double dx,
    const std::array<double, CH_SPACEDIM> &center,
    const FixedBGTorusParameters &torus,
    const FixedBGTorusConfig &config,
    const AtmosphereOptions &atmosphere = AtmosphereOptions())
{
    const auto background = background_factory(time);
    const DisjointBoxLayout &grids = state.disjointBoxLayout();
    DataIterator dit = grids.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        state[dit].setVal(0.0);
        BoxIterator bit(state[dit].box());
        for (bit.begin(); bit.ok(); ++bit)
        {
            const IntVect iv = bit();
            const Coordinates<double> coords(iv, dx, center);
            const auto geometry =
                load_fixed_background_geometry(background, iv, dx, center);
            auto primitive = fixed_bg_torus_primitive(
                coords.x, coords.y, coords.z, eos, geometry, torus,
                config, atmosphere);
            store_primitive(state[dit], iv, primitive);
            store_conserved(state[dit], iv,
                            compute_conserved(
                                primitive, eos,
                                geometry.spatial_metric_LL));
        }
    }
}
} // namespace GRHD

#endif /* GRHD_FIXED_BG_TORUS_HPP_ */
