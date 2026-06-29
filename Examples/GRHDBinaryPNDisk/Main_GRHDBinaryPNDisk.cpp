/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#include "DimensionDefinitions.hpp"
#include "Atmosphere.hpp"
#include "FiniteVolume.hpp"
#include "FixedBGMetric.hpp"
#include "IdealGasEOS.hpp"
#include "CircularBinaryPN.hpp"
#include "Reconstruction.hpp"
#include "RiemannSolvers.hpp"
#include "StaticMetric.hpp"
#include "Valencia.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

namespace
{
const double PI = 3.141592653589793238462643383279502884;

struct SimulationConfig
{
    int num_cells = 80;
    int num_ghosts = 2;
    double x_min = -50.0;
    double x_max = 50.0;
    double final_time = 15.0;
    double cfl = 0.22;
    double frame_interval = 1.0;
    double mass_1 = 0.5;
    double mass_2 = 0.5;
    double separation = 8.0;
    double binary_phase = 0.0;
    double binary_time = 0.0;
    double binary_angular_velocity = 0.0;
    bool evolve_binary_time = true;
    bool use_time_volume_source = true;
    double metric_time_derivative_step = 1.0e-3;
    double softening_radius = 1.0;
    double excision_radius = 0.0;
    double torus_inner_radius = 14.0;
    double torus_pressure_max_radius = 18.0;
    double rho_peak = 1.0e-5;
    double limiter_theta = 1.0;
    double rho_floor = 1.0e-10;
    double pressure_floor = 1.0e-12;
    double max_velocity_squared = 0.95;
    double atmosphere_repair_factor = 1000.0;
    int image_scale = 8;
    bool use_reconstruction = true;
};

struct TorusParameters
{
    double ell = 0.0;
    double surface_potential = 0.0;
    double polytropic_constant = 0.0;
    double max_enthalpy = 1.0;
};

struct Diagnostics
{
    double mass = 0.0;
    double disk_mass = 0.0;
    double tau = 0.0;
    double max_rho = 0.0;
    double max_pressure = 0.0;
    double max_velocity = 0.0;
    double max_lorentz_factor = 1.0;
    int atmosphere_resets = 0;
};

struct RecoveryScratch
{
    std::vector<GRHD::Primitive<double>> primitives;
    int resets = 0;
};

int total_cells_1d(const SimulationConfig &config)
{
    return config.num_cells + 2 * config.num_ghosts;
}

int index(int i, int j, const SimulationConfig &config)
{
    const int n_total = total_cells_1d(config);
    return j * n_total + i;
}

double coordinate(int i, double dx, const SimulationConfig &config)
{
    return config.x_min + (i - config.num_ghosts + 0.5) * dx;
}

double interface_coordinate(int i_left, double dx,
                            const SimulationConfig &config)
{
    return config.x_min + (i_left - config.num_ghosts + 1.0) * dx;
}

double total_mass(const SimulationConfig &config)
{
    return config.mass_1 + config.mass_2;
}

bool has_excision(const SimulationConfig &config)
{
    return config.excision_radius > 0.0;
}

SimulationConfig config_at_time(const SimulationConfig &config, double time)
{
    SimulationConfig time_config = config;
    if (time_config.evolve_binary_time)
        time_config.binary_time += time;
    return time_config;
}

Tensor<1, double> regularized_position(double x, double y,
                                        const SimulationConfig &config)
{
    Tensor<1, double> position;
    FOR(i) { position[i] = 0.0; }
    const double radius = std::sqrt(x * x + y * y);
    if (!has_excision(config) || radius >= config.excision_radius)
    {
        position[0] = x;
        position[1] = y;
        return position;
    }

    const double scale = config.excision_radius / std::max(radius, 1.0e-300);
    if (radius > 0.0)
    {
        position[0] = x * scale;
        position[1] = y * scale;
    }
    else
    {
        position[0] = config.excision_radius;
    }
    return position;
}

CircularBinaryPN::params_t background_params(const SimulationConfig &config)
{
    CircularBinaryPN::params_t params;
    params.mass_1 = config.mass_1;
    params.mass_2 = config.mass_2;
    params.separation = config.separation;
    params.angular_velocity = config.binary_angular_velocity;
    params.phase = config.binary_phase;
    params.time = config.binary_time;
    params.softening_radius = config.softening_radius;
    params.center = {{0.0, 0.0, 0.0}};
    return params;
}

Coordinates<double> coordinates_at(double x, double y,
                                   const SimulationConfig &config)
{
    const auto position = regularized_position(x, y, config);
    Coordinates<double> coords(IntVect(D_DECL(0, 0, 0)), 1.0,
                               {0.0, 0.0, 0.0});
    coords.x = position[0];
    coords.y = position[1];
    coords.z = 0.0;
    return coords;
}

GRHD::CellGeometry geometry_at(double x, double y,
                               const SimulationConfig &config)
{
    CircularBinaryPN background(background_params(config), 1.0);
    return GRHD::make_cell_geometry_from_background(
        background, coordinates_at(x, y, config));
}

GRHD::StaticMetricDerivatives<double>
metric_derivatives_at(double x, double y, const SimulationConfig &config)
{
    CircularBinaryPN background(background_params(config), 1.0);
    return GRHD::make_static_metric_derivatives_from_background(
        background, coordinates_at(x, y, config));
}

double log_sqrt_det_at(double x, double y, const SimulationConfig &config)
{
    const auto geometry = geometry_at(x, y, config);
    return GRHD::spatial_metric_log_sqrt_det(geometry);
}

double log_sqrt_det_time_derivative_at(double x, double y,
                                        const SimulationConfig &config)
{
    if (!config.evolve_binary_time || !config.use_time_volume_source)
        return 0.0;

    const double dt = std::max(config.metric_time_derivative_step, 1.0e-12);
    SimulationConfig plus = config;
    SimulationConfig minus = config;
    plus.binary_time += dt;
    minus.binary_time -= dt;
    return (log_sqrt_det_at(x, y, plus) - log_sqrt_det_at(x, y, minus)) /
           (2.0 * dt);
}

Tensor<1, double> azimuthal_basis(double x, double y)
{
    Tensor<1, double> basis;
    FOR(i) { basis[i] = 0.0; }
    basis[0] = -y;
    basis[1] = x;
    return basis;
}

double azimuthal_metric(double x, double y,
                        const GRHD::CellGeometry &geometry)
{
    const Tensor<1, double> basis = azimuthal_basis(x, y);

    double g_phiphi = 0.0;
    FOR(i, j)
    {
        g_phiphi +=
            geometry.spatial_metric_LL[i][j] * basis[i] * basis[j];
    }
    return g_phiphi;
}

double covariant_azimuthal_shift(double x, double y,
                                 const GRHD::CellGeometry &geometry)
{
    const Tensor<1, double> basis = azimuthal_basis(x, y);
    Tensor<1, double> shift_L;
    FOR(i) { shift_L[i] = 0.0; }
    FOR(i, j)
    {
        shift_L[i] += geometry.spatial_metric_LL[i][j] * geometry.shift_U[j];
    }

    double beta_phi = 0.0;
    FOR(i) { beta_phi += shift_L[i] * basis[i]; }
    return beta_phi;
}

double shift_norm_squared(const GRHD::CellGeometry &geometry)
{
    return TensorAlgebra::compute_dot_product(
        geometry.shift_U, geometry.shift_U, geometry.spatial_metric_LL);
}

double coordinate_angular_velocity(double x, double y, double ell,
                                   const GRHD::CellGeometry &geometry)
{
    const double g_phiphi = azimuthal_metric(x, y, geometry);
    const double g_tphi = covariant_azimuthal_shift(x, y, geometry);
    const double g_tt =
        -geometry.lapse * geometry.lapse + shift_norm_squared(geometry);
    const double denominator = ell * g_tphi + g_phiphi;
    if (std::abs(denominator) < 1.0e-300)
        return std::numeric_limits<double>::quiet_NaN();
    return -(ell * g_tt + g_tphi) / denominator;
}

Tensor<1, double>
eulerian_velocity_from_angular_velocity(double x, double y, double omega,
                                        const GRHD::CellGeometry &geometry)
{
    const Tensor<1, double> basis = azimuthal_basis(x, y);
    Tensor<1, double> velocity_U;
    FOR(i) { velocity_U[i] = 0.0; }
    FOR(i)
    {
        velocity_U[i] =
            (omega * basis[i] + geometry.shift_U[i]) / geometry.lapse;
    }
    return velocity_U;
}

double isotropic_to_areal_radius(double mass, double radius)
{
    const double q = mass / (2.0 * radius);
    const double psi = 1.0 + q;
    return radius * psi * psi;
}

double areal_to_radius(double mass, double areal_radius)
{
    if (areal_radius <= 2.0 * mass)
        throw std::runtime_error("areal radius is inside the horizon");
    return 0.5 * (areal_radius - mass +
                  std::sqrt(areal_radius * (areal_radius - 2.0 * mass)));
}

double circular_specific_angular_momentum(double mass, double areal_radius)
{
    if (areal_radius <= 3.0 * mass)
        throw std::runtime_error("circular orbit radius is too small");
    return std::sqrt(mass * areal_radius) /
           (1.0 - 2.0 * mass / areal_radius);
}

double torus_inner_radius(const SimulationConfig &config)
{
    return config.torus_inner_radius;
}

double torus_potential(double x, double y, double ell,
                       const SimulationConfig &config)
{
    const double radius = std::sqrt(x * x + y * y);
    if (has_excision(config) && radius <= config.excision_radius)
        return std::numeric_limits<double>::infinity();

    const auto geometry = geometry_at(x, y, config);
    const double coordinate_omega =
        coordinate_angular_velocity(x, y, ell, geometry);
    if (!std::isfinite(coordinate_omega))
        return std::numeric_limits<double>::infinity();

    const Tensor<1, double> velocity_U =
        eulerian_velocity_from_angular_velocity(x, y, coordinate_omega,
                                                geometry);
    const double velocity_squared =
        GRHD::compute_velocity_squared(velocity_U,
                                       geometry.spatial_metric_LL);
    if (velocity_squared >= config.max_velocity_squared ||
        velocity_squared >= 1.0)
        return std::numeric_limits<double>::infinity();

    const double g_phiphi = azimuthal_metric(x, y, geometry);
    const double g_tphi = covariant_azimuthal_shift(x, y, geometry);
    const double g_tt =
        -geometry.lapse * geometry.lapse + shift_norm_squared(geometry);
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


double radial_potential_derivative(double radius, double ell,
                                   const SimulationConfig &config)
{
    const double dr = std::max(1.0e-3 * radius, 1.0e-4);
    const double r_minus = std::max(radius - dr, 1.0e-6);
    const double w_plus = torus_potential(radius + dr, 0.0, ell, config);
    const double w_minus = torus_potential(r_minus, 0.0, ell, config);
    if (!std::isfinite(w_plus) || !std::isfinite(w_minus))
        return std::numeric_limits<double>::quiet_NaN();
    return (w_plus - w_minus) / (radius + dr - r_minus);
}

double pressure_max_specific_angular_momentum(
    const SimulationConfig &config)
{
    const double guess = circular_specific_angular_momentum(
        total_mass(config), config.torus_pressure_max_radius);

    const double min_factor = 0.25;
    const double max_factor = 4.0;
    const int samples = 120;
    double previous_ell = min_factor * guess;
    double previous_derivative = radial_potential_derivative(
        config.torus_pressure_max_radius, previous_ell, config);

    for (int sample = 1; sample <= samples; ++sample)
    {
        const double fraction = static_cast<double>(sample) / samples;
        const double ell = guess * (min_factor + fraction *
                                    (max_factor - min_factor));
        const double derivative = radial_potential_derivative(
            config.torus_pressure_max_radius, ell, config);
        if (!std::isfinite(derivative))
            continue;
        if (std::isfinite(previous_derivative) &&
            previous_derivative * derivative <= 0.0)
        {
            double lower = previous_ell;
            double upper = ell;
            double f_lower = previous_derivative;
            for (int iter = 0; iter < 80; ++iter)
            {
                const double mid = 0.5 * (lower + upper);
                const double f_mid = radial_potential_derivative(
                    config.torus_pressure_max_radius, mid, config);
                if (!std::isfinite(f_mid))
                    break;
                if (std::abs(f_mid) < 1.0e-13)
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

GRHD::AtmosphereOptions atmosphere_options(const SimulationConfig &config)
{
    GRHD::AtmosphereOptions options;
    options.rho_floor = config.rho_floor;
    options.pressure_floor = config.pressure_floor;
    options.max_velocity_squared = config.max_velocity_squared;
    return options;
}

GRHD::Primitive<double> atmosphere_primitive(const IdealGasEOS &eos,
                                             const SimulationConfig &config)
{
    return GRHD::make_atmosphere_primitive<double>(
        eos, atmosphere_options(config));
}

GRHD::Conserved<double> atmosphere_conserved(const IdealGasEOS &eos,
                                             const GRHD::CellGeometry &geometry,
                                             const SimulationConfig &config)
{
    return GRHD::compute_atmosphere_conserved<double>(
        eos, geometry.spatial_metric_LL, atmosphere_options(config));
}

bool is_low_density_atmosphere_state(
    const GRHD::Conserved<double> &conserved, const IdealGasEOS &eos,
    const GRHD::CellGeometry &geometry, const SimulationConfig &config)
{
    const auto atmosphere = atmosphere_conserved(eos, geometry, config);
    const double factor = config.atmosphere_repair_factor;
    const double tau_scale =
        std::max(std::abs(atmosphere.tau), config.pressure_floor);
    const double momentum_squared = TensorAlgebra::compute_dot_product(
        conserved.S_L, conserved.S_L, geometry.spatial_metric_UU);
    const double momentum_limit = factor * std::max(atmosphere.D, config.rho_floor);

    return conserved.D <= factor * atmosphere.D &&
           conserved.tau <= factor * tau_scale &&
           momentum_squared <= momentum_limit * momentum_limit;
}

GRHD::Primitive<double> zero_primitive()
{
    GRHD::Primitive<double> primitive;
    primitive.rho = 0.0;
    primitive.eps = 0.0;
    primitive.pressure = 0.0;
    FOR(i) { primitive.velocity_U[i] = 0.0; }
    return primitive;
}

GRHD::Conserved<double> zero_conserved()
{
    return GRHD::zero_conserved<double>();
}

double specific_enthalpy_from_potential(double potential,
                                        const TorusParameters &torus)
{
    if (!std::isfinite(potential))
        return 1.0;
    return std::exp(torus.surface_potential - potential);
}

TorusParameters make_torus_parameters(const SimulationConfig &config,
                                      const IdealGasEOS &eos, double dx)
{
    TorusParameters torus;
    torus.ell = pressure_max_specific_angular_momentum(config);
    const double inner_radius = torus_inner_radius(config);
    torus.surface_potential =
        torus_potential(inner_radius, 0.0, torus.ell, config);
    if (!std::isfinite(torus.surface_potential))
        throw std::runtime_error("torus surface potential is invalid");

    const int lo = config.num_ghosts;
    const int hi = config.num_ghosts + config.num_cells;
    for (int j = lo; j < hi; ++j)
    {
        const double y = coordinate(j, dx, config);
        for (int i = lo; i < hi; ++i)
        {
            const double x = coordinate(i, dx, config);
            if (std::sqrt(x * x + y * y) < inner_radius)
                continue;
            const double potential = torus_potential(x, y, torus.ell, config);
            torus.max_enthalpy =
                std::max(torus.max_enthalpy,
                         specific_enthalpy_from_potential(potential, torus));
        }
    }

    if (torus.max_enthalpy <= 1.0)
        throw std::runtime_error("torus has no pressure-supported region");

    const double gamma = eos.adiabatic_index();
    torus.polytropic_constant =
        (torus.max_enthalpy - 1.0) * (gamma - 1.0) /
        (gamma * std::pow(config.rho_peak, gamma - 1.0));
    return torus;
}

GRHD::Primitive<double> torus_primitive(double x, double y,
                                        const TorusParameters &torus,
                                        const IdealGasEOS &eos,
                                        const SimulationConfig &config)
{
    GRHD::Primitive<double> primitive = atmosphere_primitive(eos, config);
    const double radius = std::sqrt(x * x + y * y);
    if ((has_excision(config) && radius <= config.excision_radius) ||
        radius < torus_inner_radius(config))
        return primitive;

    const double potential = torus_potential(x, y, torus.ell, config);
    const double h = specific_enthalpy_from_potential(potential, torus);
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

    const auto geometry = geometry_at(x, y, config);
    const double coordinate_omega =
        coordinate_angular_velocity(x, y, torus.ell, geometry);
    primitive.velocity_U = eulerian_velocity_from_angular_velocity(
        x, y, coordinate_omega, geometry);

    GRHD::enforce_primitive_floors(primitive, eos, geometry.spatial_metric_LL,
                                   atmosphere_options(config));
    return primitive;
}

bool is_excision_cell(int i, int j, double dx, const SimulationConfig &config)
{
    const double x = coordinate(i, dx, config);
    const double y = coordinate(j, dx, config);
    return has_excision(config) &&
           std::sqrt(x * x + y * y) <= config.excision_radius;
}

void apply_outer_boundaries(std::vector<GRHD::Conserved<double>> &state,
                            const SimulationConfig &config)
{
    const int n_total = total_cells_1d(config);
    const int lo = config.num_ghosts;
    const int hi = config.num_ghosts + config.num_cells - 1;
    for (int j = 0; j < n_total; ++j)
    {
        const int jj = std::min(std::max(j, lo), hi);
        for (int i = 0; i < n_total; ++i)
        {
            if (i >= lo && i <= hi && j >= lo && j <= hi)
                continue;
            const int ii = std::min(std::max(i, lo), hi);
            state[index(i, j, config)] = state[index(ii, jj, config)];
        }
    }
}

void reset_excision(std::vector<GRHD::Conserved<double>> &state,
                    const IdealGasEOS &eos, const SimulationConfig &config,
                    double dx)
{
    const int n_total = total_cells_1d(config);
    for (int j = 0; j < n_total; ++j)
    {
        const double y = coordinate(j, dx, config);
        for (int i = 0; i < n_total; ++i)
        {
            const double x = coordinate(i, dx, config);
            if (!has_excision(config) ||
                std::sqrt(x * x + y * y) > config.excision_radius)
                continue;
            const auto geometry = geometry_at(x, y, config);
            state[index(i, j, config)] =
                atmosphere_conserved(eos, geometry, config);
        }
    }
}

RecoveryScratch recover_all(std::vector<GRHD::Conserved<double>> &state,
                            const IdealGasEOS &eos,
                            const SimulationConfig &config, double dx)
{
    GRHD::RecoveryOptions options;
    options.max_iterations = 80;
    options.pressure_floor = config.pressure_floor;
    options.max_velocity_squared = config.max_velocity_squared;

    RecoveryScratch scratch;
    scratch.primitives.resize(state.size());
    const int n_total = total_cells_1d(config);
    for (int j = 0; j < n_total; ++j)
    {
        const double y = coordinate(j, dx, config);
        for (int i = 0; i < n_total; ++i)
        {
            const double x = coordinate(i, dx, config);
            const int idx = index(i, j, config);
            const auto geometry = geometry_at(x, y, config);
            if (has_excision(config) &&
                std::sqrt(x * x + y * y) <= config.excision_radius)
            {
                scratch.primitives[idx] = atmosphere_primitive(eos, config);
                state[idx] = atmosphere_conserved(eos, geometry, config);
                continue;
            }

            if (is_low_density_atmosphere_state(state[idx], eos, geometry,
                                                config))
            {
                scratch.primitives[idx] = atmosphere_primitive(eos, config);
                state[idx] = atmosphere_conserved(eos, geometry, config);
                continue;
            }

            const auto recovered = GRHD::recover_primitive(
                state[idx], eos, geometry.spatial_metric_UU, options);
            if (!recovered.success ||
                !std::isfinite(recovered.primitive.rho) ||
                !std::isfinite(recovered.primitive.pressure))
            {
                scratch.primitives[idx] = atmosphere_primitive(eos, config);
                state[idx] = atmosphere_conserved(eos, geometry, config);
                ++scratch.resets;
                continue;
            }

            scratch.primitives[idx] = recovered.primitive;
            GRHD::enforce_primitive_floors(scratch.primitives[idx], eos,
                                           geometry.spatial_metric_LL,
                                           atmosphere_options(config));
            state[idx] = GRHD::compute_conserved(
                scratch.primitives[idx], eos, geometry.spatial_metric_LL);
        }
    }
    return scratch;
}

void repair_state(std::vector<GRHD::Conserved<double>> &state,
                  const IdealGasEOS &eos, const SimulationConfig &config,
                  double dx, int &reset_count)
{
    apply_outer_boundaries(state, config);
    reset_excision(state, eos, config, dx);
    RecoveryScratch scratch = recover_all(state, eos, config, dx);
    reset_count += scratch.resets;
    apply_outer_boundaries(state, config);
}

std::vector<GRHD::Primitive<double>> compute_slopes(
    const std::vector<GRHD::Primitive<double>> &primitives,
    const SimulationConfig &config, int direction)
{
    const int n_total = total_cells_1d(config);
    std::vector<GRHD::Primitive<double>> slopes(primitives.size());
    for (int idx = 0; idx < static_cast<int>(slopes.size()); ++idx)
        slopes[idx] = zero_primitive();

    for (int j = 0; j < n_total; ++j)
    {
        for (int i = 0; i < n_total; ++i)
        {
            int i_left = i;
            int i_right = i;
            int j_left = j;
            int j_right = j;
            if (direction == 0)
            {
                i_left = i - 1;
                i_right = i + 1;
            }
            else
            {
                j_left = j - 1;
                j_right = j + 1;
            }
            if (i_left < 0 || i_right >= n_total || j_left < 0 ||
                j_right >= n_total)
                continue;
            slopes[index(i, j, config)] = GRHD::compute_limited_slope(
                primitives[index(i_left, j_left, config)],
                primitives[index(i, j, config)],
                primitives[index(i_right, j_right, config)],
                config.limiter_theta);
        }
    }
    return slopes;
}

std::array<std::vector<GRHD::Primitive<double>>, 2>
compute_all_slopes(const std::vector<GRHD::Primitive<double>> &primitives,
                   const SimulationConfig &config)
{
    return {compute_slopes(primitives, config, 0),
            compute_slopes(primitives, config, 1)};
}

double compute_time_step(const std::vector<GRHD::Primitive<double>> &primitives,
                         const IdealGasEOS &eos,
                         const SimulationConfig &config, double dx)
{
    double max_speed_sum = 0.0;
    const int lo = config.num_ghosts;
    const int hi = config.num_ghosts + config.num_cells;
    for (int j = lo; j < hi; ++j)
    {
        const double y = coordinate(j, dx, config);
        for (int i = lo; i < hi; ++i)
        {
            if (is_excision_cell(i, j, dx, config))
                continue;
            const double x = coordinate(i, dx, config);
            const auto geometry = geometry_at(x, y, config);
            const auto &primitive = primitives[index(i, j, config)];
            double speed_sum = 0.0;
            for (int dir = 0; dir < 2; ++dir)
            {
                const auto speeds = GRHD::compute_characteristic_speeds(
                    primitive, eos, geometry.spatial_metric_LL,
                    geometry.spatial_metric_UU, dir, geometry.lapse,
                    geometry.shift_U);
                speed_sum += std::max(std::abs(speeds.lower),
                                      std::abs(speeds.upper));
            }
            max_speed_sum = std::max(max_speed_sum, speed_sum);
        }
    }
    return config.cfl * dx / std::max(max_speed_sum, 1.0e-14);
}

GRHD::Flux<double> interface_flux(
    const std::vector<GRHD::Primitive<double>> &primitives,
    const std::array<std::vector<GRHD::Primitive<double>>, 2> &slopes,
    const IdealGasEOS &eos, const SimulationConfig &config, double dx,
    int i_left, int j_left, int direction)
{
    int i_right = i_left;
    int j_right = j_left;
    if (direction == 0)
        ++i_right;
    else
        ++j_right;

    const int left = index(i_left, j_left, config);
    const int right = index(i_right, j_right, config);

    double x_face = coordinate(i_left, dx, config);
    double y_face = coordinate(j_left, dx, config);
    if (direction == 0)
        x_face = interface_coordinate(i_left, dx, config);
    else
        y_face = interface_coordinate(j_left, dx, config);

    const auto geometry = geometry_at(x_face, y_face, config);
    return GRHD::compute_muscl_hlle_flux(
        primitives[left], primitives[right], slopes[direction][left],
        slopes[direction][right], eos, geometry.spatial_metric_LL,
        geometry.spatial_metric_UU, direction, geometry.lapse,
        geometry.shift_U, atmosphere_options(config),
        config.use_reconstruction);
}

std::vector<GRHD::Conserved<double>>
compute_rhs(const std::vector<GRHD::Conserved<double>> &state,
            const IdealGasEOS &eos, const SimulationConfig &config, double dx,
            int &reset_count)
{
    std::vector<GRHD::Conserved<double>> bounded_state = state;
    repair_state(bounded_state, eos, config, dx, reset_count);
    RecoveryScratch scratch = recover_all(bounded_state, eos, config, dx);
    reset_count += scratch.resets;
    const auto slopes = compute_all_slopes(scratch.primitives, config);

    std::vector<GRHD::Conserved<double>> rhs(state.size());
    for (int idx = 0; idx < static_cast<int>(rhs.size()); ++idx)
        rhs[idx] = zero_conserved();

    const double inverse_dx = 1.0 / dx;
    const int lo = config.num_ghosts;
    const int hi = config.num_ghosts + config.num_cells;
    for (int j = lo; j < hi; ++j)
    {
        const double y = coordinate(j, dx, config);
        for (int i = lo; i < hi; ++i)
        {
            if (is_excision_cell(i, j, dx, config))
                continue;

            const double x = coordinate(i, dx, config);
            const auto flux_x_hi = interface_flux(
                scratch.primitives, slopes, eos, config, dx, i, j, 0);
            const auto flux_x_lo = interface_flux(
                scratch.primitives, slopes, eos, config, dx, i - 1, j, 0);
            const auto flux_y_hi = interface_flux(
                scratch.primitives, slopes, eos, config, dx, i, j, 1);
            const auto flux_y_lo = interface_flux(
                scratch.primitives, slopes, eos, config, dx, i, j - 1, 1);

            std::array<GRHD::Flux<double>, CH_SPACEDIM> flux_hi = {
                flux_x_hi, flux_y_hi, GRHD::Flux<double>()};
            std::array<GRHD::Flux<double>, CH_SPACEDIM> flux_lo = {
                flux_x_lo, flux_y_lo, GRHD::Flux<double>()};
            flux_hi[2].D = 0.0;
            flux_hi[2].tau = 0.0;
            FOR(momentum_dir) { flux_hi[2].S_L[momentum_dir] = 0.0; }
            flux_lo[2].D = 0.0;
            flux_lo[2].tau = 0.0;
            FOR(momentum_dir) { flux_lo[2].S_L[momentum_dir] = 0.0; }

            Tensor<1, double> inverse_dx_U;
            FOR(dir) { inverse_dx_U[dir] = 0.0; }
            inverse_dx_U[0] = inverse_dx;
            inverse_dx_U[1] = inverse_dx;
            rhs[index(i, j, config)] =
                GRHD::compute_flux_divergence(flux_hi, flux_lo,
                                              inverse_dx_U);

            const auto geometry = geometry_at(x, y, config);
            const auto derivatives = metric_derivatives_at(x, y, config);
            const auto source = GRHD::compute_static_metric_source_terms(
                scratch.primitives[index(i, j, config)],
                bounded_state[index(i, j, config)], eos, geometry,
                derivatives);
            rhs[index(i, j, config)].D += source.D;
            rhs[index(i, j, config)].tau += source.tau;
            FOR(dir)
            {
                rhs[index(i, j, config)].S_L[dir] += source.S_L[dir];
            }

            const double d_log_sqrt_det_dt =
                log_sqrt_det_time_derivative_at(x, y, config);
            const auto volume_time_source =
                GRHD::compute_metric_volume_time_source_terms(
                    bounded_state[index(i, j, config)],
                    d_log_sqrt_det_dt);
            rhs[index(i, j, config)].D += volume_time_source.D;
            rhs[index(i, j, config)].tau += volume_time_source.tau;
            FOR(dir)
            {
                rhs[index(i, j, config)].S_L[dir] +=
                    volume_time_source.S_L[dir];
            }
        }
    }
    return rhs;
}

std::vector<GRHD::Conserved<double>>
add_scaled_state(const std::vector<GRHD::Conserved<double>> &state,
                 const std::vector<GRHD::Conserved<double>> &rhs,
                 double scale)
{
    std::vector<GRHD::Conserved<double>> out = state;
    for (int idx = 0; idx < static_cast<int>(out.size()); ++idx)
    {
        out[idx].D += scale * rhs[idx].D;
        out[idx].tau += scale * rhs[idx].tau;
        FOR(dir) { out[idx].S_L[dir] += scale * rhs[idx].S_L[dir]; }
    }
    return out;
}

void advance(std::vector<GRHD::Conserved<double>> &state,
             const IdealGasEOS &eos, const SimulationConfig &config,
             double time, double dt, double dx, int &reset_count)
{
    const SimulationConfig initial_config = config_at_time(config, time);
    const SimulationConfig stage_config = config_at_time(config, time + dt);
    const auto rhs_initial =
        compute_rhs(state, eos, initial_config, dx, reset_count);
    auto stage_state = add_scaled_state(state, rhs_initial, dt);
    repair_state(stage_state, eos, stage_config, dx, reset_count);

    const auto rhs_stage =
        compute_rhs(stage_state, eos, stage_config, dx, reset_count);
    for (int idx = 0; idx < static_cast<int>(state.size()); ++idx)
    {
        state[idx].D =
            0.5 * (state[idx].D + stage_state[idx].D + dt * rhs_stage[idx].D);
        state[idx].tau = 0.5 * (state[idx].tau + stage_state[idx].tau +
                                dt * rhs_stage[idx].tau);
        FOR(dir)
        {
            state[idx].S_L[dir] =
                0.5 * (state[idx].S_L[dir] + stage_state[idx].S_L[dir] +
                       dt * rhs_stage[idx].S_L[dir]);
        }
    }
    repair_state(state, eos, stage_config, dx, reset_count);
}

Diagnostics compute_diagnostics(
    const std::vector<GRHD::Conserved<double>> &state,
    const std::vector<GRHD::Primitive<double>> &primitives,
    const SimulationConfig &config, double dx, int reset_count)
{
    Diagnostics diagnostics;
    diagnostics.atmosphere_resets = reset_count;
    const double cell_area = dx * dx;
    const int lo = config.num_ghosts;
    const int hi = config.num_ghosts + config.num_cells;
    for (int j = lo; j < hi; ++j)
    {
        const double y = coordinate(j, dx, config);
        for (int i = lo; i < hi; ++i)
        {
            if (is_excision_cell(i, j, dx, config))
                continue;
            const double x = coordinate(i, dx, config);
            const int idx = index(i, j, config);
            const auto geometry = geometry_at(x, y, config);
            const double sqrt_det =
                std::exp(GRHD::spatial_metric_log_sqrt_det(geometry));
            const double volume = sqrt_det * cell_area;
            diagnostics.mass += state[idx].D * volume;
            diagnostics.tau += state[idx].tau * volume;
            if (primitives[idx].rho > 10.0 * config.rho_floor)
                diagnostics.disk_mass += state[idx].D * volume;
            diagnostics.max_rho =
                std::max(diagnostics.max_rho, primitives[idx].rho);
            diagnostics.max_pressure =
                std::max(diagnostics.max_pressure, primitives[idx].pressure);
            const double velocity_squared = GRHD::compute_velocity_squared(
                primitives[idx].velocity_U, geometry.spatial_metric_LL);
            diagnostics.max_velocity =
                std::max(diagnostics.max_velocity,
                         std::sqrt(std::max(0.0, velocity_squared)));
            diagnostics.max_lorentz_factor = std::max(
                diagnostics.max_lorentz_factor,
                GRHD::compute_lorentz_factor(primitives[idx].velocity_U,
                                             geometry.spatial_metric_LL));
        }
    }
    return diagnostics;
}

std::string default_output_dir()
{
    struct stat info;
    if (stat("Tests/GRHDTest", &info) == 0)
        return "Tests/GRHDTest/output";
    return "output";
}

void ensure_directory(const std::string &dir) { mkdir(dir.c_str(), 0755); }

std::string frame_path(const std::string &frame_dir, int frame)
{
    std::ostringstream path;
    path << frame_dir << "/frame_" << std::setw(4) << std::setfill('0')
         << frame << ".ppm";
    return path.str();
}

std::array<unsigned char, 3> interpolate_color(
    const std::array<unsigned char, 3> &a,
    const std::array<unsigned char, 3> &b, double fraction)
{
    std::array<unsigned char, 3> out;
    for (int c = 0; c < 3; ++c)
    {
        out[c] = static_cast<unsigned char>(
            std::round((1.0 - fraction) * a[c] + fraction * b[c]));
    }
    return out;
}

std::array<unsigned char, 3> density_color(double rho,
                                           const SimulationConfig &config)
{
    static const std::array<std::array<unsigned char, 3>, 5> palette = {{
        {{8, 29, 88}},
        {{37, 52, 148}},
        {{34, 139, 141}},
        {{110, 206, 88}},
        {{248, 230, 33}},
    }};
    const double low = std::log10(config.rho_floor);
    const double high = std::log10(config.rho_peak);
    double scaled = (std::log10(std::max(rho, config.rho_floor)) - low) /
                    (high - low);
    scaled = std::min(std::max(scaled, 0.0), 1.0);
    const double position = scaled * (palette.size() - 1);
    const int lower = std::min(static_cast<int>(std::floor(position)),
                               static_cast<int>(palette.size()) - 1);
    const int upper = std::min(lower + 1,
                               static_cast<int>(palette.size()) - 1);
    return interpolate_color(palette[lower], palette[upper],
                             position - lower);
}

void write_frame_ppm(const std::string &path,
                     const std::vector<GRHD::Primitive<double>> &primitives,
                     const SimulationConfig &config, double dx)
{
    const int width = config.num_cells * config.image_scale;
    const int height = config.num_cells * config.image_scale;
    std::ofstream out(path.c_str(), std::ios::binary);
    if (!out)
        throw std::runtime_error("could not open " + path);
    out << "P6\n" << width << ' ' << height << "\n255\n";

    const int lo = config.num_ghosts;
    for (int py = 0; py < height; ++py)
    {
        const int j = lo + config.num_cells - 1 - py / config.image_scale;
        const double y = coordinate(j, dx, config);
        for (int px = 0; px < width; ++px)
        {
            const int i = lo + px / config.image_scale;
            const double x = coordinate(i, dx, config);
            std::array<unsigned char, 3> rgb = density_color(
                primitives[index(i, j, config)].rho, config);
            if (has_excision(config) &&
                std::sqrt(x * x + y * y) <= config.excision_radius)
                rgb = {{0, 0, 0}};
            out.write(reinterpret_cast<const char *>(rgb.data()), 3);
        }
    }
}

void write_slice_csv(const std::string &path,
                     const std::vector<GRHD::Conserved<double>> &state,
                     const std::vector<GRHD::Primitive<double>> &primitives,
                     const SimulationConfig &config, double dx)
{
    std::ofstream out(path.c_str());
    if (!out)
        throw std::runtime_error("could not open " + path);
    out << "x,y,radius,rho,pressure,vx,vy,D,Sx,Sy,tau,lapse\n";
    out << std::setprecision(17);
    const int lo = config.num_ghosts;
    const int hi = config.num_ghosts + config.num_cells;
    for (int j = lo; j < hi; ++j)
    {
        const double y = coordinate(j, dx, config);
        for (int i = lo; i < hi; ++i)
        {
            const double x = coordinate(i, dx, config);
            const int idx = index(i, j, config);
            const auto geometry = geometry_at(x, y, config);
            const double radius = std::sqrt(x * x + y * y);
            out << x << ',' << y << ','
                << (has_excision(config)
                        ? std::max(radius, config.excision_radius)
                        : radius)
                << ',' << primitives[idx].rho << ','
                << primitives[idx].pressure << ','
                << primitives[idx].velocity_U[0] << ','
                << primitives[idx].velocity_U[1] << ',' << state[idx].D
                << ',' << state[idx].S_L[0] << ',' << state[idx].S_L[1]
                << ',' << state[idx].tau << ',' << geometry.lapse << '\n';
        }
    }
}

void append_diagnostics(std::ofstream &out, int step, int frame, double time,
                        const Diagnostics &diagnostics)
{
    out << step << ',' << frame << ',' << std::setprecision(17) << time
        << ',' << diagnostics.mass << ',' << diagnostics.disk_mass << ','
        << diagnostics.tau << ',' << diagnostics.max_rho << ','
        << diagnostics.max_pressure << ',' << diagnostics.max_velocity << ','
        << diagnostics.max_lorentz_factor << ','
        << diagnostics.atmosphere_resets << '\n';
}

void write_summary(const std::string &path, const SimulationConfig &config,
                   const TorusParameters &torus,
                   const Diagnostics &initial_diagnostics,
                   const Diagnostics &final_diagnostics, double final_time,
                   int steps, int frames)
{
    std::ofstream out(path.c_str());
    if (!out)
        throw std::runtime_error("could not open " + path);
    out << std::setprecision(17);
    out << "driver GRHDBinaryPNDisk\n";
    out << "method fixed_grdzhadzha_circular_binary_pn_moving_phase_muscl_hlle_ssprk2\n";
    out << "num_cells " << config.num_cells << '\n';
    out << "domain [" << config.x_min << "," << config.x_max << "]\n";
    out << "final_time " << final_time << '\n';
    out << "steps " << steps << '\n';
    out << "frames " << frames << '\n';
    out << "mass_1 " << config.mass_1 << '\n';
    out << "mass_2 " << config.mass_2 << '\n';
    out << "total_mass " << total_mass(config) << '\n';
    out << "separation " << config.separation << '\n';
    out << "binary_phase " << config.binary_phase << '\n';
    out << "binary_time " << config.binary_time << '\n';
    out << "final_binary_time "
        << (config.evolve_binary_time ? config.binary_time + final_time
                                      : config.binary_time)
        << '\n';
    out << "binary_angular_velocity " << config.binary_angular_velocity
        << '\n';
    out << "evolve_binary_time " << config.evolve_binary_time << '\n';
    out << "use_time_volume_source " << config.use_time_volume_source
        << '\n';
    out << "metric_time_derivative_step "
        << config.metric_time_derivative_step << '\n';
    out << "softening_radius " << config.softening_radius << '\n';
    out << "excision_radius " << config.excision_radius << '\n';
    out << "torus_inner_radius " << config.torus_inner_radius
        << '\n';
    out << "torus_pressure_max_radius "
        << config.torus_pressure_max_radius << '\n';
    out << "torus_specific_angular_momentum " << torus.ell << '\n';
    out << "torus_surface_potential " << torus.surface_potential << '\n';
    out << "torus_max_enthalpy " << torus.max_enthalpy << '\n';
    out << "torus_polytropic_constant " << torus.polytropic_constant << '\n';
    out << "initial_mass " << initial_diagnostics.mass << '\n';
    out << "final_mass " << final_diagnostics.mass << '\n';
    out << "relative_mass_change "
        << (final_diagnostics.mass - initial_diagnostics.mass) /
               initial_diagnostics.mass
        << '\n';
    out << "initial_disk_mass " << initial_diagnostics.disk_mass << '\n';
    out << "final_disk_mass " << final_diagnostics.disk_mass << '\n';
    out << "relative_disk_mass_change "
        << (final_diagnostics.disk_mass - initial_diagnostics.disk_mass) /
               initial_diagnostics.disk_mass
        << '\n';
    out << "max_rho " << final_diagnostics.max_rho << '\n';
    out << "max_pressure " << final_diagnostics.max_pressure << '\n';
    out << "max_velocity " << final_diagnostics.max_velocity << '\n';
    out << "max_lorentz_factor " << final_diagnostics.max_lorentz_factor
        << '\n';
    out << "atmosphere_resets " << final_diagnostics.atmosphere_resets
        << '\n';
}
} // namespace

int main()
{
    try
    {
        const SimulationConfig config;
        const IdealGasEOS eos(5.0 / 3.0);
        const int n_total = total_cells_1d(config);
        const double dx = (config.x_max - config.x_min) / config.num_cells;
        const TorusParameters torus = make_torus_parameters(config, eos, dx);

        std::vector<GRHD::Conserved<double>> state(n_total * n_total);
        for (int j = 0; j < n_total; ++j)
        {
            const double y = coordinate(j, dx, config);
            for (int i = 0; i < n_total; ++i)
            {
                const double x = coordinate(i, dx, config);
                const auto geometry = geometry_at(x, y, config);
                const auto primitive =
                    torus_primitive(x, y, torus, eos, config);
                state[index(i, j, config)] = GRHD::compute_conserved(
                    primitive, eos, geometry.spatial_metric_LL);
            }
        }

        int reset_count = 0;
        repair_state(state, eos, config, dx, reset_count);
        RecoveryScratch scratch = recover_all(state, eos, config, dx);
        reset_count += scratch.resets;
        Diagnostics initial_diagnostics = compute_diagnostics(
            state, scratch.primitives, config, dx, reset_count);

        const std::string out_dir = default_output_dir();
        ensure_directory(out_dir);
        const std::string frame_dir = out_dir + "/binary_pn_disk_frames";
        ensure_directory(frame_dir);
        const std::string base = out_dir + "/binary_pn_disk_static";
        std::ofstream diagnostics_csv((base + "_diagnostics.csv").c_str());
        if (!diagnostics_csv)
            throw std::runtime_error("could not open diagnostics CSV");
        diagnostics_csv
            << "step,frame,time,mass,disk_mass,tau,max_rho,max_pressure,"
               "max_velocity,max_lorentz_factor,atmosphere_resets\n";

        double time = 0.0;
        double next_frame_time = 0.0;
        int steps = 0;
        int frames = 0;
        append_diagnostics(diagnostics_csv, steps, frames, time,
                           initial_diagnostics);
        write_frame_ppm(frame_path(frame_dir, frames), scratch.primitives,
                        config, dx);
        ++frames;
        next_frame_time += config.frame_interval;

        while (time < config.final_time - 1.0e-14)
        {
            const SimulationConfig time_config = config_at_time(config, time);
            scratch = recover_all(state, eos, time_config, dx);
            reset_count += scratch.resets;
            double dt = compute_time_step(scratch.primitives, eos,
                                          time_config, dx);
            if (time + dt > config.final_time)
                dt = config.final_time - time;
            if (time + dt > next_frame_time)
                dt = next_frame_time - time;
            if (dt <= 0.0)
                dt = std::min(config.final_time - time,
                              config.cfl * dx);

            advance(state, eos, config, time, dt, dx, reset_count);
            time += dt;
            ++steps;

            if (time >= next_frame_time - 1.0e-12 ||
                time >= config.final_time - 1.0e-14)
            {
                const SimulationConfig output_config =
                    config_at_time(config, time);
                scratch = recover_all(state, eos, output_config, dx);
                reset_count += scratch.resets;
                const Diagnostics diagnostics = compute_diagnostics(
                    state, scratch.primitives, output_config, dx, reset_count);
                append_diagnostics(diagnostics_csv, steps, frames, time,
                                   diagnostics);
                write_frame_ppm(frame_path(frame_dir, frames),
                                scratch.primitives, config, dx);
                ++frames;
                next_frame_time += config.frame_interval;
            }
        }

        const SimulationConfig final_config = config_at_time(config, time);
        scratch = recover_all(state, eos, final_config, dx);
        reset_count += scratch.resets;
        const Diagnostics final_diagnostics = compute_diagnostics(
            state, scratch.primitives, final_config, dx, reset_count);
        write_slice_csv(base + "_final_slice.csv", state, scratch.primitives,
                        final_config, dx);
        write_summary(base + "_summary.txt", config, torus,
                      initial_diagnostics, final_diagnostics, time, steps,
                      frames);

        if (!std::isfinite(final_diagnostics.max_rho) ||
            final_diagnostics.max_rho <= config.rho_floor)
            throw std::runtime_error("disk density became invalid");
        if (final_diagnostics.disk_mass <
            0.10 * initial_diagnostics.disk_mass)
            throw std::runtime_error("disk lost too much mass");
        if (final_diagnostics.max_velocity >
            std::sqrt(config.max_velocity_squared) * 1.001)
            throw std::runtime_error("velocity cap was violated");

        std::cout << "GRHD fixed binary PN disk passed..." << std::endl;
        std::cout << "  cells: " << config.num_cells << "^2" << std::endl;
        std::cout << "  steps: " << steps << std::endl;
        std::cout << "  frames: " << frames << std::endl;
        std::cout << "  final time: " << time << std::endl;
        std::cout << "  initial disk mass: "
                  << initial_diagnostics.disk_mass << std::endl;
        std::cout << "  final disk mass: " << final_diagnostics.disk_mass
                  << std::endl;
        std::cout << "  max rho: " << final_diagnostics.max_rho
                  << std::endl;
        std::cout << "  max velocity: " << final_diagnostics.max_velocity
                  << std::endl;
        std::cout << "  atmosphere resets: "
                  << final_diagnostics.atmosphere_resets << std::endl;
        std::cout << "  wrote " << base << "_diagnostics.csv" << std::endl;
        std::cout << "  wrote " << base << "_final_slice.csv" << std::endl;
        std::cout << "  wrote " << base << "_summary.txt" << std::endl;
        std::cout << "  wrote frames in " << frame_dir << std::endl;
    }
    catch (const std::exception &error)
    {
        std::cout << "GRHD fixed binary PN disk failed: "
                  << error.what() << std::endl;
        return 1;
    }
    return 0;
}
