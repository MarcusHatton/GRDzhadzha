/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#include "CircularBinaryPN.hpp"
#include "Coordinates.hpp"
#include "FixedBGFArrayBoxFiniteVolume.hpp"
#include "IdealGasEOS.hpp"
#include "TensorAlgebra.hpp"
#include "UserVariables.hpp"
#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdlib>
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

#ifdef CH_MPI
#include "mpi.h"
#endif

namespace
{
struct SimulationConfig
{
    int num_cells = 64;
    int num_z_cells = 13;
    int num_ghosts = 2;
    double domain_length = 32.0;
    double final_time = 10.0;
    double output_interval = 0.1;
    double cfl = 0.14;
    double max_dt = 0.02;

    double mass_1 = 0.5;
    double mass_2 = 0.5;
    double separation = 12.0;
    double binary_phase = 0.0;
    double binary_time = 0.0;
    double binary_angular_velocity = 0.0;
    double softening_radius = 0.75;
    double metric_time_derivative_step = 1.0e-3;

    double mini_disk_inner_radius = 1.6;
    double mini_disk_peak_radius = 2.4;
    double mini_disk_outer_radius = 3.25;
    double mini_disk_radial_width = 0.45;
    double mini_disk_vertical_width = 0.55;
    double rho_peak = 4.0e-5;
    double pressure_over_density = 8.0e-4;
    double orbital_velocity_factor = 0.64;
    double matched_atmosphere_width = 0.0;
    double matched_atmosphere_density_factor = 1.0;

    double rho_floor = 1.0e-10;
    double pressure_floor = 8.0e-14;
    double max_velocity_squared = 0.72;
    double limiter_theta = 1.3;
    bool use_reconstruction = true;
    bool use_static_metric_sources = true;
    bool use_metric_volume_time_sources = true;
    bool write_density_frames = true;
    int frame_pixel_scale = 4;
};

struct BinaryPNFactory
{
    CircularBinaryPN::params_t params;
    double dx = 1.0;

    CircularBinaryPN operator()(double time) const
    {
        auto time_params = params;
        time_params.time += time;
        return CircularBinaryPN(time_params, dx);
    }
};

struct BinaryState
{
    std::array<std::array<double, CH_SPACEDIM>, 2> position;
    std::array<std::array<double, CH_SPACEDIM>, 2> velocity;
    std::array<double, 2> mass;
    double omega = 0.0;
};

struct DiskCandidate
{
    bool active = false;
    int index = 0;
    double rho = 0.0;
    double radius = 0.0;
    std::array<double, CH_SPACEDIM> relative_position;
};

struct MiniDiskEquilibrium
{
    std::array<double, 2> ell;
    std::array<double, 2> peak_potential;
    std::array<double, 2> surface_potential;
    std::array<double, 2> pressure_over_density;
    double density_exponent = 1.5;
};

struct Diagnostics
{
    double mass = 0.0;
    double disk_mass = 0.0;
    double disk_1_mass = 0.0;
    double disk_2_mass = 0.0;
    double tau = 0.0;
    double max_rho = 0.0;
    double max_pressure = 0.0;
    double max_velocity = 0.0;
    double max_lorentz_factor = 1.0;
    long long failed_recoveries = 0;
    long long floored_primitives = 0;
    long long conserved_resets = 0;
};

int mpi_rank()
{
#ifdef CH_MPI
    int rank = 0;
    MPI_Comm_rank(Chombo_MPI::comm, &rank);
    return rank;
#else
    return 0;
#endif
}

int mpi_size()
{
#ifdef CH_MPI
    int size = 1;
    MPI_Comm_size(Chombo_MPI::comm, &size);
    return size;
#else
    return 1;
#endif
}

bool is_io_rank() { return mpi_rank() == 0; }

void synchronize_ranks()
{
#ifdef CH_MPI
    MPI_Barrier(Chombo_MPI::comm);
#endif
}

void ensure_directory(const std::string &dir) { mkdir(dir.c_str(), 0755); }

double parse_positive_double(const char *text, const std::string &name)
{
    errno = 0;
    char *end = nullptr;
    const double value = std::strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !std::isfinite(value) ||
        value <= 0.0)
    {
        throw std::runtime_error(name + " must be a positive finite number");
    }
    return value;
}

SimulationConfig make_config_from_args(int argc, char *argv[])
{
    SimulationConfig config;
    if (argc > 3)
        throw std::runtime_error(
            "usage: Main_GRHDLevelDataBinaryPNMiniDisks [final_time [orbital_velocity_factor]]");
    if (argc >= 2)
        config.final_time = parse_positive_double(argv[1], "final_time");
    if (argc == 3)
        config.orbital_velocity_factor =
            parse_positive_double(argv[2], "orbital_velocity_factor");
    return config;
}

std::string default_output_dir()
{
    struct stat info;
    if (stat("Examples/GRHDLevelDataBinaryPNMiniDisks", &info) == 0)
        return "Examples/GRHDLevelDataBinaryPNMiniDisks/output";
    return "output";
}

Box make_domain_box(const SimulationConfig &config)
{
    return Box(IntVect(D_DECL(0, 0, 0)),
               IntVect(D_DECL(config.num_cells - 1,
                              config.num_cells - 1,
                              config.num_z_cells - 1)));
}

DisjointBoxLayout make_x_split_layout(const Box &domain_box)
{
    const int x_mid = (domain_box.smallEnd(0) + domain_box.bigEnd(0)) / 2;

    IntVect left_lo = domain_box.smallEnd();
    IntVect left_hi = domain_box.bigEnd();
    left_hi[0] = x_mid;
    IntVect right_lo = domain_box.smallEnd();
    right_lo[0] = x_mid + 1;
    IntVect right_hi = domain_box.bigEnd();

    Vector<Box> boxes(2);
    boxes[0] = Box(left_lo, left_hi);
    boxes[1] = Box(right_lo, right_hi);

    const int num_procs = mpi_size();
    Vector<int> proc_map(2);
    proc_map[0] = 0;
    proc_map[1] = (num_procs > 1) ? 1 : 0;

    DisjointBoxLayout grids(boxes, proc_map, ProblemDomain(domain_box));
    grids.close();
    return grids;
}

std::array<double, CH_SPACEDIM> make_center(const SimulationConfig &config,
                                            double dx)
{
    return {0.5 * config.num_cells * dx, 0.5 * config.num_cells * dx,
            0.5 * config.num_z_cells * dx};
}

CircularBinaryPN::params_t make_background_params(
    const SimulationConfig &config)
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

BinaryPNFactory make_background_factory(const SimulationConfig &config,
                                        double dx)
{
    BinaryPNFactory factory;
    factory.params = make_background_params(config);
    factory.dx = dx;
    return factory;
}

GRHD::AtmosphereOptions make_atmosphere_options(
    const SimulationConfig &config)
{
    GRHD::AtmosphereOptions atmosphere;
    atmosphere.rho_floor = config.rho_floor;
    atmosphere.pressure_floor = config.pressure_floor;
    atmosphere.max_velocity_squared = config.max_velocity_squared;
    return atmosphere;
}

GRHD::RecoveryOptions make_recovery_options(const SimulationConfig &config)
{
    GRHD::RecoveryOptions options;
    options.max_iterations = 80;
    options.pressure_floor = config.pressure_floor;
    options.max_velocity_squared = config.max_velocity_squared;
    return options;
}

double total_mass(const SimulationConfig &config)
{
    return config.mass_1 + config.mass_2;
}

double binary_angular_velocity(const SimulationConfig &config)
{
    if (config.binary_angular_velocity != 0.0)
        return config.binary_angular_velocity;
    return std::sqrt(total_mass(config) /
                     (config.separation * config.separation *
                      config.separation));
}

BinaryState make_binary_state(const SimulationConfig &config, double time)
{
    BinaryState state;
    state.mass[0] = config.mass_1;
    state.mass[1] = config.mass_2;
    state.omega = binary_angular_velocity(config);

    const double phase =
        config.binary_phase + state.omega * (config.binary_time + time);
    const double cos_phase = std::cos(phase);
    const double sin_phase = std::sin(phase);
    const double radius_1 = config.mass_2 / total_mass(config) *
                            config.separation;
    const double radius_2 = config.mass_1 / total_mass(config) *
                            config.separation;

    for (int body = 0; body < 2; ++body)
    {
        FOR(dir)
        {
            state.position[body][dir] = 0.0;
            state.velocity[body][dir] = 0.0;
        }
    }

    state.position[0][0] = radius_1 * cos_phase;
    state.position[0][1] = radius_1 * sin_phase;
    state.position[1][0] = -radius_2 * cos_phase;
    state.position[1][1] = -radius_2 * sin_phase;

    state.velocity[0][0] = -state.omega * radius_1 * sin_phase;
    state.velocity[0][1] = state.omega * radius_1 * cos_phase;
    state.velocity[1][0] = state.omega * radius_2 * sin_phase;
    state.velocity[1][1] = -state.omega * radius_2 * cos_phase;
    return state;
}

double binary_roche_potential(
    const std::array<double, CH_SPACEDIM> &position,
    const BinaryState &binary, const SimulationConfig &config)
{
    double potential =
        -0.5 * binary.omega * binary.omega *
        (position[0] * position[0] + position[1] * position[1]);
    for (int body = 0; body < 2; ++body)
    {
        double radius_squared =
            config.softening_radius * config.softening_radius;
        FOR(dir)
        {
            const double displacement =
                position[dir] - binary.position[body][dir];
            radius_squared += displacement * displacement;
        }
        potential -= binary.mass[body] / std::sqrt(radius_squared);
    }
    return potential;
}

std::array<double, CH_SPACEDIM> body_radial_unit(
    const BinaryState &binary, int body)
{
    std::array<double, CH_SPACEDIM> unit = {{1.0, 0.0, 0.0}};
    const double radius = std::sqrt(
        binary.position[body][0] * binary.position[body][0] +
        binary.position[body][1] * binary.position[body][1]);
    if (radius > 1.0e-14)
    {
        unit[0] = binary.position[body][0] / radius;
        unit[1] = binary.position[body][1] / radius;
    }
    return unit;
}

std::array<double, CH_SPACEDIM> body_tangent_unit(
    const std::array<double, CH_SPACEDIM> &radial_unit)
{
    return {{-radial_unit[1], radial_unit[0], 0.0}};
}

double mini_disk_specific_angular_momentum(
    const BinaryState &binary, const SimulationConfig &config, int body)
{
    const double radius = config.mini_disk_peak_radius;
    const double radius_squared = radius * radius;
    const double softened_radius_squared =
        radius_squared + config.softening_radius * config.softening_radius;
    const double ell_squared =
        binary.mass[body] * radius_squared * radius_squared /
        std::pow(softened_radius_squared, 1.5);
    return config.orbital_velocity_factor *
           std::sqrt(std::max(0.0, ell_squared));
}

double mini_disk_effective_potential(
    const std::array<double, CH_SPACEDIM> &position,
    const BinaryState &binary, const SimulationConfig &config, int body,
    double ell)
{
    const double dx = position[0] - binary.position[body][0];
    const double dy = position[1] - binary.position[body][1];
    const double radius_squared = dx * dx + dy * dy;
    if (radius_squared <= 1.0e-14)
        return std::numeric_limits<double>::infinity();
    return binary_roche_potential(position, binary, config) +
           0.5 * ell * ell / radius_squared;
}

std::array<double, CH_SPACEDIM> body_ring_position(
    const BinaryState &binary, int body, double radius, double angle)
{
    const auto radial = body_radial_unit(binary, body);
    const auto tangent = body_tangent_unit(radial);
    const double cos_angle = std::cos(angle);
    const double sin_angle = std::sin(angle);
    std::array<double, CH_SPACEDIM> position = binary.position[body];
    FOR(dir)
    {
        position[dir] +=
            radius * (cos_angle * radial[dir] + sin_angle * tangent[dir]);
    }
    return position;
}

double sample_mini_disk_ring_min_potential(
    const BinaryState &binary, const SimulationConfig &config, int body,
    double radius, double ell)
{
    constexpr int num_samples = 96;
    const double two_pi = 8.0 * std::atan(1.0);
    double min_potential = std::numeric_limits<double>::infinity();
    for (int sample = 0; sample < num_samples; ++sample)
    {
        const double angle =
            two_pi * static_cast<double>(sample) /
            static_cast<double>(num_samples);
        const auto position = body_ring_position(binary, body, radius, angle);
        min_potential = std::min(
            min_potential,
            mini_disk_effective_potential(position, binary, config, body,
                                          ell));
    }
    return min_potential;
}

template <class eos_t>
MiniDiskEquilibrium make_mini_disk_equilibrium(
    const BinaryState &binary, const SimulationConfig &config,
    const eos_t &eos)
{
    MiniDiskEquilibrium equilibrium;
    const double gamma = eos.adiabatic_index();
    equilibrium.density_exponent = 1.0 / (gamma - 1.0);

    for (int body = 0; body < 2; ++body)
    {
        equilibrium.ell[body] =
            mini_disk_specific_angular_momentum(binary, config, body);
        equilibrium.peak_potential[body] =
            sample_mini_disk_ring_min_potential(
                binary, config, body, config.mini_disk_peak_radius,
                equilibrium.ell[body]);
        const double inner_surface = sample_mini_disk_ring_min_potential(
            binary, config, body, config.mini_disk_inner_radius,
            equilibrium.ell[body]);
        const double outer_surface = sample_mini_disk_ring_min_potential(
            binary, config, body, config.mini_disk_outer_radius,
            equilibrium.ell[body]);
        equilibrium.surface_potential[body] =
            std::min(inner_surface, outer_surface);

        if (!std::isfinite(equilibrium.peak_potential[body]) ||
            !std::isfinite(equilibrium.surface_potential[body]) ||
            equilibrium.surface_potential[body] <=
                equilibrium.peak_potential[body])
        {
            equilibrium.surface_potential[body] =
                equilibrium.peak_potential[body] + 1.0e-8;
        }

        equilibrium.pressure_over_density[body] =
            std::max(config.pressure_floor / config.rho_peak,
                     config.pressure_over_density);
    }
    return equilibrium;
}

DiskCandidate mini_disk_candidate(
    const std::array<double, CH_SPACEDIM> &position,
    const BinaryState &binary, const SimulationConfig &config,
    const MiniDiskEquilibrium &equilibrium, int body)
{
    DiskCandidate candidate;
    candidate.index = body;
    FOR(dir)
    {
        candidate.relative_position[dir] =
            position[dir] - binary.position[body][dir];
    }
    candidate.radius = std::sqrt(
        candidate.relative_position[0] * candidate.relative_position[0] +
        candidate.relative_position[1] * candidate.relative_position[1]);
    if (candidate.radius < config.mini_disk_inner_radius ||
        candidate.radius > config.mini_disk_outer_radius)
        return candidate;

    const double potential = mini_disk_effective_potential(
        position, binary, config, body, equilibrium.ell[body]);
    const double potential_depth =
        equilibrium.surface_potential[body] -
        equilibrium.peak_potential[body];
    if (!std::isfinite(potential) || potential_depth <= 0.0 ||
        potential >= equilibrium.surface_potential[body])
        return candidate;

    const double normalized_depth = std::min(
        1.0, std::max(0.0,
                      (equilibrium.surface_potential[body] - potential) /
                          potential_depth));
    candidate.rho =
        config.rho_peak *
        std::pow(normalized_depth, equilibrium.density_exponent);
    candidate.active = candidate.rho > 20.0 * config.rho_floor;
    return candidate;
}

DiskCandidate select_mini_disk(
    const std::array<double, CH_SPACEDIM> &position,
    const BinaryState &binary, const SimulationConfig &config,
    const MiniDiskEquilibrium &equilibrium)
{
    const DiskCandidate first =
        mini_disk_candidate(position, binary, config, equilibrium, 0);
    const DiskCandidate second =
        mini_disk_candidate(position, binary, config, equilibrium, 1);
    if (first.rho >= second.rho)
        return first;
    return second;
}

DiskCandidate matched_atmosphere_candidate(
    const std::array<double, CH_SPACEDIM> &position,
    const BinaryState &binary, const SimulationConfig &config, int body)
{
    DiskCandidate candidate;
    candidate.index = body;
    if (config.matched_atmosphere_width <= 0.0)
        return candidate;
    FOR(dir)
    {
        candidate.relative_position[dir] =
            position[dir] - binary.position[body][dir];
    }
    candidate.radius = std::sqrt(
        candidate.relative_position[0] * candidate.relative_position[0] +
        candidate.relative_position[1] * candidate.relative_position[1]);

    const double inner_radius = config.mini_disk_inner_radius;
    const double outer_radius =
        config.mini_disk_outer_radius + config.matched_atmosphere_width;
    candidate.active = candidate.radius >= inner_radius &&
                       candidate.radius <= outer_radius;
    return candidate;
}

DiskCandidate select_matched_atmosphere(
    const std::array<double, CH_SPACEDIM> &position,
    const BinaryState &binary, const SimulationConfig &config)
{
    const DiskCandidate first =
        matched_atmosphere_candidate(position, binary, config, 0);
    const DiskCandidate second =
        matched_atmosphere_candidate(position, binary, config, 1);
    if (!second.active ||
        (first.active && first.radius <= second.radius))
        return first;
    return second;
}

Tensor<1, double> eulerian_velocity_for_mini_disk(
    const DiskCandidate &disk, const BinaryState &binary,
    const SimulationConfig &config, const MiniDiskEquilibrium &equilibrium,
    const GRHD::CellGeometry &geometry)
{
    Tensor<1, double> coordinate_velocity;
    FOR(dir) { coordinate_velocity[dir] = binary.velocity[disk.index][dir]; }

    const double radius_squared =
        std::max(disk.radius * disk.radius, 1.0e-14);
    const double local_omega =
        equilibrium.ell[disk.index] / radius_squared;
    coordinate_velocity[0] += -local_omega * disk.relative_position[1];
    coordinate_velocity[1] += local_omega * disk.relative_position[0];

    Tensor<1, double> velocity_U;
    FOR(dir)
    {
        velocity_U[dir] =
            (coordinate_velocity[dir] + geometry.shift_U[dir]) /
            geometry.lapse;
    }

    const double velocity_squared = GRHD::compute_velocity_squared(
        velocity_U, geometry.spatial_metric_LL);
    const double target_velocity_squared =
        0.96 * config.max_velocity_squared;
    if (velocity_squared > target_velocity_squared && velocity_squared > 0.0)
    {
        const double scale = std::sqrt(target_velocity_squared / velocity_squared);
        FOR(dir) { velocity_U[dir] *= scale; }
    }
    return velocity_U;
}

template <class eos_t>
GRHD::Primitive<double> mini_disk_primitive(
    const std::array<double, CH_SPACEDIM> &position,
    const BinaryState &binary, const SimulationConfig &config,
    const eos_t &eos, const GRHD::CellGeometry &geometry,
    const GRHD::AtmosphereOptions &atmosphere,
    const MiniDiskEquilibrium &equilibrium)
{
    GRHD::Primitive<double> primitive =
        GRHD::make_atmosphere_primitive<double>(eos, atmosphere);
    const DiskCandidate disk =
        select_mini_disk(position, binary, config, equilibrium);
    if (!disk.active)
    {
        const DiskCandidate atmosphere_disk =
            select_matched_atmosphere(position, binary, config);
        if (!atmosphere_disk.active)
            return primitive;

        primitive.rho = std::max(
            atmosphere.rho_floor,
            config.matched_atmosphere_density_factor * atmosphere.rho_floor);
        primitive.pressure = std::max(
            atmosphere.pressure_floor,
            config.pressure_over_density * primitive.rho);
        primitive.eps = primitive.pressure /
                        ((eos.adiabatic_index() - 1.0) * primitive.rho);
        primitive.velocity_U = eulerian_velocity_for_mini_disk(
            atmosphere_disk, binary, config, equilibrium, geometry);
        GRHD::enforce_primitive_floors(
            primitive, eos, geometry.spatial_metric_LL, atmosphere);
        return primitive;
    }

    primitive.rho = disk.rho;
    const double density_fraction =
        std::max(0.0, primitive.rho / config.rho_peak);
    primitive.pressure = std::max(
        config.pressure_floor,
        equilibrium.pressure_over_density[disk.index] * config.rho_peak *
            std::pow(density_fraction, eos.adiabatic_index()));
    primitive.eps =
        primitive.pressure / ((eos.adiabatic_index() - 1.0) * primitive.rho);
    primitive.velocity_U =
        eulerian_velocity_for_mini_disk(disk, binary, config, equilibrium,
                                       geometry);

    GRHD::enforce_primitive_floors(primitive, eos,
                                   geometry.spatial_metric_LL, atmosphere);
    return primitive;
}

template <class eos_t>
void fill_mini_disk_level_data(
    LevelData<FArrayBox> &state, const BinaryPNFactory &background_factory,
    const eos_t &eos, double time, double dx,
    const std::array<double, CH_SPACEDIM> &center,
    const SimulationConfig &config,
    const GRHD::AtmosphereOptions &atmosphere)
{
    const auto background = background_factory(time);
    const BinaryState binary = make_binary_state(config, time);
    const MiniDiskEquilibrium equilibrium =
        make_mini_disk_equilibrium(binary, config, eos);
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
            const std::array<double, CH_SPACEDIM> position = {
                coords.x, coords.y, coords.z};
            const auto geometry = GRHD::load_fixed_background_geometry(
                background, iv, dx, center);
            const auto primitive = mini_disk_primitive(
                position, binary, config, eos, geometry, atmosphere,
                equilibrium);
            GRHD::store_primitive(state[dit], iv, primitive);
            GRHD::store_conserved(
                state[dit], iv,
                GRHD::compute_conserved(primitive, eos,
                                        geometry.spatial_metric_LL));
        }
    }
}

Diagnostics compute_diagnostics(
    const LevelData<FArrayBox> &state,
    const BinaryPNFactory &background_factory, double time, double dx,
    const std::array<double, CH_SPACEDIM> &center,
    const SimulationConfig &config,
    const GRHD::AtmosphereOptions &atmosphere,
    const GRHD::AtmosphereResetDiagnostics &recovery_totals)
{
    Diagnostics diagnostics;
    diagnostics.failed_recoveries = recovery_totals.num_failed_recoveries;
    diagnostics.floored_primitives = recovery_totals.num_floored_primitives;
    diagnostics.conserved_resets = recovery_totals.num_conserved_resets;

    const double cell_volume = std::pow(dx, CH_SPACEDIM);
    const auto background = background_factory(time);
    const BinaryState binary = make_binary_state(config, time);
    const DisjointBoxLayout &grids = state.disjointBoxLayout();
    DataIterator dit = grids.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        BoxIterator bit(grids[dit]);
        for (bit.begin(); bit.ok(); ++bit)
        {
            const IntVect iv = bit();
            const auto conserved = GRHD::load_conserved(state[dit], iv);
            const auto primitive = GRHD::load_primitive(state[dit], iv);
            const auto geometry = GRHD::load_fixed_background_geometry(
                background, iv, dx, center);
            const double proper_volume =
                std::exp(GRHD::spatial_metric_log_sqrt_det(geometry)) *
                cell_volume;
            diagnostics.mass += conserved.D * proper_volume;
            diagnostics.tau += conserved.tau * proper_volume;
            if (primitive.rho > 20.0 * atmosphere.rho_floor)
            {
                diagnostics.disk_mass += conserved.D * proper_volume;
                const Coordinates<double> coords(iv, dx, center);
                const double dx1 = coords.x - binary.position[0][0];
                const double dy1 = coords.y - binary.position[0][1];
                const double dx2 = coords.x - binary.position[1][0];
                const double dy2 = coords.y - binary.position[1][1];
                if (dx1 * dx1 + dy1 * dy1 <= dx2 * dx2 + dy2 * dy2)
                    diagnostics.disk_1_mass += conserved.D * proper_volume;
                else
                    diagnostics.disk_2_mass += conserved.D * proper_volume;
            }
            diagnostics.max_rho =
                std::max(diagnostics.max_rho, primitive.rho);
            diagnostics.max_pressure =
                std::max(diagnostics.max_pressure, primitive.pressure);
            const double velocity_squared = GRHD::compute_velocity_squared(
                primitive.velocity_U, geometry.spatial_metric_LL);
            diagnostics.max_velocity = std::max(
                diagnostics.max_velocity,
                std::sqrt(std::max(0.0, velocity_squared)));
            diagnostics.max_lorentz_factor = std::max(
                diagnostics.max_lorentz_factor,
                GRHD::compute_lorentz_factor(primitive.velocity_U,
                                             geometry.spatial_metric_LL));
        }
    }

#ifdef CH_MPI
    double local_sums[5] = {diagnostics.mass, diagnostics.disk_mass,
                            diagnostics.disk_1_mass, diagnostics.disk_2_mass,
                            diagnostics.tau};
    double global_sums[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    MPI_Allreduce(local_sums, global_sums, 5, MPI_DOUBLE, MPI_SUM,
                  Chombo_MPI::comm);
    double local_maxima[4] = {diagnostics.max_rho,
                              diagnostics.max_pressure,
                              diagnostics.max_velocity,
                              diagnostics.max_lorentz_factor};
    double global_maxima[4] = {0.0, 0.0, 0.0, 1.0};
    MPI_Allreduce(local_maxima, global_maxima, 4, MPI_DOUBLE, MPI_MAX,
                  Chombo_MPI::comm);
    diagnostics.mass = global_sums[0];
    diagnostics.disk_mass = global_sums[1];
    diagnostics.disk_1_mass = global_sums[2];
    diagnostics.disk_2_mass = global_sums[3];
    diagnostics.tau = global_sums[4];
    diagnostics.max_rho = global_maxima[0];
    diagnostics.max_pressure = global_maxima[1];
    diagnostics.max_velocity = global_maxima[2];
    diagnostics.max_lorentz_factor = global_maxima[3];
#endif
    return diagnostics;
}

bool leveldata_state_is_finite(const LevelData<FArrayBox> &state)
{
    bool local_ok = true;
    const DisjointBoxLayout &grids = state.disjointBoxLayout();
    DataIterator dit = grids.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        BoxIterator bit(grids[dit]);
        for (bit.begin(); bit.ok(); ++bit)
        {
            const auto conserved = GRHD::load_conserved(state[dit], bit());
            const auto primitive = GRHD::load_primitive(state[dit], bit());
            local_ok = local_ok && std::isfinite(conserved.D) &&
                       std::isfinite(conserved.tau) &&
                       std::isfinite(primitive.rho) &&
                       std::isfinite(primitive.pressure);
            FOR(dir)
            {
                local_ok = local_ok && std::isfinite(conserved.S_L[dir]) &&
                           std::isfinite(primitive.velocity_U[dir]);
            }
        }
    }
#ifdef CH_MPI
    int local_value = local_ok ? 1 : 0;
    int global_value = 0;
    MPI_Allreduce(&local_value, &global_value, 1, MPI_INT, MPI_MIN,
                  Chombo_MPI::comm);
    return global_value == 1;
#else
    return local_ok;
#endif
}

void append_diagnostics(std::ofstream &out, int step, double time,
                        const Diagnostics &diagnostics)
{
    out << step << ',' << std::setprecision(17) << time << ','
        << diagnostics.mass << ',' << diagnostics.disk_mass << ','
        << diagnostics.disk_1_mass << ',' << diagnostics.disk_2_mass << ','
        << diagnostics.tau << ',' << diagnostics.max_rho << ','
        << diagnostics.max_pressure << ',' << diagnostics.max_velocity << ','
        << diagnostics.max_lorentz_factor << ','
        << diagnostics.failed_recoveries << ','
        << diagnostics.floored_primitives << ','
        << diagnostics.conserved_resets << '\n';
}

std::string rank_suffix()
{
    std::ostringstream suffix;
    suffix << "rank" << std::setw(4) << std::setfill('0') << mpi_rank();
    return suffix.str();
}

std::string final_slice_path(const std::string &base)
{
    if (mpi_size() == 1)
        return base + "_final_slice.csv";
    return base + "_final_slice_" + rank_suffix() + ".csv";
}

void write_slice_csv(const std::string &path, const LevelData<FArrayBox> &state,
                     const BinaryPNFactory &background_factory, double time,
                     double dx,
                     const std::array<double, CH_SPACEDIM> &center,
                     const SimulationConfig &config, int k_mid)
{
    std::ofstream out(path.c_str());
    if (!out)
        throw std::runtime_error("could not open " + path);
    out << "x,y,radius_to_bh1,radius_to_bh2,rho,pressure,vx,vy,D,Sx,Sy,tau,lapse\n";
    out << std::setprecision(17);
    const auto background = background_factory(time);
    const BinaryState binary = make_binary_state(config, time);
    const DisjointBoxLayout &grids = state.disjointBoxLayout();
    DataIterator dit = grids.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        const Box &box = grids[dit];
        for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j)
        {
            for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i)
            {
                const IntVect iv(D_DECL(i, j, k_mid));
                const Coordinates<double> coords(iv, dx, center);
                const auto primitive = GRHD::load_primitive(state[dit], iv);
                const auto conserved = GRHD::load_conserved(state[dit], iv);
                const auto geometry = GRHD::load_fixed_background_geometry(
                    background, iv, dx, center);
                const double r1 = std::sqrt(
                    (coords.x - binary.position[0][0]) *
                        (coords.x - binary.position[0][0]) +
                    (coords.y - binary.position[0][1]) *
                        (coords.y - binary.position[0][1]));
                const double r2 = std::sqrt(
                    (coords.x - binary.position[1][0]) *
                        (coords.x - binary.position[1][0]) +
                    (coords.y - binary.position[1][1]) *
                        (coords.y - binary.position[1][1]));
                out << coords.x << ',' << coords.y << ',' << r1 << ','
                    << r2 << ',' << primitive.rho << ','
                    << primitive.pressure << ',' << primitive.velocity_U[0]
                    << ',' << primitive.velocity_U[1] << ',' << conserved.D
                    << ',' << conserved.S_L[0] << ',' << conserved.S_L[1]
                    << ',' << conserved.tau << ',' << geometry.lapse << '\n';
            }
        }
    }
}

std::array<unsigned char, 3> interpolate_color(double value)
{
    const std::array<std::array<int, 3>, 5> palette = {{{68, 1, 84},
                                                        {59, 82, 139},
                                                        {33, 145, 140},
                                                        {94, 201, 98},
                                                        {253, 231, 37}}};
    value = std::min(1.0, std::max(0.0, value));
    const double scaled = value * static_cast<double>(palette.size() - 1);
    int index = static_cast<int>(std::floor(scaled));
    if (index >= static_cast<int>(palette.size()) - 1)
        index = static_cast<int>(palette.size()) - 2;
    const double fraction = scaled - index;
    std::array<unsigned char, 3> color;
    for (int component = 0; component < 3; ++component)
    {
        const double channel =
            palette[index][component] +
            fraction * (palette[index + 1][component] -
                        palette[index][component]);
        color[component] = static_cast<unsigned char>(
            std::min(255.0, std::max(0.0, std::round(channel))));
    }
    return color;
}

void set_pixel(std::vector<unsigned char> &pixels, int width, int height,
               int x, int y, const std::array<unsigned char, 3> &color)
{
    if (x < 0 || x >= width || y < 0 || y >= height)
        return;
    const int offset = 3 * (y * width + x);
    pixels[offset] = color[0];
    pixels[offset + 1] = color[1];
    pixels[offset + 2] = color[2];
}

void draw_marker(std::vector<unsigned char> &pixels, int width, int height,
                 double x, double y, double dx,
                 const std::array<double, CH_SPACEDIM> &center,
                 int scale, const std::array<unsigned char, 3> &color)
{
    const int i = static_cast<int>(std::round((x + center[0]) / dx - 0.5));
    const int j = static_cast<int>(std::round((y + center[1]) / dx - 0.5));
    const int px = (i * scale) + scale / 2;
    const int py = height - 1 - ((j * scale) + scale / 2);
    const int radius = std::max(2, 2 * scale);
    for (int offset = -radius; offset <= radius; ++offset)
    {
        set_pixel(pixels, width, height, px + offset, py, color);
        set_pixel(pixels, width, height, px, py + offset, color);
    }
}

void write_density_frame_ppm(
    const std::string &frame_dir, const LevelData<FArrayBox> &state,
    const BinaryPNFactory &background_factory, double time, double dx,
    const std::array<double, CH_SPACEDIM> &center,
    const SimulationConfig &config, int frame_index)
{
    if (mpi_size() != 1 || !is_io_rank())
        return;
    ensure_directory(frame_dir);
    const int scale = std::max(1, config.frame_pixel_scale);
    const int width = config.num_cells * scale;
    const int height = config.num_cells * scale;
    std::vector<unsigned char> pixels(3 * width * height, 0);

    const double log_min = std::log10(config.rho_floor);
    const double log_max = std::log10(config.rho_peak);
    const double log_span = std::max(1.0e-300, log_max - log_min);
    const int k_mid = config.num_z_cells / 2;
    const DisjointBoxLayout &grids = state.disjointBoxLayout();
    DataIterator dit = grids.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        const Box &box = grids[dit];
        for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j)
        {
            for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i)
            {
                const IntVect iv(D_DECL(i, j, k_mid));
                const auto primitive = GRHD::load_primitive(state[dit], iv);
                const double value =
                    (std::log10(std::max(config.rho_floor, primitive.rho)) -
                     log_min) /
                    log_span;
                const auto color = interpolate_color(value);
                const int x0 = i * scale;
                const int y0 = height - (j + 1) * scale;
                for (int py = y0; py < y0 + scale; ++py)
                {
                    for (int px = x0; px < x0 + scale; ++px)
                        set_pixel(pixels, width, height, px, py, color);
                }
            }
        }
    }

    const BinaryState binary = make_binary_state(config, time);
    const std::array<unsigned char, 3> white = {{255, 255, 255}};
    draw_marker(pixels, width, height, binary.position[0][0],
                binary.position[0][1], dx, center, scale, white);
    draw_marker(pixels, width, height, binary.position[1][0],
                binary.position[1][1], dx, center, scale, white);

    std::ostringstream path;
    path << frame_dir << "/frame_" << std::setw(4) << std::setfill('0')
         << frame_index << ".ppm";
    std::ofstream out(path.str().c_str(), std::ios::binary);
    if (!out)
        throw std::runtime_error("could not open " + path.str());
    out << "P6\n" << width << ' ' << height << "\n255\n";
    out.write(reinterpret_cast<const char *>(pixels.data()),
              static_cast<std::streamsize>(pixels.size()));
    (void)background_factory;
}

void write_summary(const std::string &path, const SimulationConfig &config,
                   int num_boxes, int num_ranks,
                   const Diagnostics &initial_diagnostics,
                   const Diagnostics &final_diagnostics, double final_time,
                   int steps, double dx)
{
    std::ofstream out(path.c_str());
    if (!out)
        throw std::runtime_error("could not open " + path);
    out << std::setprecision(17);
    out << "driver GRHDLevelDataBinaryPNMiniDisks\n";
    out << "method fixed_grdzhadzha_circular_binary_pn_leveldata_two_effective_potential_minidisks_muscl_hlle_ssprk2\n";
    out << "num_cells " << config.num_cells << '\n';
    out << "num_z_cells " << config.num_z_cells << '\n';
    out << "num_boxes " << num_boxes << '\n';
    out << "mpi_ranks " << num_ranks << '\n';
    out << "dx " << dx << '\n';
    out << "domain_length " << config.domain_length << '\n';
    out << "final_time " << final_time << '\n';
    out << "max_dt " << config.max_dt << '\n';
    out << "steps " << steps << '\n';
    out << "mass_1 " << config.mass_1 << '\n';
    out << "mass_2 " << config.mass_2 << '\n';
    out << "separation " << config.separation << '\n';
    out << "binary_phase " << config.binary_phase << '\n';
    out << "binary_time " << config.binary_time << '\n';
    out << "final_binary_time " << config.binary_time + final_time << '\n';
    out << "binary_angular_velocity " << binary_angular_velocity(config)
        << '\n';
    out << "softening_radius " << config.softening_radius << '\n';
    out << "mini_disk_inner_radius " << config.mini_disk_inner_radius << '\n';
    out << "mini_disk_peak_radius " << config.mini_disk_peak_radius << '\n';
    out << "mini_disk_outer_radius " << config.mini_disk_outer_radius << '\n';
    out << "mini_disk_radial_width " << config.mini_disk_radial_width << '\n';
    out << "mini_disk_vertical_width " << config.mini_disk_vertical_width
        << '\n';
    out << "rho_peak " << config.rho_peak << '\n';
    out << "pressure_over_density " << config.pressure_over_density << '\n';
    out << "orbital_velocity_factor " << config.orbital_velocity_factor
        << '\n';
    out << "matched_atmosphere_width "
        << config.matched_atmosphere_width << '\n';
    out << "matched_atmosphere_density_factor "
        << config.matched_atmosphere_density_factor << '\n';
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
    out << "initial_disk_1_mass " << initial_diagnostics.disk_1_mass << '\n';
    out << "final_disk_1_mass " << final_diagnostics.disk_1_mass << '\n';
    out << "initial_disk_2_mass " << initial_diagnostics.disk_2_mass << '\n';
    out << "final_disk_2_mass " << final_diagnostics.disk_2_mass << '\n';
    out << "max_rho " << final_diagnostics.max_rho << '\n';
    out << "max_pressure " << final_diagnostics.max_pressure << '\n';
    out << "max_velocity " << final_diagnostics.max_velocity << '\n';
    out << "max_lorentz_factor " << final_diagnostics.max_lorentz_factor
        << '\n';
    out << "failed_recoveries " << final_diagnostics.failed_recoveries
        << '\n';
    out << "floored_primitives " << final_diagnostics.floored_primitives
        << '\n';
    out << "conserved_resets " << final_diagnostics.conserved_resets << '\n';
}
} // namespace

int main(int argc, char *argv[])
{
#ifdef CH_MPI
    MPI_Init(&argc, &argv);
#endif
    auto finish = [](int code) {
#ifdef CH_MPI
        MPI_Finalize();
#endif
        return code;
    };

    try
    {
        const SimulationConfig config = make_config_from_args(argc, argv);
        const IdealGasEOS eos(5.0 / 3.0);
        const double dx = config.domain_length / config.num_cells;
        const Box domain_box = make_domain_box(config);
        const ProblemDomain domain(domain_box);
        const DisjointBoxLayout grids = make_x_split_layout(domain_box);
        LevelData<FArrayBox> state(grids, NUM_VARS,
                                   config.num_ghosts * IntVect::Unit);
        const auto center = make_center(config, dx);
        const auto background_factory = make_background_factory(config, dx);
        const auto atmosphere = make_atmosphere_options(config);
        const auto recovery_options = make_recovery_options(config);

        Tensor<1, double> inverse_dx_U;
        FOR(dir) { inverse_dx_U[dir] = 1.0 / dx; }

        GRHD::TimeDependentFixedBGLevelDataFiniteVolumeOperator<
            IdealGasEOS, BinaryPNFactory>
            level_operator(eos, background_factory, dx, center, atmosphere,
                           config.limiter_theta, config.use_reconstruction,
                           recovery_options, config.use_static_metric_sources,
                           config.use_metric_volume_time_sources,
                           config.metric_time_derivative_step);

        double time = 0.0;
        fill_mini_disk_level_data(state, background_factory, eos, time, dx,
                                  center, config, atmosphere);
        GRHD::AtmosphereResetDiagnostics recovery_totals;
        recovery_totals.add(level_operator.recover_primitives(state, time));

        const std::string out_dir = default_output_dir();
        if (is_io_rank())
        {
            ensure_directory(out_dir);
            ensure_directory(out_dir + "/leveldata_binary_pn_minidisks_frames");
        }
        synchronize_ranks();
        const std::string base = out_dir + "/leveldata_binary_pn_minidisks";
        const std::string frame_dir =
            out_dir + "/leveldata_binary_pn_minidisks_frames";

        std::ofstream diagnostics_csv;
        if (is_io_rank())
        {
            diagnostics_csv.open((base + "_diagnostics.csv").c_str());
            if (!diagnostics_csv)
                throw std::runtime_error("could not open diagnostics CSV");
            diagnostics_csv
                << "step,time,mass,disk_mass,disk_1_mass,disk_2_mass,tau,"
                   "max_rho,max_pressure,max_velocity,max_lorentz_factor,"
                   "failed_recoveries,floored_primitives,conserved_resets\n";
        }

        Diagnostics initial_diagnostics = compute_diagnostics(
            state, background_factory, time, dx, center, config, atmosphere,
            recovery_totals);
        if (is_io_rank())
            append_diagnostics(diagnostics_csv, 0, time,
                               initial_diagnostics);
        if (config.write_density_frames && mpi_size() == 1)
            write_density_frame_ppm(frame_dir, state, background_factory, time,
                                    dx, center, config, 0);

        double next_output_time = config.output_interval;
        int steps = 0;
        int frame_index = 1;
        while (time < config.final_time - 1.0e-14)
        {
            recovery_totals.add(level_operator.recover_primitives(state,
                                                                  time));
            const double stable_dt = level_operator.compute_stable_dt(
                state, domain_box, inverse_dx_U, config.cfl, time);
            if (!std::isfinite(stable_dt) || stable_dt <= 0.0)
                throw std::runtime_error("stable dt became invalid");

            double dt = std::min(std::min(stable_dt, config.max_dt),
                                 config.final_time - time);
            if (time + dt > next_output_time)
                dt = next_output_time - time;
            if (dt <= 1.0e-14)
            {
                const Diagnostics diagnostics = compute_diagnostics(
                    state, background_factory, time, dx, center, config,
                    atmosphere, recovery_totals);
                if (is_io_rank())
                    append_diagnostics(diagnostics_csv, steps, time,
                                       diagnostics);
                if (config.write_density_frames && mpi_size() == 1)
                {
                    write_density_frame_ppm(frame_dir, state,
                                            background_factory, time, dx,
                                            center, config, frame_index);
                    ++frame_index;
                }
                next_output_time += config.output_interval;
                continue;
            }

            level_operator.advance_ssprk2(state, domain, inverse_dx_U, time,
                                          dt);
            time += dt;
            ++steps;
            recovery_totals.add(level_operator.recover_primitives(state,
                                                                  time));
            if (!leveldata_state_is_finite(state))
                throw std::runtime_error("state became non-finite");

            if (time >= next_output_time - 1.0e-12 ||
                time >= config.final_time - 1.0e-14)
            {
                const Diagnostics diagnostics = compute_diagnostics(
                    state, background_factory, time, dx, center, config,
                    atmosphere, recovery_totals);
                if (is_io_rank())
                    append_diagnostics(diagnostics_csv, steps, time,
                                       diagnostics);
                if (config.write_density_frames && mpi_size() == 1)
                {
                    write_density_frame_ppm(frame_dir, state,
                                            background_factory, time, dx,
                                            center, config, frame_index);
                    ++frame_index;
                }
                next_output_time += config.output_interval;
            }
        }

        const Diagnostics final_diagnostics = compute_diagnostics(
            state, background_factory, time, dx, center, config, atmosphere,
            recovery_totals);
        if (is_io_rank())
        {
            diagnostics_csv.close();
            write_summary(base + "_summary.txt", config, grids.size(),
                          mpi_size(), initial_diagnostics, final_diagnostics,
                          time, steps, dx);
        }

        write_slice_csv(final_slice_path(base), state, background_factory,
                        time, dx, center, config, config.num_z_cells / 2);
        synchronize_ranks();

        if (!std::isfinite(final_diagnostics.max_rho) ||
            final_diagnostics.max_rho <= config.rho_floor)
            throw std::runtime_error("mini-disk density became invalid");
        if (final_diagnostics.disk_mass <
            0.75 * initial_diagnostics.disk_mass)
            throw std::runtime_error("mini-disks lost too much mass");
        if (final_diagnostics.disk_1_mass <
                0.65 * initial_diagnostics.disk_1_mass ||
            final_diagnostics.disk_2_mass <
                0.65 * initial_diagnostics.disk_2_mass)
            throw std::runtime_error("one mini-disk lost too much mass");
        if (final_diagnostics.max_velocity >
            std::sqrt(config.max_velocity_squared) * 1.001)
            throw std::runtime_error("velocity cap was violated");

        if (is_io_rank())
        {
            std::cout << "GRHD LevelData binary PN mini-disks passed...\n"
                      << "  cells: " << config.num_cells << "^2 x "
                      << config.num_z_cells << "\n"
                      << "  boxes: " << grids.size() << "\n"
                      << "  mpi ranks: " << mpi_size() << "\n"
                      << "  steps: " << steps << "\n"
                      << "  final time: " << time << "\n"
                      << "  initial disk masses: "
                      << initial_diagnostics.disk_1_mass << ", "
                      << initial_diagnostics.disk_2_mass << "\n"
                      << "  final disk masses: "
                      << final_diagnostics.disk_1_mass << ", "
                      << final_diagnostics.disk_2_mass << "\n"
                      << "  max rho: " << final_diagnostics.max_rho << "\n"
                      << "  max velocity: "
                      << final_diagnostics.max_velocity << "\n"
                      << "  conserved resets: "
                      << final_diagnostics.conserved_resets << "\n"
                      << "  wrote " << base << "_diagnostics.csv\n"
                      << "  wrote " << final_slice_path(base)
                      << ((mpi_size() == 1) ? "" : " on each rank") << "\n"
                      << "  wrote " << base << "_summary.txt";
            if (config.write_density_frames && mpi_size() == 1)
                std::cout << "\n  wrote density frames under " << frame_dir;
            std::cout << std::endl;
        }
    }
    catch (const std::exception &error)
    {
        if (is_io_rank())
            std::cerr << "GRHD LevelData binary PN mini-disks failed: "
                      << error.what() << std::endl;
        return finish(1);
    }

    return finish(0);
}
