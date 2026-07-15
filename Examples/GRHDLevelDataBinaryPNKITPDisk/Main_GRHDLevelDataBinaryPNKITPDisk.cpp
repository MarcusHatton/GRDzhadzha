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
#include <cstdio>
#include <cstdint>
#include <cstdlib>
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

#ifdef CH_MPI
#include "mpi.h"
#endif

namespace
{
struct SimulationConfig
{
    int num_cells = 192;
    int num_z_cells = 25;
    int num_ghosts = 2;
    double domain_length = 96.0;
    double final_time = 0.5;
    double output_interval = 0.5;
    double cfl = 0.14;
    double max_dt = 0.02;

    double mass_1 = 0.5;
    double mass_2 = 0.5;
    double separation = 8.0;
    double binary_phase = 0.0;
    double binary_time = 0.0;
    double binary_angular_velocity = 0.0;
    double softening_radius = 0.5;
    double metric_time_derivative_step = 1.0e-3;

    double mach = 10.0;
    double aspect_ratio = 0.1;
    double cavity_radius_factor = 2.5;
    double cavity_delta = 1.0e-5;
    double outer_cutoff_radius = 1.0e10;
    double rho_peak = 4.0e-5;

    double rho_floor = 1.0e-10;
    double pressure_floor = 8.0e-14;
    double max_velocity_squared = 0.72;
    double limiter_theta = 1.3;
    bool use_reconstruction = true;
    bool use_static_metric_sources = true;
    bool use_metric_volume_time_sources = true;

    double alpha_viscosity = 0.03;
    double viscosity_beta = 0.0;
    double sink_size = 1.0;
    double sink_rate = 1.0e4;
    double sink_shape = 4.0;

    bool write_density_frames = true;
    int frame_pixel_scale = 4;

    double checkpoint_interval = 50.0;
    std::string restart_file;
    std::string output_dir;
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

struct Diagnostics
{
    double mass = 0.0;
    double disk_mass = 0.0;
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

int parse_positive_int(const char *text, const std::string &name)
{
    const double value = parse_positive_double(text, name);
    if (value > static_cast<double>(std::numeric_limits<int>::max()) ||
        std::floor(value) != value)
        throw std::runtime_error(name + " must be a positive integer");
    return static_cast<int>(value);
}

SimulationConfig make_config_from_args(int argc, char *argv[])
{
    SimulationConfig config;
    bool have_final_time = false;
    for (int arg = 1; arg < argc; ++arg)
    {
        const std::string option = argv[arg];
        auto require_value = [&](const std::string &name) {
            if (++arg >= argc)
                throw std::runtime_error(name + " requires a value");
            return argv[arg];
        };

        if (option == "--restart")
            config.restart_file = require_value(option);
        else if (option == "--checkpoint-interval")
            config.checkpoint_interval = parse_positive_double(
                require_value(option), "checkpoint_interval");
        else if (option == "--output-dir")
            config.output_dir = require_value(option);
        else if (option == "--num-cells")
            config.num_cells =
                parse_positive_int(require_value(option), "num_cells");
        else if (option == "--num-z-cells")
            config.num_z_cells =
                parse_positive_int(require_value(option), "num_z_cells");
        else if (option == "--domain-length")
            config.domain_length = parse_positive_double(
                require_value(option), "domain_length");
        else if (option == "--no-frames")
            config.write_density_frames = false;
        else if (!option.empty() && option[0] == '-')
            throw std::runtime_error("unknown option: " + option);
        else if (!have_final_time)
        {
            config.final_time = parse_positive_double(argv[arg], "final_time");
            have_final_time = true;
        }
        else
            throw std::runtime_error("multiple final_time arguments");
    }
    return config;
}

std::string default_output_dir()
{
    struct stat info;
    if (stat("Examples/GRHDLevelDataBinaryPNKITPDisk", &info) == 0)
        return "Examples/GRHDLevelDataBinaryPNKITPDisk/output";
    return "output";
}

Box make_domain_box(const SimulationConfig &config)
{
    return Box(IntVect(D_DECL(0, 0, 0)),
               IntVect(D_DECL(config.num_cells - 1,
                              config.num_cells - 1,
                              config.num_z_cells - 1)));
}

std::array<int, 2> choose_xy_tiling(const Box &domain_box, int num_tiles)
{
    const int nx = domain_box.size(0);
    const int ny = domain_box.size(1);
    num_tiles = std::max(1, std::min(num_tiles, nx * ny));

    std::array<int, 2> best = {{1, 1}};
    double best_score = std::numeric_limits<double>::max();
    for (int tiles_y = 1; tiles_y <= num_tiles; ++tiles_y)
    {
        if (num_tiles % tiles_y != 0)
            continue;
        const int tiles_x = num_tiles / tiles_y;
        if (tiles_x > nx || tiles_y > ny)
            continue;

        const double tile_dx = static_cast<double>(nx) / tiles_x;
        const double tile_dy = static_cast<double>(ny) / tiles_y;
        const double score = std::abs(std::log(tile_dx / tile_dy));
        if (score < best_score)
        {
            best_score = score;
            best = {{tiles_x, tiles_y}};
        }
    }
    return best;
}

DisjointBoxLayout make_xy_tiled_layout(const Box &domain_box)
{
    const int num_procs = mpi_size();
    const auto tiling = choose_xy_tiling(domain_box, num_procs);
    const int tiles_x = tiling[0];
    const int tiles_y = tiling[1];
    const int nx = domain_box.size(0);
    const int ny = domain_box.size(1);

    Vector<Box> boxes;
    Vector<int> proc_map;
    int proc = 0;
    for (int tile_y = 0; tile_y < tiles_y; ++tile_y)
    {
        for (int tile_x = 0; tile_x < tiles_x; ++tile_x)
        {
            IntVect lo = domain_box.smallEnd();
            IntVect hi = domain_box.bigEnd();
            lo[0] = domain_box.smallEnd(0) + (tile_x * nx) / tiles_x;
            hi[0] = domain_box.smallEnd(0) +
                    ((tile_x + 1) * nx) / tiles_x - 1;
            lo[1] = domain_box.smallEnd(1) + (tile_y * ny) / tiles_y;
            hi[1] = domain_box.smallEnd(1) +
                    ((tile_y + 1) * ny) / tiles_y - 1;
            boxes.push_back(Box(lo, hi));
            proc_map.push_back(proc % num_procs);
            ++proc;
        }
    }

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
    options.max_velocity_squared = std::min(0.95, config.max_velocity_squared + 0.03);
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

double stable_logistic_cutoff(double radius, double outer_radius)
{
    const double arg = -2.0 * (radius - outer_radius);
    if (arg > 700.0)
        return 1.0;
    if (arg < -700.0)
        return 0.0;
    return 1.0 - 1.0 / (1.0 + std::exp(arg));
}

double kitp_sigma_midplane(double radius, const SimulationConfig &config)
{
    const double safe_radius = std::max(radius, 1.0e-12);
    const double cavity_radius = config.cavity_radius_factor * config.separation;
    const double cavity_profile =
        (1.0 - config.cavity_delta) *
            std::exp(-std::pow(cavity_radius / safe_radius, 12.0)) +
        config.cavity_delta;
    return cavity_profile *
           stable_logistic_cutoff(radius, config.outer_cutoff_radius);
}

double kitp_radial_velocity(double radius, double phi,
                            const SimulationConfig &config)
{
    const double omega_bin = binary_angular_velocity(config);
    const double v0 = 1.0e-4 * omega_bin * config.separation;
    return v0 * std::sin(phi) * (radius / config.separation) *
           std::exp(-std::pow(radius / (3.5 * config.separation), 6.0));
}

double kitp_phi_velocity(double radius, const SimulationConfig &config)
{
    const double safe_radius = std::max(radius, 1.0e-12);
    const double omega_bin = binary_angular_velocity(config);
    const double omega_gas = std::sqrt(
        total_mass(config) / (safe_radius * safe_radius * safe_radius) *
        std::max(0.0, 1.0 - 1.0 / (config.mach * config.mach)));
    if (omega_gas <= 0.0)
        return 0.0;
    const double omega = std::pow(std::pow(omega_bin, -4.0) +
                                      std::pow(omega_gas, -4.0),
                                  -0.25);
    return omega * safe_radius;
}

double binary_point_mass_potential(
    const std::array<double, CH_SPACEDIM> &position,
    const BinaryState &binary, const SimulationConfig &config)
{
    double potential = 0.0;
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

template <class eos_t>
GRHD::Primitive<double> kitp_primitive(
    const std::array<double, CH_SPACEDIM> &position,
    const BinaryState &binary, const SimulationConfig &config,
    const eos_t &eos, const GRHD::CellGeometry &geometry,
    const GRHD::AtmosphereOptions &atmosphere)
{
    GRHD::Primitive<double> primitive =
        GRHD::make_atmosphere_primitive<double>(eos, atmosphere);

    const double radius = std::sqrt(position[0] * position[0] +
                                    position[1] * position[1]);
    if (radius <= 1.0e-12)
        return primitive;

    const double phi = std::atan2(position[1], position[0]);
    const double scale_height = std::max(config.aspect_ratio * radius, 1.0e-12);
    const double vertical_profile =
        std::exp(-0.5 * (position[2] / scale_height) *
                            (position[2] / scale_height));
    const double zero_sigma = config.rho_floor / config.rho_peak;
    const double density_scale =
        kitp_sigma_midplane(radius, config) * vertical_profile + zero_sigma;

    primitive.rho = std::max(config.rho_floor, config.rho_peak * density_scale);
    const double potential = binary_point_mass_potential(position, binary, config);
    const double sound_speed_squared =
        std::max(config.pressure_floor / primitive.rho,
                 -potential / (config.mach * config.mach));
    primitive.pressure = std::max(
        config.pressure_floor,
        primitive.rho * sound_speed_squared / eos.adiabatic_index());
    primitive.eps = primitive.pressure /
                    ((eos.adiabatic_index() - 1.0) * primitive.rho);

    const double radial_velocity = kitp_radial_velocity(radius, phi, config);
    const double phi_velocity = kitp_phi_velocity(radius, config);
    Tensor<1, double> coordinate_velocity;
    coordinate_velocity[0] =
        radial_velocity * std::cos(phi) - phi_velocity * std::sin(phi);
    coordinate_velocity[1] =
        radial_velocity * std::sin(phi) + phi_velocity * std::cos(phi);
    coordinate_velocity[2] = 0.0;

    FOR(dir)
    {
        primitive.velocity_U[dir] =
            (coordinate_velocity[dir] + geometry.shift_U[dir]) /
            geometry.lapse;
    }

    GRHD::enforce_primitive_floors(primitive, eos,
                                   geometry.spatial_metric_LL, atmosphere);
    return primitive;
}

template <class eos_t>
void fill_kitp_level_data(LevelData<FArrayBox> &state,
                          const BinaryPNFactory &background_factory,
                          const eos_t &eos, double time, double dx,
                          const std::array<double, CH_SPACEDIM> &center,
                          const SimulationConfig &config,
                          const GRHD::AtmosphereOptions &atmosphere)
{
    const auto background = background_factory(time);
    const BinaryState binary = make_binary_state(config, time);
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
            const auto primitive = kitp_primitive(
                position, binary, config, eos, geometry, atmosphere);
            GRHD::store_primitive(state[dit], iv, primitive);
            GRHD::store_conserved(
                state[dit], iv,
                GRHD::compute_conserved(primitive, eos,
                                        geometry.spatial_metric_LL));
        }
    }
}


double sink_kernel(double radius_squared, double sink_radius,
                   const SimulationConfig &config)
{
    if (sink_radius <= 0.0 || config.sink_rate == 0.0)
        return 0.0;
    const double size_squared = sink_radius * sink_radius;
    return std::exp(-std::pow(radius_squared / size_squared,
                              0.5 * config.sink_shape));
}

void add_kitp_sink_rhs(FArrayBox &rhs, const FArrayBox &state,
                       const Box &interior_box, double dx,
                       const std::array<double, CH_SPACEDIM> &center,
                       const BinaryState &binary,
                       const SimulationConfig &config,
                       const GRHD::AtmosphereOptions &atmosphere)
{
    if (config.sink_rate == 0.0 || config.sink_size <= 0.0)
        return;

    BoxIterator bit(interior_box);
    for (bit.begin(); bit.ok(); ++bit)
    {
        const IntVect iv = bit();
        const auto primitive = GRHD::load_primitive(state, iv);
        if (primitive.rho <= 10.0 * atmosphere.rho_floor)
            continue;

        const Coordinates<double> coords(iv, dx, center);
        const std::array<double, CH_SPACEDIM> position = {
            coords.x, coords.y, coords.z};
        double rate = 0.0;
        for (int body = 0; body < 2; ++body)
        {
            double radius_squared = 0.0;
            FOR(dir)
            {
                const double displacement =
                    position[dir] - binary.position[body][dir];
                radius_squared += displacement * displacement;
            }
            const double sink_radius = config.sink_size * binary.mass[body];
            rate += config.sink_rate *
                    sink_kernel(radius_squared, sink_radius, config);
        }
        if (rate == 0.0)
            continue;

        const auto conserved = GRHD::load_conserved(state, iv);
        rhs(iv, c_GRHD_D) -= rate * conserved.D;
        rhs(iv, c_GRHD_tau) -= rate * conserved.tau;
        FOR(dir) { rhs(iv, c_GRHD_S1 + dir) -= rate * conserved.S_L[dir]; }
    }
}

void apply_kitp_sink_decay(LevelData<FArrayBox> &state, double time,
                           double dt, double dx,
                           const std::array<double, CH_SPACEDIM> &center,
                           const SimulationConfig &config)
{
    if (config.sink_rate == 0.0 || config.sink_size <= 0.0)
        return;

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
            if (conserved.D <= 0.0)
                continue;

            const Coordinates<double> coords(iv, dx, center);
            const std::array<double, CH_SPACEDIM> position = {
                coords.x, coords.y, coords.z};
            double rate = 0.0;
            for (int body = 0; body < 2; ++body)
            {
                double radius_squared = 0.0;
                FOR(dir)
                {
                    const double displacement =
                        position[dir] - binary.position[body][dir];
                    radius_squared += displacement * displacement;
                }
                const double sink_radius =
                    config.sink_size * binary.mass[body];
                rate += config.sink_rate *
                        sink_kernel(radius_squared, sink_radius, config);
            }
            if (rate == 0.0)
                continue;

            const double decay = std::exp(-rate * dt);
            state[dit](iv, c_GRHD_D) *= decay;
            state[dit](iv, c_GRHD_tau) *= decay;
            FOR(dir)
            {
                state[dit](iv, c_GRHD_S1 + dir) *= decay;
            }
        }
    }
}

IntVect clamped_neighbor(const IntVect &iv, int direction, int offset,
                         const Box &domain_box)
{
    IntVect neighbor = iv;
    neighbor[direction] += offset;
    if (neighbor[direction] < domain_box.smallEnd(direction))
        neighbor[direction] = domain_box.smallEnd(direction);
    if (neighbor[direction] > domain_box.bigEnd(direction))
        neighbor[direction] = domain_box.bigEnd(direction);
    return neighbor;
}

double alpha_dynamic_viscosity(const GRHD::Primitive<double> &primitive,
                               const std::array<double, CH_SPACEDIM> &position,
                               const BinaryState &binary,
                               const SimulationConfig &config,
                               const IdealGasEOS &eos)
{
    if (config.alpha_viscosity == 0.0)
        return 0.0;
    double omega_squared = 0.0;
    for (int body = 0; body < 2; ++body)
    {
        double radius_squared = 1.0e-10;
        FOR(dir)
        {
            const double displacement =
                position[dir] - binary.position[body][dir];
            radius_squared += displacement * displacement;
        }
        omega_squared += binary.mass[body] * std::pow(radius_squared, -1.5);
    }
    const double sound_speed_squared =
        eos.adiabatic_index() * primitive.pressure /
        std::max(primitive.rho, config.rho_floor);
    return primitive.rho * config.alpha_viscosity * sound_speed_squared /
           std::sqrt(std::max(omega_squared, 1.0e-300));
}

GRHD::Conserved<double> viscous_flux_for_cell(
    const FArrayBox &state, const IntVect &iv, int flux_direction,
    const Box &domain_box, double dx,
    const std::array<double, CH_SPACEDIM> &center,
    const BinaryState &binary, const SimulationConfig &config,
    const IdealGasEOS &eos,
    const GRHD::AtmosphereOptions &atmosphere)
{
    GRHD::Conserved<double> flux = GRHD::zero_conserved<double>();
    const auto primitive = GRHD::load_primitive(state, iv);
    if (primitive.rho <= 10.0 * atmosphere.rho_floor)
        return flux;

    const Coordinates<double> coords(iv, dx, center);
    const std::array<double, CH_SPACEDIM> position = {
        coords.x, coords.y, coords.z};
    const double mu = alpha_dynamic_viscosity(
        primitive, position, binary, config, eos);
    if (mu == 0.0)
        return flux;

    double gradv[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
    for (int coord_dir = 0; coord_dir < 2; ++coord_dir)
    {
        const IntVect minus_iv = clamped_neighbor(iv, coord_dir, -1, domain_box);
        const IntVect plus_iv = clamped_neighbor(iv, coord_dir, 1, domain_box);
        const double denominator =
            std::max(1, plus_iv[coord_dir] - minus_iv[coord_dir]) * dx;
        const auto minus_primitive = GRHD::load_primitive(state, minus_iv);
        const auto plus_primitive = GRHD::load_primitive(state, plus_iv);
        for (int velocity_dir = 0; velocity_dir < 2; ++velocity_dir)
        {
            gradv[velocity_dir][coord_dir] =
                (plus_primitive.velocity_U[velocity_dir] -
                 minus_primitive.velocity_U[velocity_dir]) /
                denominator;
        }
    }

    const double divergence = gradv[0][0] + gradv[1][1];
    double tau[2][2];
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            tau[i][j] = mu * (gradv[i][j] + gradv[j][i] +
                              (config.viscosity_beta - 2.0 / 3.0) *
                                  static_cast<double>(i == j) * divergence);
        }
    }

    const double vx = primitive.velocity_U[0];
    const double vy = primitive.velocity_U[1];
    if (flux_direction == 0)
    {
        flux.S_L[0] = -tau[0][0];
        flux.S_L[1] = -tau[0][1];
        flux.tau = -tau[0][0] * vx - tau[0][1] * vy;
    }
    else
    {
        flux.S_L[0] = -tau[1][0];
        flux.S_L[1] = -tau[1][1];
        flux.tau = -tau[1][0] * vx - tau[1][1] * vy;
    }
    return flux;
}

void add_kitp_viscous_rhs(FArrayBox &rhs, const FArrayBox &state,
                          const Box &interior_box, const Box &domain_box,
                          double dx,
                          const std::array<double, CH_SPACEDIM> &center,
                          const BinaryState &binary,
                          const SimulationConfig &config,
                          const IdealGasEOS &eos,
                          const GRHD::AtmosphereOptions &atmosphere)
{
    if (config.alpha_viscosity == 0.0)
        return;

    BoxIterator bit(interior_box);
    for (bit.begin(); bit.ok(); ++bit)
    {
        const IntVect iv = bit();
        for (int direction = 0; direction < 2; ++direction)
        {
            const IntVect lo = clamped_neighbor(iv, direction, -1, domain_box);
            const IntVect hi = clamped_neighbor(iv, direction, 1, domain_box);
            const auto flux_lo = viscous_flux_for_cell(
                state, lo, direction, domain_box, dx, center, binary, config,
                eos, atmosphere);
            const auto flux_hi = viscous_flux_for_cell(
                state, hi, direction, domain_box, dx, center, binary, config,
                eos, atmosphere);
            rhs(iv, c_GRHD_tau) -= (flux_hi.tau - flux_lo.tau) /
                                   (2.0 * dx);
            rhs(iv, c_GRHD_S1) -= (flux_hi.S_L[0] - flux_lo.S_L[0]) /
                                  (2.0 * dx);
            rhs(iv, c_GRHD_S2) -= (flux_hi.S_L[1] - flux_lo.S_L[1]) /
                                  (2.0 * dx);
        }
    }
}

template <class eos_t>
void add_kitp_source_rhs(LevelData<FArrayBox> &rhs,
                         LevelData<FArrayBox> &state,
                         const Box &domain_box, double time, double dx,
                         const std::array<double, CH_SPACEDIM> &center,
                         const SimulationConfig &config, const eos_t &eos,
                         const GRHD::AtmosphereOptions &atmosphere)
{
    state.exchange();
    const BinaryState binary = make_binary_state(config, time);
    const DisjointBoxLayout &grids = state.disjointBoxLayout();
    DataIterator dit = grids.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        add_kitp_viscous_rhs(rhs[dit], state[dit], grids[dit], domain_box,
                             dx, center, binary, config, eos, atmosphere);
    }
}

template <class level_operator_t, class eos_t>
void compute_kitp_rhs(level_operator_t &level_operator,
                      LevelData<FArrayBox> &rhs, LevelData<FArrayBox> &state,
                      const ProblemDomain &domain,
                      const Tensor<1, double> &inverse_dx_U, double time,
                      double dx,
                      const std::array<double, CH_SPACEDIM> &center,
                      const SimulationConfig &config, const eos_t &eos,
                      const GRHD::AtmosphereOptions &atmosphere)
{
    level_operator.compute_rhs(rhs, state, domain, inverse_dx_U, time);
    add_kitp_source_rhs(rhs, state, domain.domainBox(), time, dx, center,
                        config, eos, atmosphere);
}

template <class level_operator_t, class eos_t>
void advance_kitp_ssprk2(level_operator_t &level_operator,
                         LevelData<FArrayBox> &state,
                         const ProblemDomain &domain,
                         const Tensor<1, double> &inverse_dx_U, double time,
                         double dt, double dx,
                         const std::array<double, CH_SPACEDIM> &center,
                         const SimulationConfig &config, const eos_t &eos,
                         const GRHD::AtmosphereOptions &atmosphere)
{
    const DisjointBoxLayout &grids = state.disjointBoxLayout();
    LevelData<FArrayBox> old_state(grids, NUM_VARS, state.ghostVect());
    LevelData<FArrayBox> stage_state(grids, NUM_VARS, state.ghostVect());
    LevelData<FArrayBox> rhs_initial(grids, NUM_VARS, IntVect::Zero);
    LevelData<FArrayBox> rhs_stage(grids, NUM_VARS, IntVect::Zero);

    state.copyTo(old_state);
    state.copyTo(stage_state);
    compute_kitp_rhs(level_operator, rhs_initial, state, domain, inverse_dx_U,
                     time, dx, center, config, eos, atmosphere);
    level_operator.update_conserved(stage_state, rhs_initial, dt, time + dt);
    compute_kitp_rhs(level_operator, rhs_stage, stage_state, domain,
                     inverse_dx_U, time + dt, dx, center, config, eos,
                     atmosphere);

    DataIterator dit = grids.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        GRHD::combine_ssprk2_conserved_state(state[dit], old_state[dit],
                                             stage_state[dit], rhs_stage[dit],
                                             grids[dit], dt);
    }
    apply_kitp_sink_decay(state, time + 0.5 * dt, dt, dx, center, config);
    level_operator.recover_primitives(state, time + dt);
}

Diagnostics compute_diagnostics(
    const LevelData<FArrayBox> &state,
    const BinaryPNFactory &background_factory, double time, double dx,
    const std::array<double, CH_SPACEDIM> &center,
    const GRHD::AtmosphereOptions &atmosphere,
    const GRHD::AtmosphereResetDiagnostics &recovery_totals)
{
    Diagnostics diagnostics;
    diagnostics.failed_recoveries = recovery_totals.num_failed_recoveries;
    diagnostics.floored_primitives = recovery_totals.num_floored_primitives;
    diagnostics.conserved_resets = recovery_totals.num_conserved_resets;

    const double cell_volume = std::pow(dx, CH_SPACEDIM);
    const auto background = background_factory(time);
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
            if (primitive.rho > 10.0 * atmosphere.rho_floor)
                diagnostics.disk_mass += conserved.D * proper_volume;
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
    double local_sums[3] = {diagnostics.mass, diagnostics.disk_mass,
                            diagnostics.tau};
    double global_sums[3] = {0.0, 0.0, 0.0};
    MPI_Allreduce(local_sums, global_sums, 3, MPI_DOUBLE, MPI_SUM,
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
    diagnostics.tau = global_sums[2];
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
                     int k_mid)
{
    std::ofstream out(path.c_str());
    if (!out)
        throw std::runtime_error("could not open " + path);
    out << "x,y,radius,rho,pressure,vx,vy,D,Sx,Sy,tau,lapse\n";
    out << std::setprecision(17);
    const auto background = background_factory(time);
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
                const double radius =
                    std::sqrt(coords.x * coords.x + coords.y * coords.y);
                out << coords.x << ',' << coords.y << ',' << radius << ','
                    << primitive.rho << ',' << primitive.pressure << ','
                    << primitive.velocity_U[0] << ','
                    << primitive.velocity_U[1] << ',' << conserved.D << ','
                    << conserved.S_L[0] << ',' << conserved.S_L[1] << ','
                    << conserved.tau << ',' << geometry.lapse << '\n';
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
    const int scale = std::max(1, config.frame_pixel_scale);
    const int width = config.num_cells * scale;
    const int height = config.num_cells * scale;
    const int num_pixels = config.num_cells * config.num_cells;
    std::vector<double> local_density(num_pixels, 0.0);

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
                local_density[j * config.num_cells + i] = primitive.rho;
            }
        }
    }

#ifdef CH_MPI
    std::vector<double> global_density(is_io_rank() ? num_pixels : 1, 0.0);
    MPI_Reduce(local_density.data(),
               is_io_rank() ? global_density.data() : nullptr, num_pixels,
               MPI_DOUBLE, MPI_SUM, 0, Chombo_MPI::comm);
    if (!is_io_rank())
        return;
    local_density.swap(global_density);
#else
    if (!is_io_rank())
        return;
#endif

    ensure_directory(frame_dir);
    std::vector<unsigned char> pixels(3 * width * height, 0);
    const double log_min = std::log10(config.rho_floor);
    const double log_max = std::log10(config.rho_peak);
    const double log_span = std::max(1.0e-300, log_max - log_min);
    for (int j = 0; j < config.num_cells; ++j)
    {
        for (int i = 0; i < config.num_cells; ++i)
        {
            const double rho = local_density[j * config.num_cells + i];
            const double value =
                (std::log10(std::max(config.rho_floor, rho)) - log_min) /
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

struct RestartData
{
    double time = 0.0;
    double next_output_time = 0.0;
    double next_checkpoint_time = 0.0;
    int steps = 0;
    int frame_index = 0;
    Diagnostics initial_diagnostics;
    GRHD::AtmosphereResetDiagnostics recovery_totals;
};

std::string config_signature(const SimulationConfig &config)
{
    std::ostringstream out;
    out << std::setprecision(17) << config.num_cells << ' '
        << config.num_z_cells << ' ' << config.num_ghosts << ' '
        << config.domain_length << ' ' << config.output_interval << ' '
        << config.cfl << ' ' << config.max_dt << ' ' << config.mass_1 << ' '
        << config.mass_2 << ' ' << config.separation << ' '
        << config.binary_phase << ' ' << config.binary_time << ' '
        << config.binary_angular_velocity << ' ' << config.softening_radius
        << ' ' << config.metric_time_derivative_step << ' ' << config.mach
        << ' ' << config.aspect_ratio << ' ' << config.cavity_radius_factor
        << ' ' << config.cavity_delta << ' ' << config.outer_cutoff_radius
        << ' ' << config.rho_peak << ' ' << config.rho_floor << ' '
        << config.pressure_floor << ' ' << config.max_velocity_squared << ' '
        << config.limiter_theta << ' ' << config.use_reconstruction << ' '
        << config.use_static_metric_sources << ' '
        << config.use_metric_volume_time_sources << ' '
        << config.alpha_viscosity << ' ' << config.viscosity_beta << ' '
        << config.sink_size << ' ' << config.sink_rate << ' '
        << config.sink_shape;
    return out.str();
}

template <class value_t>
void write_binary_value(std::ostream &out, const value_t &value)
{
    out.write(reinterpret_cast<const char *>(&value), sizeof(value_t));
}

template <class value_t>
value_t read_binary_value(std::istream &in)
{
    value_t value;
    in.read(reinterpret_cast<char *>(&value), sizeof(value_t));
    if (!in)
        throw std::runtime_error("checkpoint ended unexpectedly");
    return value;
}

void write_checkpoint_diagnostics(std::ostream &out,
                                  const Diagnostics &diagnostics)
{
    write_binary_value(out, diagnostics.mass);
    write_binary_value(out, diagnostics.disk_mass);
    write_binary_value(out, diagnostics.tau);
    write_binary_value(out, diagnostics.max_rho);
    write_binary_value(out, diagnostics.max_pressure);
    write_binary_value(out, diagnostics.max_velocity);
    write_binary_value(out, diagnostics.max_lorentz_factor);
    write_binary_value(out,
                       static_cast<std::uint64_t>(
                           diagnostics.failed_recoveries));
    write_binary_value(out,
                       static_cast<std::uint64_t>(
                           diagnostics.floored_primitives));
    write_binary_value(out,
                       static_cast<std::uint64_t>(
                           diagnostics.conserved_resets));
}

Diagnostics read_checkpoint_diagnostics(std::istream &in)
{
    Diagnostics diagnostics;
    diagnostics.mass = read_binary_value<double>(in);
    diagnostics.disk_mass = read_binary_value<double>(in);
    diagnostics.tau = read_binary_value<double>(in);
    diagnostics.max_rho = read_binary_value<double>(in);
    diagnostics.max_pressure = read_binary_value<double>(in);
    diagnostics.max_velocity = read_binary_value<double>(in);
    diagnostics.max_lorentz_factor = read_binary_value<double>(in);
    diagnostics.failed_recoveries =
        static_cast<long long>(read_binary_value<std::uint64_t>(in));
    diagnostics.floored_primitives =
        static_cast<long long>(read_binary_value<std::uint64_t>(in));
    diagnostics.conserved_resets =
        static_cast<long long>(read_binary_value<std::uint64_t>(in));
    return diagnostics;
}

struct CheckpointFileHeader
{
    RestartData restart;
    std::uint64_t value_count = 0;
    std::streampos state_offset;
};

CheckpointFileHeader read_checkpoint_header(
    std::istream &in, const SimulationConfig &config)
{
    char magic[8];
    in.read(magic, sizeof(magic));
    const std::string expected_magic("KITPCHK1", 8);
    if (!in || std::string(magic, sizeof(magic)) != expected_magic)
        throw std::runtime_error("not a KITP checkpoint");

    const std::uint32_t version =
        read_binary_value<std::uint32_t>(in);
    const std::int32_t num_components =
        read_binary_value<std::int32_t>(in);
    const std::int32_t num_cells = read_binary_value<std::int32_t>(in);
    const std::int32_t num_z_cells =
        read_binary_value<std::int32_t>(in);
    const std::int32_t num_ghosts =
        read_binary_value<std::int32_t>(in);
    if (version != 1 || num_components != NUM_VARS)
        throw std::runtime_error("unsupported KITP checkpoint format");
    if (num_cells != config.num_cells ||
        num_z_cells != config.num_z_cells ||
        num_ghosts != config.num_ghosts)
        throw std::runtime_error(
            "checkpoint grid does not match this run");

    CheckpointFileHeader header;
    header.restart.steps = read_binary_value<std::int32_t>(in);
    header.restart.frame_index = read_binary_value<std::int32_t>(in);
    header.restart.time = read_binary_value<double>(in);
    header.restart.next_output_time = read_binary_value<double>(in);
    header.restart.next_checkpoint_time = read_binary_value<double>(in);
    header.restart.recovery_totals.num_cells =
        static_cast<long long>(read_binary_value<std::uint64_t>(in));
    header.restart.recovery_totals.num_failed_recoveries =
        static_cast<long long>(read_binary_value<std::uint64_t>(in));
    header.restart.recovery_totals.num_floored_primitives =
        static_cast<long long>(read_binary_value<std::uint64_t>(in));
    header.restart.recovery_totals.num_conserved_resets =
        static_cast<long long>(read_binary_value<std::uint64_t>(in));
    header.restart.initial_diagnostics =
        read_checkpoint_diagnostics(in);

    const std::uint64_t signature_size =
        read_binary_value<std::uint64_t>(in);
    if (signature_size > 16384)
        throw std::runtime_error("invalid checkpoint configuration signature");
    std::string signature(static_cast<std::size_t>(signature_size), '\0');
    in.read(&signature[0],
            static_cast<std::streamsize>(signature.size()));
    if (!in || signature != config_signature(config))
        throw std::runtime_error(
            "checkpoint configuration does not match this run");

    header.value_count = read_binary_value<std::uint64_t>(in);
    header.state_offset = in.tellg();
    return header;
}

std::size_t checkpoint_value_index(
    const IntVect &iv, int component, const Box &storage_box)
{
    const std::size_t nx = storage_box.size(0);
    const std::size_t ny = storage_box.size(1);
    const std::size_t i = iv[0] - storage_box.smallEnd(0);
    const std::size_t j = iv[1] - storage_box.smallEnd(1);
    const std::size_t k = iv[2] - storage_box.smallEnd(2);
    return (((k * ny + j) * nx + i) * NUM_VARS +
            static_cast<std::size_t>(component));
}

Box checkpoint_storage_box(const SimulationConfig &config)
{
    Box box = make_domain_box(config);
    box.grow(config.num_ghosts);
    return box;
}

std::uint64_t checkpoint_value_count(const Box &storage_box)
{
    std::uint64_t count = NUM_VARS;
    FOR(dir)
    {
        count *= static_cast<std::uint64_t>(storage_box.size(dir));
    }
    return count;
}

std::string checkpoint_path(const std::string &checkpoint_dir, int steps)
{
    std::ostringstream path;
    path << checkpoint_dir << "/checkpoint_step_" << std::setw(8)
         << std::setfill('0') << steps << ".bin";
    return path.str();
}

void write_checkpoint(
    const std::string &path, const LevelData<FArrayBox> &state,
    const SimulationConfig &config, double time, int steps, int frame_index,
    double next_output_time, double next_checkpoint_time,
    const Diagnostics &initial_diagnostics,
    const GRHD::AtmosphereResetDiagnostics &recovery_totals)
{
    const Box domain_box = make_domain_box(config);
    const Box storage_box = checkpoint_storage_box(config);
    const std::uint64_t count64 = checkpoint_value_count(storage_box);
    if (count64 > static_cast<std::uint64_t>(
                      std::numeric_limits<int>::max()))
        throw std::runtime_error("checkpoint is too large for MPI reduction");
    const int count = static_cast<int>(count64);

    std::vector<double> local_state(static_cast<std::size_t>(count), 0.0);
    const DisjointBoxLayout &grids = state.disjointBoxLayout();
    DataIterator dit = grids.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        BoxIterator bit(state[dit].box());
        for (bit.begin(); bit.ok(); ++bit)
        {
            const IntVect iv = bit();
            IntVect owner_iv = iv;
            FOR(dir)
            {
                owner_iv[dir] =
                    std::max(domain_box.smallEnd(dir),
                             std::min(domain_box.bigEnd(dir), owner_iv[dir]));
            }
            if (!grids[dit].contains(owner_iv))
                continue;
            for (int component = 0; component < NUM_VARS; ++component)
                local_state[checkpoint_value_index(
                    iv, component, storage_box)] =
                    state[dit](iv, component);
        }
    }

#ifdef CH_MPI
    std::vector<double> global_state(
        is_io_rank() ? static_cast<std::size_t>(count) : 1, 0.0);
    MPI_Reduce(local_state.data(),
               is_io_rank() ? global_state.data() : nullptr, count,
               MPI_DOUBLE, MPI_SUM, 0, Chombo_MPI::comm);
#else
    std::vector<double> global_state;
    global_state.swap(local_state);
#endif

    int write_ok = 1;
    std::string write_error;
    if (is_io_rank())
    {
        try
        {
            const std::string temporary_path = path + ".tmp";
            std::ofstream out(temporary_path.c_str(),
                              std::ios::binary | std::ios::trunc);
            if (!out)
                throw std::runtime_error(
                    "could not create checkpoint " + temporary_path);
            out.write("KITPCHK1", 8);
            write_binary_value(out, static_cast<std::uint32_t>(1));
            write_binary_value(out,
                               static_cast<std::int32_t>(NUM_VARS));
            write_binary_value(
                out, static_cast<std::int32_t>(config.num_cells));
            write_binary_value(
                out, static_cast<std::int32_t>(config.num_z_cells));
            write_binary_value(
                out, static_cast<std::int32_t>(config.num_ghosts));
            write_binary_value(out, static_cast<std::int32_t>(steps));
            write_binary_value(out, static_cast<std::int32_t>(frame_index));
            write_binary_value(out, time);
            write_binary_value(out, next_output_time);
            write_binary_value(out, next_checkpoint_time);
            write_binary_value(
                out, static_cast<std::uint64_t>(recovery_totals.num_cells));
            write_binary_value(
                out, static_cast<std::uint64_t>(
                         recovery_totals.num_failed_recoveries));
            write_binary_value(
                out, static_cast<std::uint64_t>(
                         recovery_totals.num_floored_primitives));
            write_binary_value(
                out, static_cast<std::uint64_t>(
                         recovery_totals.num_conserved_resets));
            write_checkpoint_diagnostics(out, initial_diagnostics);
            const std::string signature = config_signature(config);
            write_binary_value(
                out, static_cast<std::uint64_t>(signature.size()));
            out.write(signature.data(),
                      static_cast<std::streamsize>(signature.size()));
            write_binary_value(out, count64);
            out.write(
                reinterpret_cast<const char *>(global_state.data()),
                static_cast<std::streamsize>(
                    count64 * sizeof(double)));
            out.close();
            if (!out)
                throw std::runtime_error(
                    "failed while writing checkpoint " + temporary_path);
            if (std::rename(temporary_path.c_str(), path.c_str()) != 0)
                throw std::runtime_error(
                    "could not finalize checkpoint " + path);
        }
        catch (const std::exception &error)
        {
            write_ok = 0;
            write_error = error.what();
        }
    }
#ifdef CH_MPI
    MPI_Bcast(&write_ok, 1, MPI_INT, 0, Chombo_MPI::comm);
#endif
    if (!write_ok)
        throw std::runtime_error(
            is_io_rank() ? write_error : "checkpoint write failed on rank 0");
}

RestartData read_checkpoint_metadata(const std::string &path,
                                     const SimulationConfig &config)
{
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in)
        throw std::runtime_error("could not open checkpoint " + path);
    return read_checkpoint_header(in, config).restart;
}

void read_checkpoint_state(const std::string &path,
                           LevelData<FArrayBox> &state,
                           const SimulationConfig &config)
{
    const Box storage_box = checkpoint_storage_box(config);
    const std::uint64_t expected_count =
        checkpoint_value_count(storage_box);
    if (expected_count > static_cast<std::uint64_t>(
                             std::numeric_limits<int>::max()))
        throw std::runtime_error("checkpoint is too large for MPI broadcast");

    std::vector<double> global_state(
        static_cast<std::size_t>(expected_count), 0.0);
    int read_ok = 1;
    std::string read_error;
    if (is_io_rank())
    {
        try
        {
            std::ifstream in(path.c_str(), std::ios::binary);
            if (!in)
                throw std::runtime_error(
                    "could not reopen checkpoint " + path);
            const CheckpointFileHeader header =
                read_checkpoint_header(in, config);
            if (header.value_count != expected_count)
                throw std::runtime_error(
                    "checkpoint state size does not match the grid");
            in.read(
                reinterpret_cast<char *>(global_state.data()),
                static_cast<std::streamsize>(
                    expected_count * sizeof(double)));
            if (!in)
                throw std::runtime_error(
                    "checkpoint state data is incomplete");
        }
        catch (const std::exception &error)
        {
            read_ok = 0;
            read_error = error.what();
        }
    }
#ifdef CH_MPI
    MPI_Bcast(&read_ok, 1, MPI_INT, 0, Chombo_MPI::comm);
#endif
    if (!read_ok)
        throw std::runtime_error(
            is_io_rank() ? read_error : "checkpoint read failed on rank 0");
#ifdef CH_MPI
    MPI_Bcast(global_state.data(), static_cast<int>(expected_count),
              MPI_DOUBLE, 0, Chombo_MPI::comm);
#endif

    const DisjointBoxLayout &grids = state.disjointBoxLayout();
    DataIterator dit = grids.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        BoxIterator bit(state[dit].box());
        for (bit.begin(); bit.ok(); ++bit)
        {
            const IntVect iv = bit();
            if (!storage_box.contains(iv))
                throw std::runtime_error(
                    "state ghost box exceeds checkpoint storage");
            for (int component = 0; component < NUM_VARS; ++component)
                state[dit](iv, component) =
                    global_state[checkpoint_value_index(
                        iv, component, storage_box)];
        }
    }
}

bool file_exists(const std::string &path)
{
    struct stat info;
    return stat(path.c_str(), &info) == 0;
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
    out << "driver GRHDLevelDataBinaryPNKITPDisk\n";
    out << "method fixed_grdzhadzha_circular_binary_pn_leveldata_kitp_circumbinary_disk_muscl_hlle_ssprk2\n";
    out << "num_cells " << config.num_cells << '\n';
    out << "num_z_cells " << config.num_z_cells << '\n';
    out << "num_boxes " << num_boxes << '\n';
    out << "mpi_ranks " << num_ranks << '\n';
    out << "slice_output "
        << ((num_ranks == 1) ? "single_file" : "rank_local_files") << '\n';
    out << "dx " << dx << '\n';
    out << "domain_length " << config.domain_length << '\n';
    out << "final_time " << final_time << '\n';
    out << "max_dt " << config.max_dt << '\n';
    out << "checkpoint_interval " << config.checkpoint_interval << '\n';
    out << "restart_file "
        << (config.restart_file.empty() ? "none" : config.restart_file)
        << '\n';
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
    out << "mach " << config.mach << '\n';
    out << "aspect_ratio " << config.aspect_ratio << '\n';
    out << "cavity_radius "
        << config.cavity_radius_factor * config.separation << '\n';
    out << "cavity_delta " << config.cavity_delta << '\n';
    out << "outer_cutoff_radius " << config.outer_cutoff_radius << '\n';
    out << "rho_peak " << config.rho_peak << '\n';
    out << "rho_floor " << config.rho_floor << '\n';
    out << "pressure_floor " << config.pressure_floor << '\n';
    out << "alpha_viscosity " << config.alpha_viscosity << '\n';
    out << "viscosity_beta " << config.viscosity_beta << '\n';
    out << "sink_size " << config.sink_size << '\n';
    out << "sink_rate " << config.sink_rate << '\n';
    out << "sink_shape " << config.sink_shape << '\n';
    out << "sink_update exact_exponential_operator_split" << '\n';
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
        const bool restarting = !config.restart_file.empty();

        RestartData restart;
        DisjointBoxLayout grids;
        if (restarting)
        {
            restart = read_checkpoint_metadata(config.restart_file, config);
            grids = make_xy_tiled_layout(domain_box);
            if (restart.time > config.final_time + 1.0e-14)
                throw std::runtime_error(
                    "final_time precedes the checkpoint time");
        }
        else
            grids = make_xy_tiled_layout(domain_box);

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

        double time = restarting ? restart.time : 0.0;
        int steps = restarting ? restart.steps : 0;
        int frame_index = restarting ? restart.frame_index : 1;
        double next_output_time =
            restarting ? restart.next_output_time : config.output_interval;
        double next_checkpoint_time =
            restarting ? restart.next_checkpoint_time
                       : config.checkpoint_interval;
        GRHD::AtmosphereResetDiagnostics recovery_totals =
            restarting ? restart.recovery_totals
                       : GRHD::AtmosphereResetDiagnostics();

        if (restarting)
            read_checkpoint_state(config.restart_file, state, config);
        else
        {
            fill_kitp_level_data(state, background_factory, eos, time, dx,
                                 center, config, atmosphere);
            recovery_totals.add(
                level_operator.recover_primitives(state, time));
        }

        const std::string out_dir =
            config.output_dir.empty() ? default_output_dir()
                                      : config.output_dir;
        const std::string checkpoint_dir = out_dir + "/checkpoints";
        if (is_io_rank())
        {
            ensure_directory(out_dir);
            ensure_directory(out_dir + "/leveldata_binary_pn_kitp_disk_frames");
            ensure_directory(checkpoint_dir);
        }
        synchronize_ranks();
        const std::string base = out_dir + "/leveldata_binary_pn_kitp_disk";
        const std::string frame_dir =
            out_dir + "/leveldata_binary_pn_kitp_disk_frames";
        const std::string diagnostics_path = base + "_diagnostics.csv";

        const bool append_existing_diagnostics =
            restarting && file_exists(diagnostics_path);
        std::ofstream diagnostics_csv;
        if (is_io_rank())
        {
            const std::ios_base::openmode mode =
                append_existing_diagnostics ? std::ios::app : std::ios::out;
            diagnostics_csv.open(diagnostics_path.c_str(), mode);
            if (!diagnostics_csv)
                throw std::runtime_error("could not open diagnostics CSV");
            if (!append_existing_diagnostics)
                diagnostics_csv
                    << "step,time,mass,disk_mass,tau,max_rho,max_pressure,"
                       "max_velocity,max_lorentz_factor,failed_recoveries,"
                       "floored_primitives,conserved_resets\n";
        }

        Diagnostics initial_diagnostics;
        if (restarting)
            initial_diagnostics = restart.initial_diagnostics;
        else
            initial_diagnostics = compute_diagnostics(
                state, background_factory, time, dx, center, atmosphere,
                recovery_totals);

        double last_diagnostics_time = -1.0;
        if (!append_existing_diagnostics)
        {
            const Diagnostics diagnostics = compute_diagnostics(
                state, background_factory, time, dx, center, atmosphere,
                recovery_totals);
            if (is_io_rank())
                append_diagnostics(diagnostics_csv, steps, time, diagnostics);
            last_diagnostics_time = time;
        }
        if (!restarting && config.write_density_frames)
            write_density_frame_ppm(frame_dir, state, background_factory, time,
                                    dx, center, config, 0);

        int last_checkpoint_step = restarting ? steps : -1;
        auto write_scheduled_output = [&]() {
            const Diagnostics diagnostics = compute_diagnostics(
                state, background_factory, time, dx, center, atmosphere,
                recovery_totals);
            if (is_io_rank())
                append_diagnostics(diagnostics_csv, steps, time, diagnostics);
            if (config.write_density_frames)
            {
                write_density_frame_ppm(frame_dir, state, background_factory,
                                        time, dx, center, config,
                                        frame_index);
                ++frame_index;
            }
            last_diagnostics_time = time;
            do
            {
                next_output_time += config.output_interval;
            } while (next_output_time <= time + 1.0e-12);
        };
        auto write_scheduled_checkpoint = [&]() {
            do
            {
                next_checkpoint_time += config.checkpoint_interval;
            } while (next_checkpoint_time <= time + 1.0e-12);
            const std::string path = checkpoint_path(checkpoint_dir, steps);
            write_checkpoint(path, state, config, time, steps, frame_index,
                             next_output_time, next_checkpoint_time,
                             initial_diagnostics, recovery_totals);
            last_checkpoint_step = steps;
            if (is_io_rank())
                std::cout << "Wrote checkpoint " << path << " at t=" << time
                          << std::endl;
        };

        while (time < config.final_time - 1.0e-14)
        {
            recovery_totals.add(
                level_operator.recover_primitives(state, time));
            const double stable_dt = level_operator.compute_stable_dt(
                state, domain_box, inverse_dx_U, config.cfl, time);
            if (!std::isfinite(stable_dt) || stable_dt <= 0.0)
                throw std::runtime_error("stable dt became invalid");

            double dt = std::min(std::min(stable_dt, config.max_dt),
                                 config.final_time - time);
            if (next_output_time > time + 1.0e-14)
                dt = std::min(dt, next_output_time - time);
            if (next_checkpoint_time > time + 1.0e-14)
                dt = std::min(dt, next_checkpoint_time - time);
            if (dt <= 1.0e-14)
            {
                bool handled_event = false;
                if (time >= next_output_time - 1.0e-12)
                {
                    write_scheduled_output();
                    handled_event = true;
                }
                if (time >= next_checkpoint_time - 1.0e-12)
                {
                    write_scheduled_checkpoint();
                    handled_event = true;
                }
                if (!handled_event)
                    throw std::runtime_error(
                        "timestep underflow away from an output event");
                continue;
            }

            advance_kitp_ssprk2(level_operator, state, domain, inverse_dx_U,
                                time, dt, dx, center, config, eos,
                                atmosphere);
            time += dt;
            ++steps;
            recovery_totals.add(
                level_operator.recover_primitives(state, time));
            if (!leveldata_state_is_finite(state))
                throw std::runtime_error("state became non-finite");

            if (time >= next_output_time - 1.0e-12)
                write_scheduled_output();
            if (time >= next_checkpoint_time - 1.0e-12)
                write_scheduled_checkpoint();
        }

        const Diagnostics final_diagnostics = compute_diagnostics(
            state, background_factory, time, dx, center, atmosphere,
            recovery_totals);
        if (std::abs(last_diagnostics_time - time) > 1.0e-12 &&
            is_io_rank())
            append_diagnostics(diagnostics_csv, steps, time,
                               final_diagnostics);

        if (last_checkpoint_step != steps)
        {
            const std::string path = checkpoint_path(checkpoint_dir, steps);
            write_checkpoint(path, state, config, time, steps, frame_index,
                             next_output_time, next_checkpoint_time,
                             initial_diagnostics, recovery_totals);
            last_checkpoint_step = steps;
            if (is_io_rank())
                std::cout << "Wrote final checkpoint " << path << " at t="
                          << time << std::endl;
        }

        if (is_io_rank())
        {
            diagnostics_csv.close();
            write_summary(base + "_summary.txt", config, grids.size(),
                          mpi_size(), initial_diagnostics, final_diagnostics,
                          time, steps, dx);
        }

        write_slice_csv(final_slice_path(base), state, background_factory,
                        time, dx, center, config.num_z_cells / 2);
        synchronize_ranks();

        if (!std::isfinite(final_diagnostics.max_rho) ||
            final_diagnostics.max_rho <= config.rho_floor)
            throw std::runtime_error("KITP disk density became invalid");
        if (final_diagnostics.disk_mass <
            0.25 * initial_diagnostics.disk_mass)
            throw std::runtime_error("KITP disk lost too much mass");
        if (final_diagnostics.max_velocity >
            std::sqrt(config.max_velocity_squared) * 1.001)
            throw std::runtime_error("velocity cap was violated");

        if (is_io_rank())
        {
            std::cout << "GRHD LevelData binary PN KITP disk passed...\n"
                      << "  cells: " << config.num_cells << "^2 x "
                      << config.num_z_cells << "\n"
                      << "  boxes: " << grids.size() << "\n"
                      << "  mpi ranks: " << mpi_size() << "\n"
                      << "  steps: " << steps << "\n"
                      << "  final time: " << time << "\n"
                      << "  initial disk mass: "
                      << initial_diagnostics.disk_mass << "\n"
                      << "  final disk mass: " << final_diagnostics.disk_mass
                      << "\n"
                      << "  max rho: " << final_diagnostics.max_rho << "\n"
                      << "  max velocity: "
                      << final_diagnostics.max_velocity << "\n"
                      << "  conserved resets: "
                      << final_diagnostics.conserved_resets << "\n"
                      << "  wrote " << base << "_diagnostics.csv\n"
                      << "  wrote " << final_slice_path(base)
                      << ((mpi_size() == 1) ? "" : " on each rank") << "\n"
                      << "  wrote " << base << "_summary.txt";
            if (config.write_density_frames)
                std::cout << "\n  wrote density frames under " << frame_dir;
            std::cout << std::endl;
        }
    }
    catch (const std::exception &error)
    {
        if (is_io_rank())
            std::cerr << "GRHD LevelData binary PN KITP disk failed: "
                      << error.what() << std::endl;
        return finish(1);
    }

    return finish(0);
}
