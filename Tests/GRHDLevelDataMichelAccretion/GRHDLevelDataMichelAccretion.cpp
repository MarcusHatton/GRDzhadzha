/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#include "Coordinates.hpp"
#include "FixedBGFArrayBoxFiniteVolume.hpp"
#include "IdealGasEOS.hpp"
#include "KerrSchild.hpp"
#include "TensorAlgebra.hpp"
#include "UserVariables.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/stat.h>

#ifdef CH_MPI
#include "mpi.h"
#endif

namespace
{
struct MichelConfig
{
    double mass = 1.0;
    double gamma = 5.0 / 3.0;
    double sonic_radius = 8.0;
    double sonic_density = 1.0e-6;
    double atmosphere_radius = 4.0;
    int num_cells = 18;
    double dx = 2.0;
    double final_time = 0.1;
    double max_dt = 0.02;
    double cfl = 0.2;
    double root_tolerance = 1.0e-13;
};

struct MichelConstants
{
    double polytropic_K = 0.0;
    double mass_flux = 0.0;
    double bernoulli = 0.0;
    double sonic_ur = 0.0;
    double sonic_sound_speed_squared = 0.0;
};

struct MichelPoint
{
    double radius = 0.0;
    double rho = 0.0;
    double pressure = 0.0;
    double eps = 0.0;
    double ur = 0.0;
    double vr = 0.0;
};

struct Diagnostics
{
    double mass = 0.0;
    double tau = 0.0;
    double max_rho = 0.0;
    double max_pressure = 0.0;
    double max_speed_squared = 0.0;
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

std::string rank_suffix()
{
    std::ostringstream suffix;
    suffix << "rank" << std::setw(4) << std::setfill('0') << mpi_rank();
    return suffix.str();
}

bool write_density_frames_enabled()
{
    const char *env = std::getenv("GRHD_MICHEL_WRITE_FRAMES");
    return env != nullptr && std::string(env) != "0";
}

std::string default_output_dir()
{
    struct stat info;
    if (stat("Tests/GRHDLevelDataMichelAccretion", &info) == 0)
        return "Tests/GRHDLevelDataMichelAccretion/output";
    return "output";
}

void ensure_output_dir(const std::string &dir)
{
    mkdir(dir.c_str(), 0755);
}

std::string density_frame_path(const std::string &out_dir, int step)
{
    std::ostringstream path;
    path << out_dir << "/grhd_leveldata_michel_density_frame_"
         << std::setw(4) << std::setfill('0') << step;
    if (mpi_size() > 1)
        path << "_" << rank_suffix();
    path << ".csv";
    return path.str();
}

KerrSchild make_background(const MichelConfig &config, double dx)
{
    KerrSchild::params_t params;
    params.mass = config.mass;
    params.spin = 0.0;
    params.center = {{0.0, 0.0, 0.0}};
    return KerrSchild(params, dx);
}

KerrSchild::params_t make_background_params(const MichelConfig &config)
{
    KerrSchild::params_t params;
    params.mass = config.mass;
    params.spin = 0.0;
    params.center = {{0.0, 0.0, 0.0}};
    return params;
}

GRHD::CellGeometry geometry_at_radius(const MichelConfig &config,
                                      double radius)
{
    const auto background = make_background(config, 1.0);
    Coordinates<double> coords(IntVect(D_DECL(0, 0, 0)), 1.0,
                               {0.0, 0.0, 0.0});
    coords.x = radius;
    coords.y = 0.0;
    coords.z = 0.0;
    return GRHD::make_cell_geometry_from_background(background, coords);
}

GRHD::AtmosphereOptions make_atmosphere_options()
{
    GRHD::AtmosphereOptions atmosphere;
    atmosphere.rho_floor = 1.0e-12;
    atmosphere.pressure_floor = 1.0e-16;
    atmosphere.max_velocity_squared = 0.8;
    return atmosphere;
}

GRHD::RecoveryOptions make_recovery_options(
    const GRHD::AtmosphereOptions &atmosphere)
{
    GRHD::RecoveryOptions options;
    options.max_iterations = 100;
    options.pressure_floor = atmosphere.pressure_floor;
    options.max_velocity_squared = atmosphere.max_velocity_squared;
    return options;
}

double pressure_from_density(const MichelConstants &constants,
                             const MichelConfig &config, double rho)
{
    return constants.polytropic_K * std::pow(rho, config.gamma);
}

double specific_enthalpy(const MichelConstants &constants,
                         const MichelConfig &config, double rho)
{
    const double pressure = pressure_from_density(constants, config, rho);
    return 1.0 + config.gamma * pressure / ((config.gamma - 1.0) * rho);
}

MichelConstants make_constants(const MichelConfig &config)
{
    MichelConstants constants;
    constants.sonic_sound_speed_squared =
        config.mass / (2.0 * config.sonic_radius - 3.0 * config.mass);
    constants.sonic_ur =
        -std::sqrt(config.mass / (2.0 * config.sonic_radius));

    const double q = constants.sonic_sound_speed_squared /
                     (config.gamma - constants.sonic_sound_speed_squared *
                                         config.gamma / (config.gamma - 1.0));
    constants.polytropic_K =
        q / std::pow(config.sonic_density, config.gamma - 1.0);

    const double h_c = specific_enthalpy(constants, config,
                                         config.sonic_density);
    const double f_c = GRHD::schwarzschild_areal_f(config.mass,
                                                   config.sonic_radius);
    constants.bernoulli =
        h_c * std::sqrt(f_c + constants.sonic_ur * constants.sonic_ur);
    constants.mass_flux = config.sonic_radius * config.sonic_radius *
                          config.sonic_density * constants.sonic_ur;
    return constants;
}

double bernoulli_residual(const MichelConstants &constants,
                          const MichelConfig &config, double radius,
                          double rho)
{
    const double f = GRHD::schwarzschild_areal_f(config.mass, radius);
    const double ur = constants.mass_flux / (radius * radius * rho);
    const double h = specific_enthalpy(constants, config, rho);
    return h * h * (f + ur * ur) - constants.bernoulli * constants.bernoulli;
}

double bisect_density_root(const MichelConstants &constants,
                           const MichelConfig &config, double radius,
                           double lower, double upper)
{
    double f_lower = bernoulli_residual(constants, config, radius, lower);
    double f_upper = bernoulli_residual(constants, config, radius, upper);
    if (f_lower == 0.0)
        return lower;
    if (f_upper == 0.0)
        return upper;
    if (f_lower * f_upper > 0.0)
        throw std::runtime_error("Michel density bracket has no sign change");

    for (int iter = 0; iter < 200; ++iter)
    {
        const double mid = 0.5 * (lower + upper);
        const double f_mid = bernoulli_residual(constants, config, radius, mid);
        if (std::abs(f_mid) < config.root_tolerance ||
            std::abs(upper - lower) <=
                config.root_tolerance * std::max(1.0e-300, std::abs(mid)))
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

double solve_density(const MichelConstants &constants,
                     const MichelConfig &config, double radius)
{
    if (std::abs(radius - config.sonic_radius) < 1.0e-12)
        return config.sonic_density;

    const double log_min = std::log(config.sonic_density * 1.0e-6);
    const double log_max = std::log(config.sonic_density * 1.0e6);
    const int samples = 6000;
    struct Bracket
    {
        double lower;
        double upper;
        double midpoint;
    };
    std::vector<Bracket> brackets;

    double last_rho = std::exp(log_min);
    double last_value = bernoulli_residual(constants, config, radius,
                                           last_rho);
    for (int i = 1; i <= samples; ++i)
    {
        const double weight = static_cast<double>(i) / samples;
        const double rho = std::exp(log_min + weight * (log_max - log_min));
        const double value = bernoulli_residual(constants, config, radius,
                                                rho);
        if (last_value == 0.0 || value == 0.0 || last_value * value < 0.0)
        {
            brackets.push_back({last_rho, rho, 0.5 * (last_rho + rho)});
        }
        last_rho = rho;
        last_value = value;
    }

    if (brackets.empty())
        throw std::runtime_error("Michel density solve found no root bracket");

    const bool inner_branch = radius < config.sonic_radius;
    const auto selected = std::find_if(
        brackets.begin(), brackets.end(), [&](const Bracket &bracket) {
            return inner_branch ? bracket.midpoint > config.sonic_density
                                : bracket.midpoint < config.sonic_density;
        });
    const Bracket &bracket =
        (selected != brackets.end()) ? *selected : brackets.front();
    return bisect_density_root(constants, config, radius, bracket.lower,
                               bracket.upper);
}

MichelPoint make_point(const MichelConstants &constants,
                       const MichelConfig &config, const IdealGasEOS &eos,
                       double radius)
{
    const double f = GRHD::schwarzschild_areal_f(config.mass, radius);
    if (f <= 0.0)
        throw std::runtime_error("Michel profile radius is inside the horizon");

    MichelPoint point;
    point.radius = radius;
    point.rho = solve_density(constants, config, radius);
    point.pressure = pressure_from_density(constants, config, point.rho);
    point.eps = point.pressure / ((config.gamma - 1.0) * point.rho);
    point.ur = constants.mass_flux / (radius * radius * point.rho);
    const auto geometry = geometry_at_radius(config, radius);

    Tensor<1, double> shift_L;
    FOR(i) { shift_L[i] = 0.0; }
    FOR(i, j)
    {
        shift_L[i] += geometry.spatial_metric_LL[i][j] * geometry.shift_U[j];
    }

    const double shift_squared = TensorAlgebra::compute_dot_product(
        geometry.shift_U, geometry.shift_U, geometry.spatial_metric_LL);
    const double g_tt = -geometry.lapse * geometry.lapse + shift_squared;
    const double u_t = -std::sqrt(f + point.ur * point.ur);
    const double u_upper_t = (u_t - shift_L[0] * point.ur) / g_tt;
    const double W = geometry.lapse * u_upper_t;
    if (!(W > 1.0) || !std::isfinite(W))
        throw std::runtime_error("Michel Kerr-Schild Lorentz factor invalid");

    point.vr = point.ur / W + geometry.shift_U[0] / geometry.lapse;
    (void)eos;
    return point;
}

GRHD::Primitive<double> primitive_from_radius(
    const MichelConstants &constants, const MichelConfig &config,
    const IdealGasEOS &eos, double radius,
    const std::array<double, CH_SPACEDIM> &position,
    const GRHD::CellGeometry &geometry,
    const GRHD::AtmosphereOptions &atmosphere)
{
    if (radius < config.atmosphere_radius)
        return GRHD::make_atmosphere_primitive<double>(eos, atmosphere);

    const auto point = make_point(constants, config, eos, radius);
    GRHD::Primitive<double> primitive;
    primitive.rho = point.rho;
    primitive.eps = point.eps;
    primitive.pressure = point.pressure;
    FOR(i)
    {
        primitive.velocity_U[i] = point.vr * position[i] / radius;
    }
    GRHD::enforce_primitive_floors(primitive, eos, geometry.spatial_metric_LL,
                                   atmosphere);
    return primitive;
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

    int num_procs = 1;
#ifdef CH_MPI
    MPI_Comm_size(Chombo_MPI::comm, &num_procs);
#endif
    Vector<int> proc_map(2);
    proc_map[0] = 0;
    proc_map[1] = (num_procs > 1) ? 1 : 0;

    DisjointBoxLayout grids(boxes, proc_map, ProblemDomain(domain_box));
    grids.close();
    return grids;
}

void fill_michel_state(LevelData<FArrayBox> &state,
                       const MichelConstants &constants,
                       const MichelConfig &config, const KerrSchild &background,
                       const IdealGasEOS &eos,
                       const std::array<double, CH_SPACEDIM> &center,
                       const GRHD::AtmosphereOptions &atmosphere)
{
    const DisjointBoxLayout &grids = state.disjointBoxLayout();
    DataIterator dit = grids.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        state[dit].setVal(0.0);
        BoxIterator bit(state[dit].box());
        for (bit.begin(); bit.ok(); ++bit)
        {
            const IntVect iv = bit();
            const Coordinates<double> coords(iv, config.dx, center);
            const std::array<double, CH_SPACEDIM> position = {
                coords.x, coords.y, coords.z};
            const double radius = std::sqrt(coords.x * coords.x +
                                            coords.y * coords.y +
                                            coords.z * coords.z);
            const auto geometry = GRHD::load_fixed_background_geometry(
                background, iv, config.dx, center);
            const auto primitive = primitive_from_radius(
                constants, config, eos, radius, position, geometry,
                atmosphere);
            GRHD::store_primitive(state[dit], iv, primitive);
            GRHD::store_conserved(
                state[dit], iv,
                GRHD::compute_conserved(primitive, eos,
                                        geometry.spatial_metric_LL));
        }
    }
}

void write_density_frame(const std::string &out_dir,
                         const LevelData<FArrayBox> &state,
                         const MichelConfig &config,
                         const std::array<double, CH_SPACEDIM> &center,
                         int step, double time)
{
    ensure_output_dir(out_dir);
    const std::string path = density_frame_path(out_dir, step);
    std::ofstream out(path.c_str());
    if (!out)
        throw std::runtime_error("could not open " + path);

    out << "step,time,x,y,z,radius,rho,pressure,D,tau,vx,vy,vz\n";
    out << std::setprecision(17);

    const int k_mid = config.num_cells / 2;
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
                const Coordinates<double> coords(iv, config.dx, center);
                const auto primitive = GRHD::load_primitive(state[dit], iv);
                const auto conserved = GRHD::load_conserved(state[dit], iv);
                const double radius = std::sqrt(coords.x * coords.x +
                                                coords.y * coords.y +
                                                coords.z * coords.z);
                out << step << ',' << time << ',' << coords.x << ','
                    << coords.y << ',' << coords.z << ',' << radius << ','
                    << primitive.rho << ',' << primitive.pressure << ','
                    << conserved.D << ',' << conserved.tau << ','
                    << primitive.velocity_U[0] << ','
                    << primitive.velocity_U[1] << ','
                    << primitive.velocity_U[2] << '\n';
            }
        }
    }
}

Diagnostics compute_diagnostics(
    const LevelData<FArrayBox> &state, const KerrSchild &background,
    const MichelConfig &config, const IdealGasEOS &eos,
    const std::array<double, CH_SPACEDIM> &center)
{
    Diagnostics diagnostics;
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
                background, iv, config.dx, center);
            diagnostics.mass += conserved.D;
            diagnostics.tau += conserved.tau;
            diagnostics.max_rho = std::max(diagnostics.max_rho,
                                           primitive.rho);
            diagnostics.max_pressure = std::max(diagnostics.max_pressure,
                                                primitive.pressure);
            diagnostics.max_speed_squared = std::max(
                diagnostics.max_speed_squared,
                GRHD::compute_velocity_squared(primitive.velocity_U,
                                               geometry.spatial_metric_LL));
        }
    }
#ifdef CH_MPI
    double local_sums[2] = {diagnostics.mass, diagnostics.tau};
    double global_sums[2] = {0.0, 0.0};
    MPI_Allreduce(local_sums, global_sums, 2, MPI_DOUBLE, MPI_SUM,
                  Chombo_MPI::comm);
    double local_maxima[3] = {diagnostics.max_rho,
                              diagnostics.max_pressure,
                              diagnostics.max_speed_squared};
    double global_maxima[3] = {0.0, 0.0, 0.0};
    MPI_Allreduce(local_maxima, global_maxima, 3, MPI_DOUBLE, MPI_MAX,
                  Chombo_MPI::comm);
    diagnostics.mass = global_sums[0];
    diagnostics.tau = global_sums[1];
    diagnostics.max_rho = global_maxima[0];
    diagnostics.max_pressure = global_maxima[1];
    diagnostics.max_speed_squared = global_maxima[2];
#else
    (void)eos;
#endif
    return diagnostics;
}

bool diagnostics_are_valid(const Diagnostics &diagnostics,
                           const GRHD::AtmosphereOptions &atmosphere)
{
    return std::isfinite(diagnostics.mass) && diagnostics.mass > 0.0 &&
           std::isfinite(diagnostics.tau) &&
           std::isfinite(diagnostics.max_rho) &&
           diagnostics.max_rho > atmosphere.rho_floor &&
           std::isfinite(diagnostics.max_pressure) &&
           diagnostics.max_pressure > atmosphere.pressure_floor &&
           std::isfinite(diagnostics.max_speed_squared) &&
           diagnostics.max_speed_squared < atmosphere.max_velocity_squared;
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
} // namespace

int main(int argc, char *argv[])
{
#ifdef CH_MPI
    MPI_Init(&argc, &argv);
#else
    (void)argc;
    (void)argv;
#endif
    auto finish = [](int code) {
#ifdef CH_MPI
        MPI_Finalize();
#endif
        return code;
    };

    try
    {
        const MichelConfig config;
        const IdealGasEOS eos(config.gamma);
        const MichelConstants constants = make_constants(config);
        const auto atmosphere = make_atmosphere_options();
        const auto recovery_options = make_recovery_options(atmosphere);
        const KerrSchild background(make_background_params(config), config.dx);

        const Box domain_box(IntVect(D_DECL(0, 0, 0)),
                             IntVect(D_DECL(config.num_cells - 1,
                                            config.num_cells - 1,
                                            config.num_cells - 1)));
        const ProblemDomain domain(domain_box);
        const DisjointBoxLayout grids = make_x_split_layout(domain_box);
        LevelData<FArrayBox> state(grids, NUM_VARS, 2 * IntVect::Unit);
        const std::array<double, CH_SPACEDIM> center = {
            0.5 * config.num_cells * config.dx,
            0.5 * config.num_cells * config.dx,
            0.5 * config.num_cells * config.dx};

        fill_michel_state(state, constants, config, background, eos, center,
                          atmosphere);

        GRHD::FixedBGLevelDataFiniteVolumeOperator<IdealGasEOS, KerrSchild>
            level_operator(eos, background, config.dx, center, atmosphere,
                           1.2, true, recovery_options, true);
        const auto initial_recovery = level_operator.recover_primitives(state);
        if (initial_recovery.num_failed_recoveries != 0 ||
            initial_recovery.num_conserved_resets != 0)
        {
            std::cout << "initial Michel recovery reset unexpectedly"
                      << std::endl;
            return finish(1);
        }

        Tensor<1, double> inverse_dx_U;
        FOR(dir) { inverse_dx_U[dir] = 1.0 / config.dx; }
        double time = 0.0;
        const auto initial_diagnostics = compute_diagnostics(
            state, background, config, eos, center);
        const std::string output_dir = default_output_dir();
        if (write_density_frames_enabled())
            write_density_frame(output_dir, state, config, center, 0, time);
        if (!diagnostics_are_valid(initial_diagnostics, atmosphere))
        {
            std::cout << "initial Michel diagnostics are invalid" << std::endl;
            return finish(1);
        }

        int steps = 0;
        while (time < config.final_time - 1.0e-14)
        {
            const double stable_dt = level_operator.compute_stable_dt(
                state, domain_box, inverse_dx_U, config.cfl);
            if (!std::isfinite(stable_dt) || stable_dt <= 0.0)
                throw std::runtime_error("Michel stable dt became invalid");
            const double dt = std::min(std::min(stable_dt, config.max_dt),
                                       config.final_time - time);
            level_operator.advance_ssprk2(state, domain, inverse_dx_U, dt);
            time += dt;
            ++steps;
            if (!leveldata_state_is_finite(state))
                throw std::runtime_error("Michel state became non-finite");
            if (write_density_frames_enabled())
                write_density_frame(output_dir, state, config, center, steps,
                                    time);
        }

        const auto final_diagnostics = compute_diagnostics(
            state, background, config, eos, center);
        if (!diagnostics_are_valid(final_diagnostics, atmosphere))
        {
            std::cout << "final Michel diagnostics are invalid" << std::endl;
            return finish(1);
        }
        const double relative_mass_change =
            (final_diagnostics.mass - initial_diagnostics.mass) /
            initial_diagnostics.mass;
        if (std::abs(relative_mass_change) > 0.1)
        {
            std::cout << "Michel mass changed too much: "
                      << relative_mass_change << std::endl;
            return finish(1);
        }

        std::cout << "GRHD LevelData Michel accretion test passed...\n"
                  << "  boxes: " << grids.size() << "\n"
                  << "  steps: " << steps << "\n"
                  << "  final time: " << time << "\n"
                  << "  initial mass: " << initial_diagnostics.mass << "\n"
                  << "  final mass: " << final_diagnostics.mass << "\n"
                  << "  relative mass change: " << relative_mass_change
                  << "\n"
                  << "  max rho: " << final_diagnostics.max_rho << "\n"
                  << "  max pressure: " << final_diagnostics.max_pressure
                  << "\n"
                  << "  max speed squared: "
                  << final_diagnostics.max_speed_squared;
        if (write_density_frames_enabled())
            std::cout << "\n  density frames: " << output_dir;
        std::cout << std::endl;
        return finish(0);
    }
    catch (const std::exception &error)
    {
        std::cout << "GRHD LevelData Michel accretion test failed: "
                  << error.what() << std::endl;
        return finish(1);
    }
}
