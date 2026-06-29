/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#include "CircularBinaryPN.hpp"
#include "Coordinates.hpp"
#include "FixedBGFArrayBoxFiniteVolume.hpp"
#include "FixedBGTorus.hpp"
#include "IdealGasEOS.hpp"
#include "TensorAlgebra.hpp"
#include "UserVariables.hpp"
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

#ifdef CH_MPI
#include "mpi.h"
#endif

namespace
{
struct SimulationConfig
{
    int num_cells = 56;
    int num_z_cells = 9;
    int num_ghosts = 2;
    double domain_length = 80.0;
    double final_time = 2.0;
    double output_interval = 0.5;
    double cfl = 0.18;
    double max_dt = 0.05;

    double mass_1 = 0.5;
    double mass_2 = 0.5;
    double separation = 8.0;
    double binary_phase = 0.0;
    double binary_time = 0.0;
    double binary_angular_velocity = 0.0;
    double softening_radius = 1.0;
    double metric_time_derivative_step = 1.0e-3;

    double torus_inner_radius = 14.0;
    double torus_pressure_max_radius = 18.0;
    double rho_peak = 2.0e-5;

    double rho_floor = 1.0e-10;
    double pressure_floor = 1.0e-12;
    double max_velocity_squared = 0.8;
    double limiter_theta = 1.4;
    bool use_reconstruction = true;
    bool use_static_metric_sources = true;
    bool use_metric_volume_time_sources = true;
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

using TorusParameters = GRHD::FixedBGTorusParameters;

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

std::string default_output_dir()
{
    struct stat info;
    if (stat("Examples/GRHDLevelDataBinaryPNDisk", &info) == 0)
        return "Examples/GRHDLevelDataBinaryPNDisk/output";
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

GRHD::FixedBGTorusConfig make_torus_config(const SimulationConfig &config)
{
    GRHD::FixedBGTorusConfig torus_config;
    torus_config.total_mass = total_mass(config);
    torus_config.inner_radius = config.torus_inner_radius;
    torus_config.pressure_max_radius = config.torus_pressure_max_radius;
    torus_config.rho_peak = config.rho_peak;
    torus_config.max_velocity_squared = config.max_velocity_squared;
    return torus_config;
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

void write_summary(const std::string &path, const SimulationConfig &config,
                   const TorusParameters &torus, int num_boxes,
                   int num_ranks,
                   const Diagnostics &initial_diagnostics,
                   const Diagnostics &final_diagnostics, double final_time,
                   int steps, double dx)
{
    std::ofstream out(path.c_str());
    if (!out)
        throw std::runtime_error("could not open " + path);
    out << std::setprecision(17);
    out << "driver GRHDLevelDataBinaryPNDisk\n";
    out << "method fixed_grdzhadzha_circular_binary_pn_leveldata_constant_ell_torus_muscl_hlle_ssprk2\n";
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
    out << "steps " << steps << '\n';
    out << "mass_1 " << config.mass_1 << '\n';
    out << "mass_2 " << config.mass_2 << '\n';
    out << "separation " << config.separation << '\n';
    out << "binary_phase " << config.binary_phase << '\n';
    out << "binary_time " << config.binary_time << '\n';
    out << "final_binary_time " << config.binary_time + final_time << '\n';
    out << "softening_radius " << config.softening_radius << '\n';
    out << "torus_inner_radius " << config.torus_inner_radius << '\n';
    out << "torus_pressure_max_radius "
        << config.torus_pressure_max_radius << '\n';
    out << "torus_specific_angular_momentum " << torus.ell << '\n';
    out << "torus_surface_potential " << torus.surface_potential << '\n';
    out << "torus_max_enthalpy " << torus.max_enthalpy << '\n';
    out << "torus_polytropic_constant " << torus.polytropic_constant << '\n';
    out << "rho_peak " << config.rho_peak << '\n';
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
        const SimulationConfig config;
        const IdealGasEOS eos(5.0 / 3.0);
        const double dx = config.domain_length / config.num_cells;
        const Box domain_box = make_domain_box(config);
        const ProblemDomain domain(domain_box);
        const DisjointBoxLayout grids = make_x_split_layout(domain_box);
        LevelData<FArrayBox> state(grids, NUM_VARS,
                                   config.num_ghosts * IntVect::Unit);
        const auto center = make_center(config, dx);
        const auto background_factory = make_background_factory(config, dx);
        const auto initial_background = background_factory(0.0);
        const auto atmosphere = make_atmosphere_options(config);
        const auto recovery_options = make_recovery_options(config);
        const auto torus_config = make_torus_config(config);
        const TorusParameters torus = GRHD::make_fixed_bg_torus_parameters(
            torus_config, eos, initial_background, domain_box, dx, center);

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
        GRHD::fill_fixed_bg_torus_level_data(
            state, background_factory, eos, time, dx, center, torus,
            torus_config, atmosphere);
        GRHD::AtmosphereResetDiagnostics recovery_totals;
        recovery_totals.add(level_operator.recover_primitives(state, time));

        const std::string out_dir = default_output_dir();
        if (is_io_rank())
            ensure_directory(out_dir);
        synchronize_ranks();
        const std::string base = out_dir + "/leveldata_binary_pn_disk";

        std::ofstream diagnostics_csv;
        if (is_io_rank())
        {
            diagnostics_csv.open((base + "_diagnostics.csv").c_str());
            if (!diagnostics_csv)
                throw std::runtime_error("could not open diagnostics CSV");
            diagnostics_csv
                << "step,time,mass,disk_mass,tau,max_rho,max_pressure,"
                   "max_velocity,max_lorentz_factor,failed_recoveries,"
                   "floored_primitives,conserved_resets\n";
        }

        Diagnostics initial_diagnostics = compute_diagnostics(
            state, background_factory, time, dx, center, atmosphere,
            recovery_totals);
        if (is_io_rank())
            append_diagnostics(diagnostics_csv, 0, time,
                               initial_diagnostics);

        double next_output_time = config.output_interval;
        int steps = 0;
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
                    state, background_factory, time, dx, center, atmosphere,
                    recovery_totals);
                if (is_io_rank())
                    append_diagnostics(diagnostics_csv, steps, time,
                                       diagnostics);
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
                    state, background_factory, time, dx, center, atmosphere,
                    recovery_totals);
                if (is_io_rank())
                    append_diagnostics(diagnostics_csv, steps, time,
                                       diagnostics);
                next_output_time += config.output_interval;
            }
        }

        const Diagnostics final_diagnostics = compute_diagnostics(
            state, background_factory, time, dx, center, atmosphere,
            recovery_totals);
        if (is_io_rank())
        {
            diagnostics_csv.close();
            write_summary(base + "_summary.txt", config, torus, grids.size(),
                          mpi_size(), initial_diagnostics, final_diagnostics,
                          time, steps, dx);
        }

        write_slice_csv(final_slice_path(base), state, background_factory,
                        time, dx, center, config.num_z_cells / 2);
        synchronize_ranks();

        if (!std::isfinite(final_diagnostics.max_rho) ||
            final_diagnostics.max_rho <= config.rho_floor)
            throw std::runtime_error("disk density became invalid");
        if (final_diagnostics.disk_mass <
            0.25 * initial_diagnostics.disk_mass)
            throw std::runtime_error("disk lost too much mass");
        if (final_diagnostics.max_velocity >
            std::sqrt(config.max_velocity_squared) * 1.001)
            throw std::runtime_error("velocity cap was violated");

        if (is_io_rank())
        {
            std::cout << "GRHD LevelData binary PN disk passed...\n"
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
                      << "  wrote " << base << "_summary.txt" << std::endl;
        }
    }
    catch (const std::exception &error)
    {
        if (is_io_rank())
            std::cerr << "GRHD LevelData binary PN disk failed: "
                      << error.what() << std::endl;
        return finish(1);
    }

    return finish(0);
}
