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
#include <cmath>
#include <iostream>
#include <string>

#ifdef CH_MPI
#include "mpi.h"
#endif

namespace
{
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

bool check_finite(const std::string &name, double value)
{
    if (!std::isfinite(value))
    {
        std::cout << name << " is not finite: " << value << std::endl;
        return false;
    }
    return true;
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

GRHD::Primitive<double> ring_primitive(double x, double y, double z,
                                       const IdealGasEOS &eos,
                                       const GRHD::CellGeometry &geometry,
                                       const GRHD::AtmosphereOptions &atmosphere)
{
    GRHD::Primitive<double> primitive =
        GRHD::make_atmosphere_primitive<double>(eos, atmosphere);

    const double radius = std::sqrt(x * x + y * y);
    const double ring_radius = 5.0;
    const double radial_width = 1.0;
    const double vertical_width = 0.75;
    const double gaussian = std::exp(
        -0.5 * (radius - ring_radius) * (radius - ring_radius) /
            (radial_width * radial_width) -
        0.5 * z * z / (vertical_width * vertical_width));

    primitive.rho += 2.0e-5 * gaussian;
    primitive.pressure = std::max(
        atmosphere.pressure_floor,
        2.0e-4 * std::pow(primitive.rho, eos.adiabatic_index()));
    primitive.eps = primitive.pressure /
                    ((eos.adiabatic_index() - 1.0) * primitive.rho);

    FOR(i) { primitive.velocity_U[i] = 0.0; }
    if (radius > 1.0e-12)
    {
        const double speed = 0.04 * gaussian;
        primitive.velocity_U[0] = -speed * y / radius;
        primitive.velocity_U[1] = speed * x / radius;
    }

    GRHD::enforce_primitive_floors(primitive, eos, geometry.spatial_metric_LL,
                                   atmosphere);
    return primitive;
}

void fill_ring_state(LevelData<FArrayBox> &state,
                     const BinaryPNFactory &background_factory,
                     const IdealGasEOS &eos, double time, double dx,
                     const std::array<double, CH_SPACEDIM> &center,
                     const GRHD::AtmosphereOptions &atmosphere)
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
            const auto geometry = GRHD::load_fixed_background_geometry(
                background, iv, dx, center);
            auto primitive = ring_primitive(coords.x, coords.y, coords.z,
                                            eos, geometry, atmosphere);
            GRHD::store_primitive(state[dit], iv, primitive);
            GRHD::store_conserved(
                state[dit], iv,
                GRHD::compute_conserved(primitive, eos,
                                        geometry.spatial_metric_LL));
        }
    }
}

struct ConservedDiagnostics
{
    double mass = 0.0;
    double tau = 0.0;
    double max_rho = 0.0;
    double max_speed_squared = 0.0;
};

ConservedDiagnostics compute_diagnostics(
    LevelData<FArrayBox> &state, const BinaryPNFactory &background_factory,
    const IdealGasEOS &eos, double time, double dx,
    const std::array<double, CH_SPACEDIM> &center,
    const GRHD::RecoveryOptions &recovery_options)
{
    ConservedDiagnostics diagnostics;
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
            diagnostics.mass += conserved.D;
            diagnostics.tau += conserved.tau;
            const auto geometry = GRHD::load_fixed_background_geometry(
                background, iv, dx, center);
            const auto recovered = GRHD::recover_primitive(
                conserved, eos, geometry.spatial_metric_UU,
                recovery_options);
            if (recovered.success)
            {
                diagnostics.max_rho = std::max(
                    diagnostics.max_rho, recovered.primitive.rho);
                diagnostics.max_speed_squared = std::max(
                    diagnostics.max_speed_squared,
                    GRHD::compute_velocity_squared(
                        recovered.primitive.velocity_U,
                        geometry.spatial_metric_LL));
            }
        }
    }
#ifdef CH_MPI
    double global_values[4] = {0.0, 0.0, 0.0, 0.0};
    double local_values[4] = {diagnostics.mass, diagnostics.tau,
                              diagnostics.max_rho,
                              diagnostics.max_speed_squared};
    MPI_Allreduce(local_values, global_values, 2, MPI_DOUBLE, MPI_SUM,
                  Chombo_MPI::comm);
    MPI_Allreduce(local_values + 2, global_values + 2, 2, MPI_DOUBLE,
                  MPI_MAX, Chombo_MPI::comm);
    diagnostics.mass = global_values[0];
    diagnostics.tau = global_values[1];
    diagnostics.max_rho = global_values[2];
    diagnostics.max_speed_squared = global_values[3];
#endif
    return diagnostics;
}


void poison_internal_ghosts(LevelData<FArrayBox> &state, const Box &domain_box)
{
    const DisjointBoxLayout &grids = state.disjointBoxLayout();
    DataIterator dit = grids.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        const Box &interior = grids[dit];
        BoxIterator bit(state[dit].box());
        for (bit.begin(); bit.ok(); ++bit)
        {
            const IntVect iv = bit();
            bool inside_domain = true;
            bool inside_interior = true;
            FOR(dir)
            {
                inside_domain = inside_domain &&
                                iv[dir] >= domain_box.smallEnd(dir) &&
                                iv[dir] <= domain_box.bigEnd(dir);
                inside_interior = inside_interior &&
                                  iv[dir] >= interior.smallEnd(dir) &&
                                  iv[dir] <= interior.bigEnd(dir);
            }
            if (!inside_domain || inside_interior)
                continue;

            for (int comp = 0; comp < NUM_VARS; ++comp)
                state[dit](iv, comp) = 1.0e30 + comp;
        }
    }
}

bool leveldata_rhs_matches(const LevelData<FArrayBox> &lhs,
                           const LevelData<FArrayBox> &rhs, double tolerance,
                           double &max_relative_error)
{
    bool local_ok = true;
    double local_max = 0.0;
    const DisjointBoxLayout &grids = lhs.disjointBoxLayout();
    DataIterator dit = grids.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        BoxIterator bit(grids[dit]);
        for (bit.begin(); bit.ok(); ++bit)
        {
            const IntVect iv = bit();
            for (int comp = 0; comp < NUM_VARS; ++comp)
            {
                const double a = lhs[dit](iv, comp);
                const double b = rhs[dit](iv, comp);
                const double scale = std::max(1.0, std::max(std::abs(a),
                                                            std::abs(b)));
                const double error = std::abs(a - b) / scale;
                local_max = std::max(local_max, error);
                if (!std::isfinite(error) || error > tolerance)
                    local_ok = false;
            }
        }
    }

#ifdef CH_MPI
    int local_value = local_ok ? 1 : 0;
    int global_value = 0;
    MPI_Allreduce(&local_value, &global_value, 1, MPI_INT, MPI_MIN,
                  Chombo_MPI::comm);
    MPI_Allreduce(&local_max, &max_relative_error, 1, MPI_DOUBLE, MPI_MAX,
                  Chombo_MPI::comm);
    return global_value == 1;
#else
    max_relative_error = local_max;
    return local_ok;
#endif
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
            local_ok = local_ok && std::isfinite(conserved.D) &&
                       std::isfinite(conserved.tau);
            FOR(i) { local_ok = local_ok && std::isfinite(conserved.S_L[i]); }
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

    bool passed = true;
    const int num_cells = 18;
    const int num_z_cells = 5;
    const double dx = 1.0;
    const Box domain_box(IntVect(D_DECL(0, 0, 0)),
                         IntVect(D_DECL(num_cells - 1, num_cells - 1,
                                        num_z_cells - 1)));
    const ProblemDomain domain(domain_box);
    const DisjointBoxLayout grids = make_x_split_layout(domain_box);
    LevelData<FArrayBox> state(grids, NUM_VARS, 2 * IntVect::Unit);

    const std::array<double, CH_SPACEDIM> center = {
        0.5 * num_cells * dx, 0.5 * num_cells * dx,
        0.5 * num_z_cells * dx};

    BinaryPNFactory background_factory;
    background_factory.dx = dx;
    background_factory.params.mass_1 = 0.5;
    background_factory.params.mass_2 = 0.5;
    background_factory.params.separation = 5.0;
    background_factory.params.phase = 0.1;
    background_factory.params.time = 0.0;
    background_factory.params.softening_radius = 1.0;
    background_factory.params.center = {{0.0, 0.0, 0.0}};

    IdealGasEOS eos(5.0 / 3.0);
    GRHD::AtmosphereOptions atmosphere;
    atmosphere.rho_floor = 1.0e-10;
    atmosphere.pressure_floor = 1.0e-12;
    atmosphere.max_velocity_squared = 0.8;
    GRHD::RecoveryOptions recovery_options;
    recovery_options.max_iterations = 80;
    recovery_options.pressure_floor = atmosphere.pressure_floor;
    recovery_options.max_velocity_squared = atmosphere.max_velocity_squared;

    Tensor<1, double> inverse_dx_U;
    FOR(dir) { inverse_dx_U[dir] = 1.0 / dx; }

    const double metric_time_derivative_step = 1.0e-3;
    GRHD::TimeDependentFixedBGLevelDataFiniteVolumeOperator<
        IdealGasEOS, BinaryPNFactory>
        level_operator(eos, background_factory, dx, center, atmosphere,
                       1.4, true, recovery_options, true, true,
                       metric_time_derivative_step);

    double time = 0.0;
    fill_ring_state(state, background_factory, eos, time, dx, center,
                    atmosphere);
    const auto initial_recovery = level_operator.recover_primitives(state,
                                                                    time);
    if (initial_recovery.num_failed_recoveries != 0 ||
        initial_recovery.num_conserved_resets != 0)
    {
        std::cout << "initial recovery reset unexpectedly" << std::endl;
        passed = false;
    }

    LevelData<FArrayBox> clean_rhs(grids, NUM_VARS, IntVect::Zero);
    LevelData<FArrayBox> exchanged_rhs(grids, NUM_VARS, IntVect::Zero);
    level_operator.compute_rhs(clean_rhs, state, domain, inverse_dx_U, time);
    poison_internal_ghosts(state, domain_box);
    level_operator.compute_rhs(exchanged_rhs, state, domain, inverse_dx_U,
                               time);
    double max_exchange_rhs_error = 0.0;
    if (!leveldata_rhs_matches(clean_rhs, exchanged_rhs, 1.0e-12,
                               max_exchange_rhs_error))
    {
        std::cout << "multi-box exchange RHS mismatch: "
                  << max_exchange_rhs_error << std::endl;
        passed = false;
    }

    const auto initial_diagnostics = compute_diagnostics(
        state, background_factory, eos, time, dx, center, recovery_options);
    passed &= check_finite("initial mass", initial_diagnostics.mass);
    passed &= check_finite("initial tau", initial_diagnostics.tau);
    if (initial_diagnostics.mass <= 0.0 ||
        initial_diagnostics.max_rho <= atmosphere.rho_floor)
    {
        std::cout << "initial ring diagnostics are invalid" << std::endl;
        passed = false;
    }

    for (int step = 0; step < 4; ++step)
    {
        const double stable_dt = level_operator.compute_stable_dt(
            state, domain_box, inverse_dx_U, 0.2, time);
        passed &= check_finite("stable dt", stable_dt);
        if (stable_dt <= 0.0)
        {
            std::cout << "stable dt is nonpositive" << std::endl;
            passed = false;
            break;
        }
        const double dt = std::min(0.05, 0.25 * stable_dt);
        level_operator.advance_ssprk2(state, domain, inverse_dx_U, time, dt);
        time += dt;
        if (!leveldata_state_is_finite(state))
        {
            std::cout << "non-finite state after step " << step << std::endl;
            passed = false;
            break;
        }
    }

    const auto final_diagnostics = compute_diagnostics(
        state, background_factory, eos, time, dx, center, recovery_options);
    passed &= check_finite("final mass", final_diagnostics.mass);
    passed &= check_finite("final tau", final_diagnostics.tau);
    passed &= check_finite("final max rho", final_diagnostics.max_rho);
    passed &= check_finite("final max speed squared",
                           final_diagnostics.max_speed_squared);
    if (final_diagnostics.mass <= 0.0 ||
        final_diagnostics.max_rho <= atmosphere.rho_floor ||
        final_diagnostics.max_speed_squared >= atmosphere.max_velocity_squared)
    {
        std::cout << "final ring diagnostics are invalid" << std::endl;
        passed = false;
    }

    if (!passed)
        return finish(1);

    std::cout << "GRHD LevelData binary PN smoke test passed...\n"
              << "  boxes: " << grids.size() << "\n"
              << "  exchange rhs max relative error: "
              << max_exchange_rhs_error << "\n"
              << "  final time: " << time << "\n"
              << "  initial mass: " << initial_diagnostics.mass << "\n"
              << "  final mass: " << final_diagnostics.mass << "\n"
              << "  max rho: " << final_diagnostics.max_rho << "\n"
              << "  max speed squared: "
              << final_diagnostics.max_speed_squared << std::endl;
    return finish(0);
}
