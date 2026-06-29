/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#include "CircularBinaryPN.hpp"
#include "Coordinates.hpp"
#include "FixedBGAMRLevelHooks.hpp"
#include "FixedBGFArrayBoxFiniteVolume.hpp"
#include "FixedBGLevelDataDriver.hpp"
#include "IdealGasEOS.hpp"
#include "KerrSchild.hpp"
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
struct CircularBinaryPNTimeFactory
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

bool check_close(const std::string &name, double value, double expected,
                 double tolerance)
{
    const double scale = std::max(1.0, std::max(std::abs(value),
                                                std::abs(expected)));
    const double error = std::abs(value - expected) / scale;
    if (!std::isfinite(error) || error > tolerance)
    {
        std::cout << name << " mismatch: got " << value << ", expected "
                  << expected << ", relative error " << error << std::endl;
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

GRHD::Primitive<double> make_primitive(const IntVect &iv,
                                       const IdealGasEOS &eos)
{
    GRHD::Primitive<double> primitive;
    primitive.rho = 0.2 + 5.0e-4 * (iv[0] + 2 * iv[1] + 3 * iv[2]);
    primitive.eps = 0.08 + 1.0e-4 * (iv[0] + iv[1] + iv[2]);
    primitive.pressure = eos.compute_pressure(primitive.rho, primitive.eps);
    primitive.velocity_U[0] = 8.0e-4 * (1.0 + iv[0]);
    primitive.velocity_U[1] = -6.0e-4 * (1.0 + iv[1]);
    primitive.velocity_U[2] = 4.0e-4 * (1.0 + iv[2]);
    return primitive;
}

template <class background_t>
void fill_state(LevelData<FArrayBox> &state, const background_t &background,
                const IdealGasEOS &eos, double dx,
                const std::array<double, CH_SPACEDIM> &center)
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
            const auto geometry = GRHD::load_fixed_background_geometry(
                background, iv, dx, center);
            auto primitive = make_primitive(iv, eos);
            GRHD::enforce_primitive_floors(primitive, eos,
                                           geometry.spatial_metric_LL);
            GRHD::store_primitive(state[dit], iv, primitive);
            GRHD::store_conserved(
                state[dit], iv,
                GRHD::compute_conserved(primitive, eos,
                                        geometry.spatial_metric_LL));
        }
    }
}

bool leveldata_conserved_close(const LevelData<FArrayBox> &lhs,
                               const LevelData<FArrayBox> &rhs,
                               const Box &domain_box, double tolerance)
{
    bool local_ok = true;
    double local_max = 0.0;
    const DisjointBoxLayout &grids = lhs.disjointBoxLayout();
    DataIterator dit = grids.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        BoxIterator bit(grids[dit] & domain_box);
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
    double global_max = 0.0;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX,
                  Chombo_MPI::comm);
    if (global_value != 1)
        std::cout << "LevelData conserved mismatch max rel error "
                  << global_max << std::endl;
    return global_value == 1;
#else
    if (!local_ok)
        std::cout << "LevelData conserved mismatch max rel error "
                  << local_max << std::endl;
    return local_ok;
#endif
}

bool leveldata_state_is_finite(const LevelData<FArrayBox> &state,
                               const Box &domain_box)
{
    bool local_ok = true;
    const DisjointBoxLayout &grids = state.disjointBoxLayout();
    DataIterator dit = grids.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        BoxIterator bit(grids[dit] & domain_box);
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

bool check_static_hook(const DisjointBoxLayout &grids,
                       const ProblemDomain &domain, const Box &domain_box,
                       const IdealGasEOS &eos, double dx,
                       const std::array<double, CH_SPACEDIM> &center)
{
    bool passed = true;
    KerrSchild::params_t params;
    params.mass = 1.0;
    params.spin = 0.0;
    params.center = {{0.0, 0.0, 0.0}};
    KerrSchild background(params, dx);

    const IntVect ghosts = 2 * IntVect::Unit;
    LevelData<FArrayBox> hook_state(grids, NUM_VARS, ghosts);
    LevelData<FArrayBox> direct_state(grids, NUM_VARS, ghosts);
    LevelData<FArrayBox> hook_rhs(grids, NUM_VARS, IntVect::Zero);
    LevelData<FArrayBox> direct_rhs(grids, NUM_VARS, IntVect::Zero);
    LevelData<FArrayBox> hook_advanced(grids, NUM_VARS, ghosts);
    LevelData<FArrayBox> direct_advanced(grids, NUM_VARS, ghosts);
    fill_state(hook_state, background, eos, dx, center);
    hook_state.copyTo(direct_state);
    hook_state.copyTo(hook_advanced);
    hook_state.copyTo(direct_advanced);

    Tensor<1, double> inverse_dx_U;
    FOR(dir) { inverse_dx_U[dir] = 1.0 / dx; }
    GRHD::AtmosphereOptions atmosphere;
    GRHD::RecoveryOptions recovery_options;
    recovery_options.max_iterations = 80;

    GRHD::FixedBGAMRLevelHooks<IdealGasEOS, KerrSchild>
        hooks(eos, background, dx, center, atmosphere, 1.4, true,
              recovery_options, true);
    GRHD::FixedBGLevelDataFiniteVolumeOperator<IdealGasEOS, KerrSchild>
        direct_operator(eos, background, dx, center, atmosphere, 1.4, true,
                        recovery_options, true);

    hooks.specific_eval_rhs(hook_state, hook_rhs, domain, inverse_dx_U);
    direct_operator.compute_rhs(direct_rhs, direct_state, domain,
                                inverse_dx_U);
    passed &= leveldata_conserved_close(hook_rhs, direct_rhs, domain_box,
                                        1.0e-12);

    const double hook_dt = hooks.compute_stable_dt(hook_state, domain,
                                                   inverse_dx_U, 0.2);
    const double direct_dt = direct_operator.compute_stable_dt(
        direct_state, domain_box, inverse_dx_U, 0.2);
    passed &= check_close("static hook dt", hook_dt, direct_dt, 1.0e-14);
    passed &= check_close("static coupled dt",
                          hooks.compute_coupled_dt(hook_state, domain,
                                                   inverse_dx_U, 0.2,
                                                   0.5 * hook_dt),
                          0.5 * hook_dt, 1.0e-14);

    const auto recovery = hooks.specific_update_ode(hook_state);
    if (recovery.num_failed_recoveries != 0 ||
        recovery.num_conserved_resets != 0)
    {
        std::cout << "static hook recovery reset unexpectedly" << std::endl;
        passed = false;
    }

    const double dt = std::min(0.02, 0.25 * hook_dt);
    hooks.advance_ssprk2(hook_advanced, domain, inverse_dx_U, dt);
    direct_operator.advance_ssprk2(direct_advanced, domain, inverse_dx_U, dt);
    passed &= leveldata_conserved_close(hook_advanced, direct_advanced,
                                        domain_box, 1.0e-12);
    passed &= leveldata_state_is_finite(hook_advanced, domain_box);
    return passed;
}

bool check_time_dependent_hook(
    const DisjointBoxLayout &grids, const ProblemDomain &domain,
    const Box &domain_box, const IdealGasEOS &eos, double dx,
    const std::array<double, CH_SPACEDIM> &center)
{
    bool passed = true;
    CircularBinaryPNTimeFactory factory;
    factory.dx = dx;
    factory.params.mass_1 = 0.5;
    factory.params.mass_2 = 0.5;
    factory.params.separation = 5.0;
    factory.params.phase = 0.2;
    factory.params.time = 0.0;
    factory.params.softening_radius = 1.0;
    factory.params.center = {{0.0, 0.0, 0.0}};

    const double time = 0.3;
    const auto background = factory(time);
    const IntVect ghosts = 2 * IntVect::Unit;
    LevelData<FArrayBox> hook_state(grids, NUM_VARS, ghosts);
    LevelData<FArrayBox> direct_state(grids, NUM_VARS, ghosts);
    LevelData<FArrayBox> hook_rhs(grids, NUM_VARS, IntVect::Zero);
    LevelData<FArrayBox> direct_rhs(grids, NUM_VARS, IntVect::Zero);
    LevelData<FArrayBox> hook_advanced(grids, NUM_VARS, ghosts);
    LevelData<FArrayBox> direct_advanced(grids, NUM_VARS, ghosts);
    fill_state(hook_state, background, eos, dx, center);
    hook_state.copyTo(direct_state);
    hook_state.copyTo(hook_advanced);
    hook_state.copyTo(direct_advanced);

    Tensor<1, double> inverse_dx_U;
    FOR(dir) { inverse_dx_U[dir] = 1.0 / dx; }
    GRHD::AtmosphereOptions atmosphere;
    GRHD::RecoveryOptions recovery_options;
    recovery_options.max_iterations = 80;

    const double metric_time_derivative_step = 1.0e-3;
    GRHD::TimeDependentFixedBGAMRLevelHooks<IdealGasEOS,
                                            CircularBinaryPNTimeFactory>
        hooks(eos, factory, dx, center, atmosphere, 1.4, true,
              recovery_options, true, true, metric_time_derivative_step);
    GRHD::TimeDependentFixedBGLevelDataFiniteVolumeOperator<
        IdealGasEOS, CircularBinaryPNTimeFactory>
        direct_operator(eos, factory, dx, center, atmosphere, 1.4, true,
                        recovery_options, true, true,
                        metric_time_derivative_step);

    hooks.specific_eval_rhs(hook_state, hook_rhs, domain, inverse_dx_U, time);
    direct_operator.compute_rhs(direct_rhs, direct_state, domain,
                                inverse_dx_U, time);
    passed &= leveldata_conserved_close(hook_rhs, direct_rhs, domain_box,
                                        1.0e-12);

    const double hook_dt = hooks.compute_stable_dt(hook_state, domain,
                                                   inverse_dx_U, 0.2, time);
    const double direct_dt = direct_operator.compute_stable_dt(
        direct_state, domain_box, inverse_dx_U, 0.2, time);
    passed &= check_close("time-dependent hook dt", hook_dt, direct_dt,
                          1.0e-14);
    passed &= check_close("time-dependent coupled dt",
                          hooks.compute_coupled_dt(hook_state, domain,
                                                   inverse_dx_U, 0.2,
                                                   0.5 * hook_dt, time),
                          0.5 * hook_dt, 1.0e-14);

    const auto recovery = hooks.specific_update_ode(hook_state, time);
    if (recovery.num_failed_recoveries != 0 ||
        recovery.num_conserved_resets != 0)
    {
        std::cout << "time-dependent hook recovery reset unexpectedly"
                  << std::endl;
        passed = false;
    }

    const double dt = std::min(0.02, 0.25 * hook_dt);
    hooks.advance_ssprk2(hook_advanced, domain, inverse_dx_U, time, dt);
    direct_operator.advance_ssprk2(direct_advanced, domain, inverse_dx_U,
                                   time, dt);
    passed &= leveldata_conserved_close(hook_advanced, direct_advanced,
                                        domain_box, 1.0e-12);
    passed &= leveldata_state_is_finite(hook_advanced, domain_box);
    return passed;
}

bool check_static_driver(const DisjointBoxLayout &grids,
                         const ProblemDomain &domain,
                         const Box &domain_box, const IdealGasEOS &eos,
                         double dx,
                         const std::array<double, CH_SPACEDIM> &center)
{
    bool passed = true;
    KerrSchild::params_t params;
    params.mass = 1.0;
    params.spin = 0.0;
    params.center = {{0.0, 0.0, 0.0}};
    KerrSchild background(params, dx);

    const IntVect ghosts = 2 * IntVect::Unit;
    LevelData<FArrayBox> driver_state(grids, NUM_VARS, ghosts);
    LevelData<FArrayBox> direct_state(grids, NUM_VARS, ghosts);
    LevelData<FArrayBox> driver_rhs(grids, NUM_VARS, IntVect::Zero);
    LevelData<FArrayBox> direct_rhs(grids, NUM_VARS, IntVect::Zero);
    LevelData<FArrayBox> driver_advanced(grids, NUM_VARS, ghosts);
    LevelData<FArrayBox> direct_advanced(grids, NUM_VARS, ghosts);
    fill_state(driver_state, background, eos, dx, center);
    driver_state.copyTo(direct_state);
    driver_state.copyTo(driver_advanced);
    driver_state.copyTo(direct_advanced);

    Tensor<1, double> inverse_dx_U;
    FOR(dir) { inverse_dx_U[dir] = 1.0 / dx; }
    GRHD::AtmosphereOptions atmosphere;
    GRHD::RecoveryOptions recovery_options;
    recovery_options.max_iterations = 80;

    GRHD::FixedBGLevelDataDriverView<IdealGasEOS, KerrSchild>
        driver(driver_state, domain, dx, center, eos, background, atmosphere,
               1.4, true, recovery_options, true);
    GRHD::FixedBGLevelDataFiniteVolumeOperator<IdealGasEOS, KerrSchild>
        direct_operator(eos, background, dx, center, atmosphere, 1.4, true,
                        recovery_options, true);

    driver.compute_rhs(driver_rhs);
    direct_operator.compute_rhs(direct_rhs, direct_state, domain,
                                inverse_dx_U);
    passed &= leveldata_conserved_close(driver_rhs, direct_rhs, domain_box,
                                        1.0e-12);

    const double driver_dt = driver.compute_stable_dt(0.2);
    const double direct_dt = direct_operator.compute_stable_dt(
        direct_state, domain_box, inverse_dx_U, 0.2);
    passed &= check_close("static driver dt", driver_dt, direct_dt,
                          1.0e-14);
    passed &= check_close("static driver coupled dt",
                          driver.compute_coupled_dt(0.2,
                                                    0.5 * driver_dt),
                          0.5 * driver_dt, 1.0e-14);

    const auto recovery = driver.recover_primitives();
    if (recovery.num_failed_recoveries != 0 ||
        recovery.num_conserved_resets != 0)
    {
        std::cout << "static driver recovery reset unexpectedly"
                  << std::endl;
        passed = false;
    }

    const double dt = std::min(0.02, 0.25 * driver_dt);
    LevelData<FArrayBox> driver_update_state(grids, NUM_VARS, ghosts);
    LevelData<FArrayBox> direct_update_state(grids, NUM_VARS, ghosts);
    driver_state.copyTo(driver_update_state);
    driver_state.copyTo(direct_update_state);
    GRHD::FixedBGLevelDataDriverView<IdealGasEOS, KerrSchild>
        update_driver(driver_update_state, domain, dx, center, eos,
                      background, atmosphere, 1.4, true, recovery_options,
                      true);
    update_driver.update_conserved(driver_rhs, dt);
    direct_operator.update_conserved(direct_update_state, direct_rhs, dt);
    passed &= leveldata_conserved_close(driver_update_state,
                                        direct_update_state, domain_box,
                                        1.0e-12);
    passed &= check_close("static driver update time", update_driver.time(),
                          dt, 1.0e-14);
    if (update_driver.step() != 1)
    {
        std::cout << "static driver update step mismatch" << std::endl;
        passed = false;
    }

    GRHD::FixedBGLevelDataDriverView<IdealGasEOS, KerrSchild>
        advance_driver(driver_advanced, domain, dx, center, eos, background,
                       atmosphere, 1.4, true, recovery_options, true);
    advance_driver.advance_ssprk2(dt);
    direct_operator.advance_ssprk2(direct_advanced, domain, inverse_dx_U, dt);
    passed &= leveldata_conserved_close(driver_advanced, direct_advanced,
                                        domain_box, 1.0e-12);
    passed &= leveldata_state_is_finite(driver_advanced, domain_box);
    passed &= check_close("static driver advance time", advance_driver.time(),
                          dt, 1.0e-14);
    if (advance_driver.step() != 1)
    {
        std::cout << "static driver advance step mismatch" << std::endl;
        passed = false;
    }
    return passed;
}

bool check_time_dependent_driver(
    const DisjointBoxLayout &grids, const ProblemDomain &domain,
    const Box &domain_box, const IdealGasEOS &eos, double dx,
    const std::array<double, CH_SPACEDIM> &center)
{
    bool passed = true;
    CircularBinaryPNTimeFactory factory;
    factory.dx = dx;
    factory.params.mass_1 = 0.5;
    factory.params.mass_2 = 0.5;
    factory.params.separation = 5.0;
    factory.params.phase = 0.2;
    factory.params.time = 0.0;
    factory.params.softening_radius = 1.0;
    factory.params.center = {{0.0, 0.0, 0.0}};

    const double time = 0.3;
    const auto background = factory(time);
    const IntVect ghosts = 2 * IntVect::Unit;
    LevelData<FArrayBox> driver_state(grids, NUM_VARS, ghosts);
    LevelData<FArrayBox> direct_state(grids, NUM_VARS, ghosts);
    LevelData<FArrayBox> driver_rhs(grids, NUM_VARS, IntVect::Zero);
    LevelData<FArrayBox> direct_rhs(grids, NUM_VARS, IntVect::Zero);
    LevelData<FArrayBox> driver_advanced(grids, NUM_VARS, ghosts);
    LevelData<FArrayBox> direct_advanced(grids, NUM_VARS, ghosts);
    fill_state(driver_state, background, eos, dx, center);
    driver_state.copyTo(direct_state);
    driver_state.copyTo(driver_advanced);
    driver_state.copyTo(direct_advanced);

    Tensor<1, double> inverse_dx_U;
    FOR(dir) { inverse_dx_U[dir] = 1.0 / dx; }
    GRHD::AtmosphereOptions atmosphere;
    GRHD::RecoveryOptions recovery_options;
    recovery_options.max_iterations = 80;

    const double metric_time_derivative_step = 1.0e-3;
    GRHD::TimeDependentFixedBGLevelDataDriverView<
        IdealGasEOS, CircularBinaryPNTimeFactory>
        driver(driver_state, domain, dx, center, eos, factory, atmosphere,
               1.4, true, recovery_options, true, true,
               metric_time_derivative_step);
    driver.set_time(time);
    GRHD::TimeDependentFixedBGLevelDataFiniteVolumeOperator<
        IdealGasEOS, CircularBinaryPNTimeFactory>
        direct_operator(eos, factory, dx, center, atmosphere, 1.4, true,
                        recovery_options, true, true,
                        metric_time_derivative_step);

    driver.compute_rhs(driver_rhs);
    direct_operator.compute_rhs(direct_rhs, direct_state, domain,
                                inverse_dx_U, time);
    passed &= leveldata_conserved_close(driver_rhs, direct_rhs, domain_box,
                                        1.0e-12);

    const double driver_dt = driver.compute_stable_dt(0.2);
    const double direct_dt = direct_operator.compute_stable_dt(
        direct_state, domain_box, inverse_dx_U, 0.2, time);
    passed &= check_close("time-dependent driver dt", driver_dt, direct_dt,
                          1.0e-14);
    passed &= check_close("time-dependent driver coupled dt",
                          driver.compute_coupled_dt(0.2,
                                                    0.5 * driver_dt),
                          0.5 * driver_dt, 1.0e-14);

    const auto recovery = driver.recover_primitives();
    if (recovery.num_failed_recoveries != 0 ||
        recovery.num_conserved_resets != 0)
    {
        std::cout << "time-dependent driver recovery reset unexpectedly"
                  << std::endl;
        passed = false;
    }

    const double dt = std::min(0.02, 0.25 * driver_dt);
    LevelData<FArrayBox> driver_update_state(grids, NUM_VARS, ghosts);
    LevelData<FArrayBox> direct_update_state(grids, NUM_VARS, ghosts);
    driver_state.copyTo(driver_update_state);
    driver_state.copyTo(direct_update_state);
    GRHD::TimeDependentFixedBGLevelDataDriverView<
        IdealGasEOS, CircularBinaryPNTimeFactory>
        update_driver(driver_update_state, domain, dx, center, eos, factory,
                      atmosphere, 1.4, true, recovery_options, true, true,
                      metric_time_derivative_step);
    update_driver.set_time(time);
    update_driver.update_conserved(driver_rhs, dt);
    direct_operator.update_conserved(direct_update_state, direct_rhs, dt,
                                     time + dt);
    passed &= leveldata_conserved_close(driver_update_state,
                                        direct_update_state, domain_box,
                                        1.0e-12);
    passed &= check_close("time-dependent driver update time",
                          update_driver.time(), time + dt, 1.0e-14);
    if (update_driver.step() != 1)
    {
        std::cout << "time-dependent driver update step mismatch"
                  << std::endl;
        passed = false;
    }

    GRHD::TimeDependentFixedBGLevelDataDriverView<
        IdealGasEOS, CircularBinaryPNTimeFactory>
        advance_driver(driver_advanced, domain, dx, center, eos, factory,
                       atmosphere, 1.4, true, recovery_options, true, true,
                       metric_time_derivative_step);
    advance_driver.set_time(time);
    advance_driver.advance_ssprk2(dt);
    direct_operator.advance_ssprk2(direct_advanced, domain, inverse_dx_U,
                                   time, dt);
    passed &= leveldata_conserved_close(driver_advanced, direct_advanced,
                                        domain_box, 1.0e-12);
    passed &= leveldata_state_is_finite(driver_advanced, domain_box);
    passed &= check_close("time-dependent driver advance time",
                          advance_driver.time(), time + dt, 1.0e-14);
    if (advance_driver.step() != 1)
    {
        std::cout << "time-dependent driver advance step mismatch"
                  << std::endl;
        passed = false;
    }
    return passed;
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
    const double dx = 1.0;
    const Box domain_box(IntVect(D_DECL(0, 0, 0)),
                         IntVect(D_DECL(11, 7, 5)));
    const ProblemDomain domain(domain_box);
    const DisjointBoxLayout grids = make_x_split_layout(domain_box);
    const std::array<double, CH_SPACEDIM> center = {6.0, 4.0, 3.0};
    const IdealGasEOS eos(5.0 / 3.0);

    passed &= check_static_hook(grids, domain, domain_box, eos, dx, center);
    passed &= check_time_dependent_hook(grids, domain, domain_box, eos, dx,
                                        center);
    passed &= check_static_driver(grids, domain, domain_box, eos, dx, center);
    passed &= check_time_dependent_driver(grids, domain, domain_box, eos, dx,
                                          center);

    if (!passed)
        return finish(1);

    std::cout << "GRHD fixed-background AMR hook test passed..." << std::endl;
    return finish(0);
}
