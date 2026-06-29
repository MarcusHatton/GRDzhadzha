/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#include "ADMFixedBGVars.hpp"
#include "Coordinates.hpp"
#include "CircularBinaryPN.hpp"
#include "DimensionDefinitions.hpp"
#include "FixedBGMetric.hpp"
#include "FixedBGFArrayBoxFiniteVolume.hpp"
#include "IdealGasEOS.hpp"
#include "KerrSchild.hpp"
#include "StaticMetric.hpp"
#include "TensorAlgebra.hpp"
#include "UserVariables.hpp"
#include "Valencia.hpp"
#ifdef CH_MPI
#include "mpi.h"
#endif
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

namespace
{
bool check_close(const std::string &name, double value, double expected,
                 double tolerance)
{
    if (!std::isfinite(value) || !std::isfinite(expected))
    {
        std::cout << name << " is not finite: got " << value
                  << ", expected " << expected << std::endl;
        return false;
    }
    const double scale = std::max(1.0, std::abs(expected));
    const double error = std::abs(value - expected) / scale;
    if (error > tolerance)
    {
        std::cout << name << " mismatch: got " << value << ", expected "
                  << expected << ", relative error " << error << std::endl;
        return false;
    }
    return true;
}

bool check_finite(const std::string &name, double value)
{
    if (!std::isfinite(value))
    {
        std::cout << name << " is not finite: " << value << std::endl;
        return false;
    }
    return true;
}

bool check_geometry_inverse(const GRHD::CellGeometry &geometry)
{
    bool passed = true;
    FOR(i, j)
    {
        double value = 0.0;
        FOR(k)
        {
            value += geometry.spatial_metric_LL[i][k] *
                     geometry.spatial_metric_UU[k][j];
        }
        passed &= check_close("gamma inverse", value,
                              TensorAlgebra::delta(i, j), 1.0e-11);
    }
    return passed;
}

bool check_sources_are_finite(const GRHD::Conserved<double> &source)
{
    bool passed = true;
    passed &= check_finite("source D", source.D);
    passed &= check_finite("source tau", source.tau);
    FOR(i) { passed &= check_finite("source S", source.S_L[i]); }
    return passed;
}

GRHD::Primitive<double> make_farraybox_test_primitive(
    const IntVect &iv, const IdealGasEOS &eos)
{
    GRHD::Primitive<double> primitive;
    primitive.rho = 0.7 + 1.0e-3 * (iv[0] + 2 * iv[1] + 3 * iv[2]);
    primitive.eps = 0.12 + 1.0e-4 * (2 * iv[0] + iv[1] + iv[2]);
    primitive.pressure = eos.compute_pressure(primitive.rho, primitive.eps);
    primitive.velocity_U[0] = 1.0e-3 * (1.0 + iv[0]);
    primitive.velocity_U[1] = -7.5e-4 * (1.0 + iv[1]);
    primitive.velocity_U[2] = 5.0e-4 * (1.0 + iv[2]);
    return primitive;
}

void store_ccz4_background_vars(
    FArrayBox &state, const IntVect &iv,
    const ADMFixedBGVars::Vars<double> &metric_vars)
{
    const double chi = std::pow(
        TensorAlgebra::compute_determinant_sym(metric_vars.gamma),
        -1.0 / 3.0);

    Tensor<2, double> h_LL;
    Tensor<2, double> A_LL;
    FOR(i, j)
    {
        h_LL[i][j] = metric_vars.gamma[i][j] * chi;
        A_LL[i][j] = metric_vars.K_tensor[i][j] * chi -
                     metric_vars.K * h_LL[i][j] / 3.0;
    }

    state(iv, c_chi) = chi;
    state(iv, c_K) = metric_vars.K;
    state(iv, c_lapse) = metric_vars.lapse;
    state(iv, c_shift1) = metric_vars.shift[0];
    state(iv, c_shift2) = metric_vars.shift[1];
    state(iv, c_shift3) = metric_vars.shift[2];

    state(iv, c_h11) = h_LL[0][0];
    state(iv, c_h12) = h_LL[0][1];
    state(iv, c_h13) = h_LL[0][2];
    state(iv, c_h22) = h_LL[1][1];
    state(iv, c_h23) = h_LL[1][2];
    state(iv, c_h33) = h_LL[2][2];

    state(iv, c_A11) = A_LL[0][0];
    state(iv, c_A12) = A_LL[0][1];
    state(iv, c_A13) = A_LL[0][2];
    state(iv, c_A22) = A_LL[1][1];
    state(iv, c_A23) = A_LL[1][2];
    state(iv, c_A33) = A_LL[2][2];
}

template <class background_t>
void fill_farraybox_states_from_background(
    FArrayBox &fixed_state, FArrayBox &ccz4_state, const Box &box,
    const background_t &background, const IdealGasEOS &eos, double dx,
    const std::array<double, CH_SPACEDIM> &center)
{
    BoxIterator bit(box);
    for (bit.begin(); bit.ok(); ++bit)
    {
        const IntVect iv = bit();
        const Coordinates<double> coords(iv, dx, center);
        ADMFixedBGVars::Vars<double> metric_vars;
        background.compute_metric_background(metric_vars, coords);
        const auto geometry = GRHD::make_cell_geometry_from_adm(metric_vars);
        const auto primitive = make_farraybox_test_primitive(iv, eos);
        const auto conserved = GRHD::compute_conserved(
            primitive, eos, geometry.spatial_metric_LL);

        GRHD::store_primitive(fixed_state, iv, primitive);
        GRHD::store_conserved(fixed_state, iv, conserved);
        GRHD::store_primitive(ccz4_state, iv, primitive);
        GRHD::store_conserved(ccz4_state, iv, conserved);
        store_ccz4_background_vars(ccz4_state, iv, metric_vars);
    }
}

bool check_conserved_close(const std::string &name,
                           const GRHD::Conserved<double> &value,
                           const GRHD::Conserved<double> &expected,
                           double tolerance)
{
    bool passed = true;
    passed &= check_close(name + " D", value.D, expected.D, tolerance);
    passed &= check_close(name + " tau", value.tau, expected.tau, tolerance);
    FOR(i)
    {
        passed &= check_close(name + " S", value.S_L[i],
                              expected.S_L[i], tolerance);
    }
    return passed;
}

bool check_conserved_farraybox_close(const std::string &name,
                                     const FArrayBox &value,
                                     const FArrayBox &expected,
                                     const Box &box,
                                     double tolerance)
{
    bool passed = true;
    BoxIterator bit(box);
    for (bit.begin(); bit.ok(); ++bit)
    {
        passed &= check_conserved_close(
            name, GRHD::load_conserved(value, bit()),
            GRHD::load_conserved(expected, bit()), tolerance);
    }
    return passed;
}

template <class background_t>
bool check_fixed_background_farraybox_helpers(
    const background_t &background, const IdealGasEOS &eos, double dx,
    const std::array<double, CH_SPACEDIM> &center)
{
    bool passed = true;
    const IntVect small(D_DECL(20, 15, 10));
    const IntVect big(D_DECL(22, 17, 12));
    const Box interior_box(small, big);
    Box ghosted_box(interior_box);
    ghosted_box.grow(2);
    const ProblemDomain domain(ghosted_box);

    FArrayBox fixed_state(ghosted_box, NUM_VARS);
    FArrayBox ccz4_state(ghosted_box, NUM_VARS);
    FArrayBox fixed_rhs(interior_box, NUM_VARS);
    FArrayBox ccz4_rhs(interior_box, NUM_VARS);
    fixed_state.setVal(0.0);
    ccz4_state.setVal(0.0);
    fixed_rhs.setVal(0.0);
    ccz4_rhs.setVal(0.0);

    fill_farraybox_states_from_background(
        fixed_state, ccz4_state, ghosted_box, background, eos, dx, center);

    GRHD::AtmosphereOptions atmosphere;
    GRHD::RecoveryOptions recovery_options;
    recovery_options.max_iterations = 80;

    FOR(direction)
    {
        GRHD::compute_directional_flux_rhs_from_fixed_background(
            fixed_rhs, fixed_state, interior_box, ghosted_box, domain,
            background, eos, direction, 1.0 / dx, dx, center, atmosphere,
            1.5, true, recovery_options);
        GRHD::compute_directional_flux_rhs_from_ccz4_geometry(
            ccz4_rhs, ccz4_state, interior_box, ghosted_box, domain,
            eos, direction, 1.0 / dx, atmosphere, 1.5, true,
            recovery_options);
        passed &= check_conserved_farraybox_close(
            "fixed-background flux RHS", fixed_rhs, ccz4_rhs,
            interior_box, 1.0e-10);
    }

    FArrayBox source_rhs(interior_box, NUM_VARS);
    source_rhs.setVal(0.0);
    GRHD::add_static_metric_sources_from_fixed_background_to_conserved_rhs(
        source_rhs, fixed_state, interior_box, background, eos, dx, center,
        atmosphere, recovery_options);

    BoxIterator bit(interior_box);
    for (bit.begin(); bit.ok(); ++bit)
    {
        const IntVect iv = bit();
        const auto geometry = GRHD::load_fixed_background_geometry(
            background, iv, dx, center);
        const auto derivatives =
            GRHD::load_fixed_background_metric_derivatives(
                background, iv, dx, center);
        const auto conserved = GRHD::load_conserved(fixed_state, iv);
        const auto recovered = GRHD::recover_primitive(
            conserved, eos, geometry.spatial_metric_UU, recovery_options);
        if (!recovered.success)
        {
            std::cout << "fixed-background FArrayBox recovery failed at "
                      << iv << std::endl;
            passed = false;
            continue;
        }
        auto primitive = recovered.primitive;
        GRHD::enforce_primitive_floors(primitive, eos,
                                       geometry.spatial_metric_LL,
                                       atmosphere);
        const auto source = GRHD::compute_static_metric_source_terms(
            primitive, conserved, eos, geometry, derivatives);
        passed &= check_conserved_close(
            "fixed-background source RHS", GRHD::load_conserved(source_rhs, iv),
            source, 1.0e-11);
    }

    return passed;
}


DisjointBoxLayout make_single_box_layout(const Box &domain_box)
{
    Vector<Box> boxes(1);
    boxes[0] = domain_box;
    Vector<int> proc_map(1);
    proc_map[0] = 0;
    DisjointBoxLayout grids(boxes, proc_map, ProblemDomain(domain_box));
    grids.close();
    return grids;
}

template <class background_t>
void fill_leveldata_state_from_background(
    LevelData<FArrayBox> &state, const background_t &background,
    const IdealGasEOS &eos, double dx,
    const std::array<double, CH_SPACEDIM> &center)
{
    const DisjointBoxLayout &grids = state.disjointBoxLayout();
    DataIterator dit = grids.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        FArrayBox ccz4_scratch(state[dit].box(), NUM_VARS);
        state[dit].setVal(0.0);
        ccz4_scratch.setVal(0.0);
        fill_farraybox_states_from_background(
            state[dit], ccz4_scratch, state[dit].box(), background, eos,
            dx, center);
    }
}

bool check_leveldata_conserved_finite(const LevelData<FArrayBox> &state,
                                      const Box &domain_box)
{
    bool passed = true;
    const DisjointBoxLayout &grids = state.disjointBoxLayout();
    DataIterator dit = grids.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        BoxIterator bit(grids[dit] & domain_box);
        for (bit.begin(); bit.ok(); ++bit)
        {
            const auto conserved = GRHD::load_conserved(state[dit], bit());
            passed &= check_finite("LevelData D", conserved.D);
            passed &= check_finite("LevelData tau", conserved.tau);
            FOR(i)
            {
                passed &= check_finite("LevelData S", conserved.S_L[i]);
            }
        }
    }
    return passed;
}

bool check_leveldata_conserved_close(const std::string &name,
                                     const LevelData<FArrayBox> &value,
                                     const LevelData<FArrayBox> &expected,
                                     const Box &domain_box,
                                     double tolerance)
{
    bool passed = true;
    const DisjointBoxLayout &grids = value.disjointBoxLayout();
    DataIterator dit = grids.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        BoxIterator bit(grids[dit] & domain_box);
        for (bit.begin(); bit.ok(); ++bit)
        {
            passed &= check_conserved_close(
                name, GRHD::load_conserved(value[dit], bit()),
                GRHD::load_conserved(expected[dit], bit()), tolerance);
        }
    }
    return passed;
}

template <class background_t>
bool check_fixed_background_leveldata_operator(
    const background_t &background, const IdealGasEOS &eos, double dx,
    const std::array<double, CH_SPACEDIM> &center)
{
    bool passed = true;
    const IntVect small(D_DECL(20, 15, 10));
    const IntVect big(D_DECL(22, 17, 12));
    const Box domain_box(small, big);
    const ProblemDomain domain(domain_box);
    const DisjointBoxLayout grids = make_single_box_layout(domain_box);
    const IntVect ghosts = 2 * IntVect::Unit;

    LevelData<FArrayBox> state(grids, NUM_VARS, ghosts);
    LevelData<FArrayBox> state_for_direct_rhs(grids, NUM_VARS, ghosts);
    LevelData<FArrayBox> rhs(grids, NUM_VARS, IntVect::Zero);
    LevelData<FArrayBox> direct_rhs(grids, NUM_VARS, IntVect::Zero);
    fill_leveldata_state_from_background(state, background, eos, dx, center);
    state.copyTo(state_for_direct_rhs);

    Tensor<1, double> inverse_dx_U;
    FOR(dir) { inverse_dx_U[dir] = 1.0 / dx; }
    GRHD::AtmosphereOptions atmosphere;
    GRHD::RecoveryOptions recovery_options;
    recovery_options.max_iterations = 80;

    GRHD::FixedBGLevelDataFiniteVolumeOperator<IdealGasEOS, background_t>
        level_operator(eos, background, dx, center, atmosphere, 1.5, true,
                       recovery_options, true);

    const auto recovery_diagnostics = level_operator.recover_primitives(state);
    if (recovery_diagnostics.num_failed_recoveries != 0 ||
        recovery_diagnostics.num_conserved_resets != 0)
    {
        std::cout << "unexpected fixed-background LevelData recovery reset"
                  << std::endl;
        passed = false;
    }

    const double max_inverse_dt = level_operator.compute_max_inverse_dt(
        state, domain_box, inverse_dx_U);
    passed &= check_finite("fixed-background LevelData inverse dt",
                           max_inverse_dt);
    if (max_inverse_dt <= 0.0)
    {
        std::cout << "fixed-background LevelData inverse dt is nonpositive"
                  << std::endl;
        passed = false;
    }

    const double stable_dt = level_operator.compute_stable_dt(
        state, domain_box, inverse_dx_U, 0.2);
    passed &= check_finite("fixed-background LevelData stable dt", stable_dt);
    if (stable_dt <= 0.0)
    {
        std::cout << "fixed-background LevelData stable dt is nonpositive"
                  << std::endl;
        passed = false;
    }

    level_operator.compute_rhs(rhs, state, domain, inverse_dx_U);
    GRHD::compute_leveldata_flux_rhs_from_fixed_background(
        direct_rhs, state_for_direct_rhs, domain, background, eos,
        inverse_dx_U, dx, center, atmosphere, 1.5, true, true,
        recovery_options);
    passed &= check_leveldata_conserved_close(
        "fixed-background LevelData RHS", rhs, direct_rhs, domain_box,
        1.0e-12);
    passed &= check_leveldata_conserved_finite(rhs, domain_box);

    level_operator.update_conserved(state, rhs, 0.01 * stable_dt);
    passed &= check_leveldata_conserved_finite(state, domain_box);

    level_operator.advance_ssprk2(state, domain, inverse_dx_U,
                                  0.01 * stable_dt);
    passed &= check_leveldata_conserved_finite(state, domain_box);

    return passed;
}


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

CircularBinaryPNTimeFactory make_binary_pn_time_factory(
    double dx, const std::array<double, CH_SPACEDIM> &center)
{
    CircularBinaryPNTimeFactory factory;
    factory.dx = dx;
    factory.params.mass_1 = 0.55;
    factory.params.mass_2 = 0.45;
    factory.params.separation = 8.0;
    factory.params.phase = 0.25;
    factory.params.time = 0.4;
    factory.params.softening_radius = 0.8;
    factory.params.center = center;
    return factory;
}

template <class background_factory_t>
bool check_time_dependent_fixed_background_leveldata_operator(
    const background_factory_t &background_factory, const IdealGasEOS &eos,
    double dx, const std::array<double, CH_SPACEDIM> &center)
{
    bool passed = true;
    const IntVect small(D_DECL(20, 15, 10));
    const IntVect big(D_DECL(22, 17, 12));
    const Box domain_box(small, big);
    const ProblemDomain domain(domain_box);
    const DisjointBoxLayout grids = make_single_box_layout(domain_box);
    const IntVect ghosts = 2 * IntVect::Unit;
    const double time = 0.35;
    const double metric_time_derivative_step = 1.0e-4;

    LevelData<FArrayBox> state(grids, NUM_VARS, ghosts);
    LevelData<FArrayBox> state_for_direct_rhs(grids, NUM_VARS, ghosts);
    LevelData<FArrayBox> rhs(grids, NUM_VARS, IntVect::Zero);
    LevelData<FArrayBox> direct_rhs(grids, NUM_VARS, IntVect::Zero);

    const auto initial_background = background_factory(time);
    fill_leveldata_state_from_background(state, initial_background, eos, dx,
                                         center);
    state.copyTo(state_for_direct_rhs);

    Tensor<1, double> inverse_dx_U;
    FOR(dir) { inverse_dx_U[dir] = 1.0 / dx; }
    GRHD::AtmosphereOptions atmosphere;
    GRHD::RecoveryOptions recovery_options;
    recovery_options.max_iterations = 80;

    const IntVect sample_iv(D_DECL(21, 16, 11));
    const double d_log_sqrt_det_dt =
        GRHD::fixed_background_log_sqrt_det_time_derivative(
            background_factory, time, sample_iv, dx, center,
            metric_time_derivative_step);
    passed &= check_finite("binary PN d log sqrt det dt",
                           d_log_sqrt_det_dt);
    if (std::abs(d_log_sqrt_det_dt) <= 1.0e-14)
    {
        std::cout << "binary PN determinant time derivative is too small"
                  << std::endl;
        passed = false;
    }

    GRHD::TimeDependentFixedBGLevelDataFiniteVolumeOperator<
        IdealGasEOS, background_factory_t>
        level_operator(eos, background_factory, dx, center, atmosphere,
                       1.5, true, recovery_options, true, true,
                       metric_time_derivative_step);

    const auto recovery_diagnostics = level_operator.recover_primitives(
        state, time);
    if (recovery_diagnostics.num_failed_recoveries != 0 ||
        recovery_diagnostics.num_conserved_resets != 0)
    {
        std::cout << "unexpected time-dependent LevelData recovery reset"
                  << std::endl;
        passed = false;
    }

    const double stable_dt = level_operator.compute_stable_dt(
        state, domain_box, inverse_dx_U, 0.18, time);
    passed &= check_finite("time-dependent LevelData stable dt", stable_dt);
    if (stable_dt <= 0.0)
    {
        std::cout << "time-dependent LevelData stable dt is nonpositive"
                  << std::endl;
        passed = false;
    }

    level_operator.compute_rhs(rhs, state, domain, inverse_dx_U, time);
    GRHD::compute_leveldata_flux_rhs_from_fixed_background(
        direct_rhs, state_for_direct_rhs, domain, initial_background, eos,
        inverse_dx_U, dx, center, atmosphere, 1.5, true, true,
        recovery_options);
    GRHD::add_leveldata_metric_volume_time_sources_from_fixed_background(
        direct_rhs, state_for_direct_rhs, domain_box, background_factory,
        time, dx, center, metric_time_derivative_step);
    passed &= check_leveldata_conserved_close(
        "time-dependent fixed-background RHS", rhs, direct_rhs,
        domain_box, 1.0e-12);
    passed &= check_leveldata_conserved_finite(rhs, domain_box);

    level_operator.advance_ssprk2(state, domain, inverse_dx_U, time,
                                  0.01 * stable_dt);
    passed &= check_leveldata_conserved_finite(state, domain_box);
    return passed;
}

} // namespace


template void GRHD::compute_leveldata_flux_rhs_from_fixed_background<
    IdealGasEOS, KerrSchild>(
    LevelData<FArrayBox> &rhs, LevelData<FArrayBox> &state,
    const ProblemDomain &domain, const KerrSchild &background,
    const IdealGasEOS &eos, const Tensor<1, double> &inverse_dx_U,
    double dx, const std::array<double, CH_SPACEDIM> &center,
    const GRHD::AtmosphereOptions &atmosphere, double limiter_theta,
    bool use_reconstruction, bool use_static_metric_sources,
    const GRHD::RecoveryOptions &recovery_options);

template class GRHD::FixedBGLevelDataFiniteVolumeOperator<IdealGasEOS,
                                                          KerrSchild>;

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

    const double dx = 0.25;
    const std::array<double, CH_SPACEDIM> center = {0.0, 0.0, 0.0};

    KerrSchild::params_t bg_params;
    bg_params.mass = 1.0;
    bg_params.spin = 0.3;
    bg_params.center = center;
    KerrSchild background(bg_params, dx);

    const IntVect iv(D_DECL(24, 15, 10));
    const Coordinates<double> coords(iv, dx, center);

    ADMFixedBGVars::Vars<double> metric_vars;
    background.compute_metric_background(metric_vars, coords);

    const auto geometry =
        GRHD::make_cell_geometry_from_background(background, coords);
    const auto geometry_from_adm =
        GRHD::make_cell_geometry_from_adm(metric_vars);
    const auto derivatives =
        GRHD::make_static_metric_derivatives_from_background(background,
                                                             coords);
    const auto derivatives_from_adm =
        GRHD::make_static_metric_derivatives_from_adm(metric_vars);

    passed &= check_finite("lapse", geometry.lapse);
    passed &= check_geometry_inverse(geometry);
    passed &= check_close("background lapse", geometry.lapse,
                          geometry_from_adm.lapse, 0.0);

    double extrinsic_norm = 0.0;
    FOR(i, j)
    {
        passed &= check_close("extrinsic curvature",
                              geometry.extrinsic_curvature_LL[i][j],
                              metric_vars.K_tensor[i][j], 0.0);
        passed &= check_finite("extrinsic curvature",
                               geometry.extrinsic_curvature_LL[i][j]);
        extrinsic_norm = std::max(
            extrinsic_norm,
            std::abs(geometry.extrinsic_curvature_LL[i][j]));
    }
    if (extrinsic_norm <= 1.0e-14)
    {
        std::cout << "fixed-background test expected nonzero K_ij"
                  << std::endl;
        passed = false;
    }

    FOR(i)
    {
        passed &= check_close("background shift", geometry.shift_U[i],
                              geometry_from_adm.shift_U[i], 0.0);
        passed &= check_close("lapse derivative", derivatives.lapse[i],
                              derivatives_from_adm.lapse[i], 0.0);
        passed &= check_finite("log sqrt det derivative",
                               derivatives.log_sqrt_det_spatial_metric[i]);
        FOR(j)
        {
            passed &= check_close("shift derivative",
                                  derivatives.shift_U[i][j],
                                  derivatives_from_adm.shift_U[i][j], 0.0);
        }
    }

    IdealGasEOS eos(5.0 / 3.0);
    GRHD::Primitive<double> primitive;
    primitive.rho = 1.1;
    primitive.eps = 0.2;
    primitive.pressure = eos.compute_pressure(primitive.rho, primitive.eps);
    primitive.velocity_U[0] = 0.01;
    primitive.velocity_U[1] = -0.015;
    primitive.velocity_U[2] = 0.005;

    const double velocity_squared = GRHD::compute_velocity_squared(
        primitive.velocity_U, geometry.spatial_metric_LL);
    if (velocity_squared >= 1.0)
    {
        std::cout << "test primitive is superluminal in fixed background"
                  << std::endl;
        return finish(1);
    }

    const auto conserved =
        GRHD::compute_conserved(primitive, eos, geometry.spatial_metric_LL);
    const auto recovered =
        GRHD::recover_primitive(conserved, eos, geometry.spatial_metric_UU);
    if (!recovered.success)
    {
        std::cout << "primitive recovery failed with residual "
                  << recovered.residual << std::endl;
        return finish(1);
    }

    passed &= check_close("rho", recovered.primitive.rho, primitive.rho,
                          1.0e-10);
    passed &= check_close("eps", recovered.primitive.eps, primitive.eps,
                          1.0e-10);
    passed &= check_close("pressure", recovered.primitive.pressure,
                          primitive.pressure, 1.0e-10);
    FOR(i)
    {
        passed &= check_close("velocity", recovered.primitive.velocity_U[i],
                              primitive.velocity_U[i], 1.0e-10);
    }

    const auto source = GRHD::compute_static_metric_source_terms(
        primitive, conserved, eos, geometry, derivatives);
    passed &= check_sources_are_finite(source);
    passed &= check_fixed_background_farraybox_helpers(
        background, eos, dx, center);
    passed &= check_fixed_background_leveldata_operator(
        background, eos, dx, center);
    const auto binary_factory = make_binary_pn_time_factory(dx, center);
    passed &= check_time_dependent_fixed_background_leveldata_operator(
        binary_factory, eos, dx, center);


    auto geometry_without_k = geometry;
    FOR(i, j)
    {
        geometry_without_k.extrinsic_curvature_LL[i][j] = 0.0;
    }
    const auto source_without_k = GRHD::compute_static_metric_source_terms(
        primitive, conserved, eos, geometry_without_k, derivatives);
    const auto stress_UU = GRHD::compute_spatial_stress_UU(
        primitive, eos, geometry);
    double expected_tau_delta = 0.0;
    FOR(i, j)
    {
        expected_tau_delta += geometry.lapse * stress_UU[i][j] *
                              geometry.extrinsic_curvature_LL[i][j];
    }
    passed &= check_close("extrinsic tau source",
                          source.tau - source_without_k.tau,
                          expected_tau_delta, 1.0e-12);
    passed &= check_close("extrinsic D source",
                          source.D - source_without_k.D, 0.0, 1.0e-14);
    FOR(i)
    {
        passed &= check_close("extrinsic momentum source",
                              source.S_L[i] - source_without_k.S_L[i],
                              0.0, 1.0e-14);
    }

    if (!passed)
        return finish(1);

    std::cout << "GRHD fixed-background coupling test passed..."
              << std::endl;
    return finish(0);
}
