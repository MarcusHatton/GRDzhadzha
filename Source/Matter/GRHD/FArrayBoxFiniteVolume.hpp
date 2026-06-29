/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef GRHD_FARRAYBOX_FINITEVOLUME_HPP_
#define GRHD_FARRAYBOX_FINITEVOLUME_HPP_

#include "Atmosphere.hpp"
#include "BoxIterator.H"
#include "DisjointBoxLayout.H"
#include "FArrayBox.H"
#include "LevelData.H"
#include "ProblemDomain.H"
#include "FiniteVolume.hpp"
#include "Reconstruction.hpp"
#include "SPMD.H"
#include "StaticMetric.hpp"
#include "TensorAlgebra.hpp"
#include "UserVariables.hpp"
#include "Valencia.hpp"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include "UsingNamespace.H"

namespace GRHD
{
inline Conserved<double> load_conserved(const FArrayBox &fab,
                                        const IntVect &iv)
{
    Conserved<double> conserved;
    conserved.D = fab(iv, c_GRHD_D);
    conserved.S_L[0] = fab(iv, c_GRHD_S1);
    conserved.S_L[1] = fab(iv, c_GRHD_S2);
    conserved.S_L[2] = fab(iv, c_GRHD_S3);
    conserved.tau = fab(iv, c_GRHD_tau);
    return conserved;
}

inline void store_conserved(FArrayBox &fab, const IntVect &iv,
                            const Conserved<double> &conserved)
{
    fab(iv, c_GRHD_D) = conserved.D;
    fab(iv, c_GRHD_S1) = conserved.S_L[0];
    fab(iv, c_GRHD_S2) = conserved.S_L[1];
    fab(iv, c_GRHD_S3) = conserved.S_L[2];
    fab(iv, c_GRHD_tau) = conserved.tau;
}

inline Primitive<double> load_primitive(const FArrayBox &fab,
                                        const IntVect &iv)
{
    Primitive<double> primitive;
    primitive.rho = fab(iv, c_GRHD_rho);
    primitive.eps = fab(iv, c_GRHD_eps);
    primitive.pressure = fab(iv, c_GRHD_pressure);
    primitive.velocity_U[0] = fab(iv, c_GRHD_v1);
    primitive.velocity_U[1] = fab(iv, c_GRHD_v2);
    primitive.velocity_U[2] = fab(iv, c_GRHD_v3);
    return primitive;
}

inline void store_primitive(FArrayBox &fab, const IntVect &iv,
                            const Primitive<double> &primitive)
{
    fab(iv, c_GRHD_rho) = primitive.rho;
    fab(iv, c_GRHD_eps) = primitive.eps;
    fab(iv, c_GRHD_pressure) = primitive.pressure;
    fab(iv, c_GRHD_v1) = primitive.velocity_U[0];
    fab(iv, c_GRHD_v2) = primitive.velocity_U[1];
    fab(iv, c_GRHD_v3) = primitive.velocity_U[2];
}


struct AtmosphereResetDiagnostics
{
    long long num_cells = 0;
    long long num_failed_recoveries = 0;
    long long num_floored_primitives = 0;
    long long num_conserved_resets = 0;

    void add(const AtmosphereResetDiagnostics &other)
    {
        num_cells += other.num_cells;
        num_failed_recoveries += other.num_failed_recoveries;
        num_floored_primitives += other.num_floored_primitives;
        num_conserved_resets += other.num_conserved_resets;
    }
};

inline AtmosphereResetDiagnostics reduce_atmosphere_reset_diagnostics(
    const AtmosphereResetDiagnostics &local_diagnostics)
{
#ifndef CH_MPI
    return local_diagnostics;
#else
    long long local_counts[4] = {
        local_diagnostics.num_cells,
        local_diagnostics.num_failed_recoveries,
        local_diagnostics.num_floored_primitives,
        local_diagnostics.num_conserved_resets};
    long long global_counts[4] = {0, 0, 0, 0};
    MPI_Allreduce(local_counts, global_counts, 4, MPI_LONG_LONG, MPI_SUM,
                  Chombo_MPI::comm);

    AtmosphereResetDiagnostics global_diagnostics;
    global_diagnostics.num_cells = global_counts[0];
    global_diagnostics.num_failed_recoveries = global_counts[1];
    global_diagnostics.num_floored_primitives = global_counts[2];
    global_diagnostics.num_conserved_resets = global_counts[3];
    return global_diagnostics;
#endif
}

inline bool is_atmosphere_primitive(
    const Primitive<double> &primitive,
    const AtmosphereOptions &atmosphere = AtmosphereOptions(),
    double threshold_factor = 1.0 + 1.0e-12)
{
    const double factor = std::max(1.0, threshold_factor);
    return primitive.rho <= factor * atmosphere.rho_floor &&
           primitive.pressure <= factor * atmosphere.pressure_floor;
}

inline bool is_metric_source_atmosphere_primitive(
    const Primitive<double> &primitive,
    const AtmosphereOptions &atmosphere = AtmosphereOptions())
{
    return is_atmosphere_primitive(
        primitive, atmosphere, atmosphere.metric_source_atmosphere_factor);
}

inline Tensor<2, double> load_conformal_spatial_metric_LL(
    const FArrayBox &fab, const IntVect &iv)
{
    Tensor<2, double> h_LL = 0.0;
    h_LL[0][0] = fab(iv, c_h11);
    h_LL[0][1] = fab(iv, c_h12);
    h_LL[1][0] = h_LL[0][1];
    h_LL[0][2] = fab(iv, c_h13);
    h_LL[2][0] = h_LL[0][2];
    h_LL[1][1] = fab(iv, c_h22);
    h_LL[1][2] = fab(iv, c_h23);
    h_LL[2][1] = h_LL[1][2];
    h_LL[2][2] = fab(iv, c_h33);
    return h_LL;
}

inline Tensor<2, double> load_conformal_extrinsic_curvature_LL(
    const FArrayBox &fab, const IntVect &iv)
{
    Tensor<2, double> A_LL = 0.0;
    A_LL[0][0] = fab(iv, c_A11);
    A_LL[0][1] = fab(iv, c_A12);
    A_LL[1][0] = A_LL[0][1];
    A_LL[0][2] = fab(iv, c_A13);
    A_LL[2][0] = A_LL[0][2];
    A_LL[1][1] = fab(iv, c_A22);
    A_LL[1][2] = fab(iv, c_A23);
    A_LL[2][1] = A_LL[1][2];
    A_LL[2][2] = fab(iv, c_A33);
    return A_LL;
}

inline CellGeometry load_ccz4_geometry(const FArrayBox &fab,
                                       const IntVect &iv)
{
    const double chi = fab(iv, c_chi);
    const Tensor<2, double> h_LL = load_conformal_spatial_metric_LL(fab, iv);
    const Tensor<2, double> h_UU = TensorAlgebra::compute_inverse_sym(h_LL);
    const Tensor<2, double> A_LL =
        load_conformal_extrinsic_curvature_LL(fab, iv);
    const double trace_K = fab(iv, c_K);

    CellGeometry geometry;
    FOR(i, j)
    {
        geometry.spatial_metric_LL[i][j] = h_LL[i][j] / chi;
        geometry.spatial_metric_UU[i][j] = chi * h_UU[i][j];
        geometry.extrinsic_curvature_LL[i][j] =
            A_LL[i][j] / chi +
            geometry.spatial_metric_LL[i][j] * trace_K / 3.0;
    }
    geometry.lapse = fab(iv, c_lapse);
    geometry.shift_U[0] = fab(iv, c_shift1);
    geometry.shift_U[1] = fab(iv, c_shift2);
    geometry.shift_U[2] = fab(iv, c_shift3);
    return geometry;
}

inline CellGeometry average_geometry(const CellGeometry &left,
                                     const CellGeometry &right)
{
    CellGeometry geometry;
    FOR(i, j)
    {
        geometry.spatial_metric_LL[i][j] =
            0.5 * (left.spatial_metric_LL[i][j] +
                   right.spatial_metric_LL[i][j]);
        geometry.spatial_metric_UU[i][j] =
            0.5 * (left.spatial_metric_UU[i][j] +
                   right.spatial_metric_UU[i][j]);
        geometry.extrinsic_curvature_LL[i][j] =
            0.5 * (left.extrinsic_curvature_LL[i][j] +
                   right.extrinsic_curvature_LL[i][j]);
    }
    geometry.lapse = 0.5 * (left.lapse + right.lapse);
    FOR(i)
    {
        geometry.shift_U[i] = 0.5 * (left.shift_U[i] + right.shift_U[i]);
    }
    return geometry;
}


inline IntVect direction_offset(int direction);

inline double centered_lapse_derivative(const FArrayBox &state,
                                        const IntVect &iv,
                                        int direction,
                                        double inverse_dx)
{
    const IntVect offset = direction_offset(direction);
    return 0.5 * inverse_dx *
           (state(iv + offset, c_lapse) - state(iv - offset, c_lapse));
}

inline double centered_shift_derivative(const FArrayBox &state,
                                        const IntVect &iv, int shift_dir,
                                        int derivative_dir,
                                        double inverse_dx)
{
    const IntVect offset = direction_offset(derivative_dir);
    return 0.5 * inverse_dx *
           (state(iv + offset, c_shift1 + shift_dir) -
            state(iv - offset, c_shift1 + shift_dir));
}

inline Tensor<2, double> centered_spatial_metric_derivative(
    const FArrayBox &state, const IntVect &iv, int direction,
    double inverse_dx)
{
    const IntVect offset = direction_offset(direction);
    const auto plus = load_ccz4_geometry(state, iv + offset);
    const auto minus = load_ccz4_geometry(state, iv - offset);

    Tensor<2, double> derivative = 0.0;
    FOR(i, j)
    {
        derivative[i][j] =
            0.5 * inverse_dx *
            (plus.spatial_metric_LL[i][j] - minus.spatial_metric_LL[i][j]);
    }
    return derivative;
}

inline double centered_log_sqrt_det_derivative(const FArrayBox &state,
                                               const IntVect &iv,
                                               int direction,
                                               double inverse_dx)
{
    const IntVect offset = direction_offset(direction);
    return 0.5 * inverse_dx *
           (spatial_metric_log_sqrt_det(
                load_ccz4_geometry(state, iv + offset)) -
            spatial_metric_log_sqrt_det(
                load_ccz4_geometry(state, iv - offset)));
}

struct MetricSourceDerivativeStencil
{
    IntVect lower_iv;
    IntVect upper_iv;
    double coefficient = 0.0;
    bool valid = false;
};

inline bool cell_has_metric_source_matter(
    const FArrayBox &state, const IntVect &iv,
    const AtmosphereOptions &atmosphere = AtmosphereOptions())
{
    return !is_metric_source_atmosphere_primitive(load_primitive(state, iv),
                                                  atmosphere);
}

inline MetricSourceDerivativeStencil make_metric_source_derivative_stencil(
    const FArrayBox &state, const IntVect &iv, int direction,
    double inverse_dx,
    const AtmosphereOptions &atmosphere = AtmosphereOptions())
{
    const IntVect offset = direction_offset(direction);
    const bool lower_active =
        cell_has_metric_source_matter(state, iv - offset, atmosphere);
    const bool upper_active =
        cell_has_metric_source_matter(state, iv + offset, atmosphere);

    MetricSourceDerivativeStencil stencil;
    if (lower_active && upper_active)
    {
        stencil.lower_iv = iv - offset;
        stencil.upper_iv = iv + offset;
        stencil.coefficient = 0.5 * inverse_dx;
        stencil.valid = true;
    }
    else if (upper_active)
    {
        stencil.lower_iv = iv;
        stencil.upper_iv = iv + offset;
        stencil.coefficient = inverse_dx;
        stencil.valid = true;
    }
    else if (lower_active)
    {
        stencil.lower_iv = iv - offset;
        stencil.upper_iv = iv;
        stencil.coefficient = inverse_dx;
        stencil.valid = true;
    }
    return stencil;
}

inline double source_lapse_derivative(
    const FArrayBox &state, const MetricSourceDerivativeStencil &stencil)
{
    return stencil.coefficient *
           (state(stencil.upper_iv, c_lapse) -
            state(stencil.lower_iv, c_lapse));
}

inline double source_shift_derivative(
    const FArrayBox &state, const MetricSourceDerivativeStencil &stencil,
    int shift_dir)
{
    return stencil.coefficient *
           (state(stencil.upper_iv, c_shift1 + shift_dir) -
            state(stencil.lower_iv, c_shift1 + shift_dir));
}

inline Tensor<2, double> source_spatial_metric_derivative(
    const FArrayBox &state, const MetricSourceDerivativeStencil &stencil)
{
    const auto upper = load_ccz4_geometry(state, stencil.upper_iv);
    const auto lower = load_ccz4_geometry(state, stencil.lower_iv);

    Tensor<2, double> derivative = 0.0;
    FOR(i, j)
    {
        derivative[i][j] = stencil.coefficient *
                           (upper.spatial_metric_LL[i][j] -
                            lower.spatial_metric_LL[i][j]);
    }
    return derivative;
}

inline double source_log_sqrt_det_derivative(
    const FArrayBox &state, const MetricSourceDerivativeStencil &stencil)
{
    return stencil.coefficient *
           (spatial_metric_log_sqrt_det(
                load_ccz4_geometry(state, stencil.upper_iv)) -
            spatial_metric_log_sqrt_det(
                load_ccz4_geometry(state, stencil.lower_iv)));
}

template <class eos_t>
void add_static_metric_sources_to_conserved_rhs(
    FArrayBox &rhs, const FArrayBox &state, const Box &interior_box,
    const eos_t &eos, const Tensor<1, double> &inverse_dx_U,
    const AtmosphereOptions &atmosphere = AtmosphereOptions(),
    const RecoveryOptions &recovery_options = RecoveryOptions())
{
    BoxIterator bit(interior_box);
    for (bit.begin(); bit.ok(); ++bit)
    {
        const IntVect iv = bit();
        const auto geometry = load_ccz4_geometry(state, iv);
        const auto conserved = load_conserved(state, iv);
        const auto recovered = recover_primitive(
            conserved, eos, geometry.spatial_metric_UU, recovery_options);
        if (!recovered.success)
        {
            std::ostringstream message;
            message << "GRHD primitive recovery failed at " << iv
                    << " while computing metric sources, residual "
                    << recovered.residual;
            throw std::runtime_error(message.str());
        }

        Primitive<double> primitive = recovered.primitive;
        enforce_primitive_floors(primitive, eos, geometry.spatial_metric_LL,
                                 atmosphere);
        if (is_metric_source_atmosphere_primitive(primitive, atmosphere))
            continue;

        StaticMetricDerivatives<double> derivatives;
        FOR(direction)
        {
            const auto stencil = make_metric_source_derivative_stencil(
                state, iv, direction, inverse_dx_U[direction], atmosphere);
            if (!stencil.valid)
                continue;

            derivatives.lapse[direction] =
                source_lapse_derivative(state, stencil);
            derivatives.log_sqrt_det_spatial_metric[direction] =
                source_log_sqrt_det_derivative(state, stencil);
            derivatives.spatial_metric_LL[direction] =
                source_spatial_metric_derivative(state, stencil);
            FOR(shift_dir)
            {
                derivatives.shift_U[shift_dir][direction] =
                    source_shift_derivative(state, stencil, shift_dir);
            }
        }

        const auto source = compute_static_metric_source_terms(
            primitive, conserved, eos, geometry, derivatives);
        rhs(iv, c_GRHD_D) += source.D;
        rhs(iv, c_GRHD_tau) += source.tau;
        FOR(momentum_dir)
        {
            rhs(iv, c_GRHD_S1 + momentum_dir) += source.S_L[momentum_dir];
        }
    }
}

inline void copy_ccz4_geometry(FArrayBox &state, const IntVect &dst_iv,
                               const IntVect &src_iv)
{
    state(dst_iv, c_chi) = state(src_iv, c_chi);
    state(dst_iv, c_h11) = state(src_iv, c_h11);
    state(dst_iv, c_h12) = state(src_iv, c_h12);
    state(dst_iv, c_h13) = state(src_iv, c_h13);
    state(dst_iv, c_h22) = state(src_iv, c_h22);
    state(dst_iv, c_h23) = state(src_iv, c_h23);
    state(dst_iv, c_h33) = state(src_iv, c_h33);
    state(dst_iv, c_lapse) = state(src_iv, c_lapse);
    state(dst_iv, c_shift1) = state(src_iv, c_shift1);
    state(dst_iv, c_shift2) = state(src_iv, c_shift2);
    state(dst_iv, c_shift3) = state(src_iv, c_shift3);
}

inline void fill_ccz4_geometry_outflow_boundaries(
    FArrayBox &state, const Box &interior_box, const Box &ghosted_box,
    const ProblemDomain &domain)
{
    const Box &domain_box = domain.domainBox();
    BoxIterator bit(ghosted_box);
    for (bit.begin(); bit.ok(); ++bit)
    {
        const IntVect iv = bit();
        bool is_physical_ghost = false;
        IntVect source_iv = iv;
        FOR(dir)
        {
            if (iv[dir] < domain_box.smallEnd(dir))
            {
                if (!domain.isPeriodic(dir))
                {
                    is_physical_ghost = true;
                    source_iv[dir] = interior_box.smallEnd(dir);
                }
            }
            else if (iv[dir] > domain_box.bigEnd(dir))
            {
                if (!domain.isPeriodic(dir))
                {
                    is_physical_ghost = true;
                    source_iv[dir] = interior_box.bigEnd(dir);
                }
            }
        }

        if (is_physical_ghost)
            copy_ccz4_geometry(state, iv, source_iv);
    }
}

inline void fill_ccz4_geometry_outflow_boundaries(FArrayBox &state,
                                                  const Box &interior_box,
                                                  const Box &ghosted_box,
                                                  const Box &domain_box)
{
    fill_ccz4_geometry_outflow_boundaries(
        state, interior_box, ghosted_box, ProblemDomain(domain_box));
}

inline void fill_conserved_outflow_boundaries(
    FArrayBox &state, const Box &interior_box, const Box &ghosted_box,
    const ProblemDomain &domain)
{
    const Box &domain_box = domain.domainBox();
    BoxIterator bit(ghosted_box);
    for (bit.begin(); bit.ok(); ++bit)
    {
        const IntVect iv = bit();
        bool is_physical_ghost = false;
        IntVect source_iv = iv;
        FOR(dir)
        {
            if (iv[dir] < domain_box.smallEnd(dir))
            {
                if (!domain.isPeriodic(dir))
                {
                    is_physical_ghost = true;
                    source_iv[dir] = interior_box.smallEnd(dir);
                }
            }
            else if (iv[dir] > domain_box.bigEnd(dir))
            {
                if (!domain.isPeriodic(dir))
                {
                    is_physical_ghost = true;
                    source_iv[dir] = interior_box.bigEnd(dir);
                }
            }
        }

        if (is_physical_ghost)
            store_conserved(state, iv, load_conserved(state, source_iv));
    }
}

inline void fill_conserved_outflow_boundaries(FArrayBox &state,
                                              const Box &interior_box,
                                              const Box &ghosted_box,
                                              const Box &domain_box)
{
    fill_conserved_outflow_boundaries(state, interior_box, ghosted_box,
                                      ProblemDomain(domain_box));
}

inline void fill_conserved_outflow_boundaries(FArrayBox &state,
                                              const Box &interior_box,
                                              const Box &ghosted_box)
{
    fill_conserved_outflow_boundaries(state, interior_box, ghosted_box,
                                      interior_box);
}

template <class eos_t>
void recover_primitives(FArrayBox &primitives, const FArrayBox &state,
                        const Box &box, const eos_t &eos,
                        const Tensor<2, double> &spatial_metric_UU,
                        const RecoveryOptions &options = RecoveryOptions())
{
    BoxIterator bit(box);
    for (bit.begin(); bit.ok(); ++bit)
    {
        const auto recovered = recover_primitive(
            load_conserved(state, bit()), eos, spatial_metric_UU, options);
        if (!recovered.success)
        {
            std::ostringstream message;
            message << "GRHD primitive recovery failed at " << bit()
                    << " with residual " << recovered.residual;
            throw std::runtime_error(message.str());
        }
        store_primitive(primitives, bit(), recovered.primitive);
    }
}

inline IntVect direction_offset(int direction)
{
    IntVect offset = IntVect::Zero;
    offset[direction] = 1;
    return offset;
}

inline void compute_primitive_slopes(FArrayBox &slopes,
                                     const FArrayBox &primitives,
                                     const Box &slope_box, int direction,
                                     double limiter_theta)
{
    slopes.setVal(0.0);
    const IntVect offset = direction_offset(direction);
    BoxIterator bit(slope_box);
    for (bit.begin(); bit.ok(); ++bit)
    {
        const IntVect iv = bit();
        const auto slope = compute_limited_slope(
            load_primitive(primitives, iv - offset), load_primitive(primitives, iv),
            load_primitive(primitives, iv + offset), limiter_theta);
        store_primitive(slopes, iv, slope);
    }
}

template <class eos_t>
Flux<double> compute_farraybox_muscl_hlle_flux(
    const FArrayBox &primitives, const FArrayBox &slopes, const eos_t &eos,
    const Tensor<2, double> &spatial_metric_LL,
    const Tensor<2, double> &spatial_metric_UU, const IntVect &left_iv,
    int direction, const double &lapse, const Tensor<1, double> &shift_U,
    const AtmosphereOptions &atmosphere = AtmosphereOptions(),
    bool use_reconstruction = true)
{
    const IntVect right_iv = left_iv + direction_offset(direction);
    return compute_muscl_hlle_flux(
        load_primitive(primitives, left_iv), load_primitive(primitives, right_iv),
        load_primitive(slopes, left_iv), load_primitive(slopes, right_iv), eos,
        spatial_metric_LL, spatial_metric_UU, direction, lapse, shift_U,
        atmosphere, use_reconstruction);
}

template <class eos_t>
void compute_directional_flux_rhs(
    FArrayBox &rhs, FArrayBox &state, const Box &interior_box,
    const Box &ghosted_box, const Box &domain_box, const eos_t &eos,
    const Tensor<2, double> &spatial_metric_LL,
    const Tensor<2, double> &spatial_metric_UU, int direction,
    const double &inverse_dx, const double &lapse,
    const Tensor<1, double> &shift_U,
    const AtmosphereOptions &atmosphere = AtmosphereOptions(),
    double limiter_theta = 1.5, bool use_reconstruction = true,
    const RecoveryOptions &recovery_options = RecoveryOptions())
{
    fill_conserved_outflow_boundaries(state, interior_box, ghosted_box,
                                      domain_box);

    FArrayBox primitives(ghosted_box, NUM_VARS);
    FArrayBox slopes(ghosted_box, NUM_VARS);
    primitives.setVal(0.0);
    slopes.setVal(0.0);
    recover_primitives(primitives, state, ghosted_box, eos, spatial_metric_UU,
                       recovery_options);

    Box slope_box = ghosted_box;
    slope_box.grow(-direction_offset(direction));
    compute_primitive_slopes(slopes, primitives, slope_box, direction,
                             limiter_theta);

    rhs.setVal(0.0);
    const IntVect offset = direction_offset(direction);
    BoxIterator bit(interior_box);
    for (bit.begin(); bit.ok(); ++bit)
    {
        const IntVect iv = bit();
        const auto flux_hi = compute_farraybox_muscl_hlle_flux(
            primitives, slopes, eos, spatial_metric_LL, spatial_metric_UU, iv,
            direction, lapse, shift_U, atmosphere, use_reconstruction);
        const auto flux_lo = compute_farraybox_muscl_hlle_flux(
            primitives, slopes, eos, spatial_metric_LL, spatial_metric_UU,
            iv - offset, direction, lapse, shift_U, atmosphere,
            use_reconstruction);

        Conserved<double> cell_rhs = zero_conserved<double>();
        cell_rhs.D = -inverse_dx * (flux_hi.D - flux_lo.D);
        cell_rhs.tau = -inverse_dx * (flux_hi.tau - flux_lo.tau);
        FOR(momentum_dir)
        {
            cell_rhs.S_L[momentum_dir] =
                -inverse_dx *
                (flux_hi.S_L[momentum_dir] - flux_lo.S_L[momentum_dir]);
        }
        store_conserved(rhs, iv, cell_rhs);
    }
}
template <class eos_t>
void compute_directional_flux_rhs(
    FArrayBox &rhs, FArrayBox &state, const Box &interior_box,
    const Box &ghosted_box, const eos_t &eos,
    const Tensor<2, double> &spatial_metric_LL,
    const Tensor<2, double> &spatial_metric_UU, int direction,
    const double &inverse_dx, const double &lapse,
    const Tensor<1, double> &shift_U,
    const AtmosphereOptions &atmosphere = AtmosphereOptions(),
    double limiter_theta = 1.5, bool use_reconstruction = true,
    const RecoveryOptions &recovery_options = RecoveryOptions())
{
    compute_directional_flux_rhs(
        rhs, state, interior_box, ghosted_box, interior_box, eos,
        spatial_metric_LL, spatial_metric_UU, direction, inverse_dx, lapse,
        shift_U, atmosphere, limiter_theta, use_reconstruction,
        recovery_options);
}


template <class eos_t>
void recover_primitives_from_ccz4_geometry(
    FArrayBox &primitives, const FArrayBox &state, const Box &box,
    const eos_t &eos, const RecoveryOptions &options = RecoveryOptions())
{
    BoxIterator bit(box);
    for (bit.begin(); bit.ok(); ++bit)
    {
        const auto geometry = load_ccz4_geometry(state, bit());
        const auto recovered = recover_primitive(
            load_conserved(state, bit()), eos, geometry.spatial_metric_UU,
            options);
        if (!recovered.success)
        {
            std::ostringstream message;
            message << "GRHD primitive recovery failed at " << bit()
                    << " with residual " << recovered.residual;
            throw std::runtime_error(message.str());
        }
        store_primitive(primitives, bit(), recovered.primitive);
    }
}

inline bool primitive_differs(const Primitive<double> &left,
                              const Primitive<double> &right)
{
    bool differs = left.rho != right.rho || left.eps != right.eps ||
                   left.pressure != right.pressure;
    FOR(i) { differs = differs || left.velocity_U[i] != right.velocity_U[i]; }
    return differs;
}

template <class eos_t>
AtmosphereResetDiagnostics
recover_primitives_and_apply_atmosphere_from_ccz4_geometry(
    FArrayBox &state, const Box &box, const eos_t &eos,
    const RecoveryOptions &options = RecoveryOptions(),
    const AtmosphereOptions &atmosphere = AtmosphereOptions())
{
    AtmosphereResetDiagnostics diagnostics;
    BoxIterator bit(box);
    for (bit.begin(); bit.ok(); ++bit)
    {
        ++diagnostics.num_cells;
        const IntVect iv = bit();
        const auto geometry = load_ccz4_geometry(state, iv);
        const auto recovered = recover_primitive(
            load_conserved(state, iv), eos, geometry.spatial_metric_UU,
            options);

        Primitive<double> primitive;
        bool reset_conserved = !recovered.success;
        if (recovered.success)
        {
            primitive = recovered.primitive;
            Primitive<double> floored_primitive = primitive;
            enforce_primitive_floors(floored_primitive, eos,
                                     geometry.spatial_metric_LL, atmosphere);
            reset_conserved = primitive_differs(primitive, floored_primitive);
            if (reset_conserved)
                ++diagnostics.num_floored_primitives;
            primitive = floored_primitive;
        }
        else
        {
            ++diagnostics.num_failed_recoveries;
            primitive =
                make_atmosphere_primitive<double>(eos, atmosphere);
        }

        store_primitive(state, iv, primitive);
        if (reset_conserved)
        {
            ++diagnostics.num_conserved_resets;
            store_conserved(state, iv,
                            compute_conserved(primitive, eos,
                                              geometry.spatial_metric_LL));
        }
    }
    return diagnostics;
}

template <class eos_t>
Flux<double> compute_farraybox_muscl_hlle_flux_from_ccz4_geometry(
    const FArrayBox &primitives, const FArrayBox &slopes,
    const FArrayBox &geometry_state, const eos_t &eos,
    const IntVect &left_iv, int direction,
    const AtmosphereOptions &atmosphere = AtmosphereOptions(),
    bool use_reconstruction = true)
{
    const IntVect right_iv = left_iv + direction_offset(direction);
    const auto geometry = average_geometry(load_ccz4_geometry(geometry_state,
                                                              left_iv),
                                           load_ccz4_geometry(geometry_state,
                                                              right_iv));
    return compute_muscl_hlle_flux(
        load_primitive(primitives, left_iv), load_primitive(primitives, right_iv),
        load_primitive(slopes, left_iv), load_primitive(slopes, right_iv), eos,
        geometry.spatial_metric_LL, geometry.spatial_metric_UU, direction,
        geometry.lapse, geometry.shift_U, atmosphere, use_reconstruction);
}

template <class eos_t>
void compute_directional_flux_rhs_from_ccz4_geometry(
    FArrayBox &rhs, FArrayBox &state, const Box &interior_box,
    const Box &ghosted_box, const ProblemDomain &domain, const eos_t &eos,
    int direction, const double &inverse_dx,
    const AtmosphereOptions &atmosphere = AtmosphereOptions(),
    double limiter_theta = 1.5, bool use_reconstruction = true,
    const RecoveryOptions &recovery_options = RecoveryOptions())
{
    fill_conserved_outflow_boundaries(state, interior_box, ghosted_box,
                                      domain);
    fill_ccz4_geometry_outflow_boundaries(state, interior_box, ghosted_box,
                                          domain);

    FArrayBox primitives(ghosted_box, NUM_VARS);
    FArrayBox slopes(ghosted_box, NUM_VARS);
    primitives.setVal(0.0);
    slopes.setVal(0.0);
    recover_primitives_from_ccz4_geometry(primitives, state, ghosted_box, eos,
                                          recovery_options);

    Box slope_box = ghosted_box;
    slope_box.grow(-direction_offset(direction));
    compute_primitive_slopes(slopes, primitives, slope_box, direction,
                             limiter_theta);

    rhs.setVal(0.0);
    const IntVect offset = direction_offset(direction);
    BoxIterator bit(interior_box);
    for (bit.begin(); bit.ok(); ++bit)
    {
        const IntVect iv = bit();
        const auto flux_hi = compute_farraybox_muscl_hlle_flux_from_ccz4_geometry(
            primitives, slopes, state, eos, iv, direction, atmosphere,
            use_reconstruction);
        const auto flux_lo = compute_farraybox_muscl_hlle_flux_from_ccz4_geometry(
            primitives, slopes, state, eos, iv - offset, direction, atmosphere,
            use_reconstruction);

        Conserved<double> cell_rhs = zero_conserved<double>();
        cell_rhs.D = -inverse_dx * (flux_hi.D - flux_lo.D);
        cell_rhs.tau = -inverse_dx * (flux_hi.tau - flux_lo.tau);
        FOR(momentum_dir)
        {
            cell_rhs.S_L[momentum_dir] =
                -inverse_dx *
                (flux_hi.S_L[momentum_dir] - flux_lo.S_L[momentum_dir]);
        }
        store_conserved(rhs, iv, cell_rhs);
    }
}

template <class eos_t>
void compute_directional_flux_rhs_from_ccz4_geometry(
    FArrayBox &rhs, FArrayBox &state, const Box &interior_box,
    const Box &ghosted_box, const Box &domain_box, const eos_t &eos,
    int direction, const double &inverse_dx,
    const AtmosphereOptions &atmosphere = AtmosphereOptions(),
    double limiter_theta = 1.5, bool use_reconstruction = true,
    const RecoveryOptions &recovery_options = RecoveryOptions())
{
    fill_conserved_outflow_boundaries(state, interior_box, ghosted_box,
                                      domain_box);
    fill_ccz4_geometry_outflow_boundaries(state, interior_box, ghosted_box,
                                          domain_box);

    FArrayBox primitives(ghosted_box, NUM_VARS);
    FArrayBox slopes(ghosted_box, NUM_VARS);
    primitives.setVal(0.0);
    slopes.setVal(0.0);
    recover_primitives_from_ccz4_geometry(primitives, state, ghosted_box, eos,
                                          recovery_options);

    Box slope_box = ghosted_box;
    slope_box.grow(-direction_offset(direction));
    compute_primitive_slopes(slopes, primitives, slope_box, direction,
                             limiter_theta);

    rhs.setVal(0.0);
    const IntVect offset = direction_offset(direction);
    BoxIterator bit(interior_box);
    for (bit.begin(); bit.ok(); ++bit)
    {
        const IntVect iv = bit();
        const auto flux_hi = compute_farraybox_muscl_hlle_flux_from_ccz4_geometry(
            primitives, slopes, state, eos, iv, direction, atmosphere,
            use_reconstruction);
        const auto flux_lo = compute_farraybox_muscl_hlle_flux_from_ccz4_geometry(
            primitives, slopes, state, eos, iv - offset, direction, atmosphere,
            use_reconstruction);

        Conserved<double> cell_rhs = zero_conserved<double>();
        cell_rhs.D = -inverse_dx * (flux_hi.D - flux_lo.D);
        cell_rhs.tau = -inverse_dx * (flux_hi.tau - flux_lo.tau);
        FOR(momentum_dir)
        {
            cell_rhs.S_L[momentum_dir] =
                -inverse_dx *
                (flux_hi.S_L[momentum_dir] - flux_lo.S_L[momentum_dir]);
        }
        store_conserved(rhs, iv, cell_rhs);
    }
}

template <class eos_t>
void compute_leveldata_directional_flux_rhs(
    LevelData<FArrayBox> &rhs, LevelData<FArrayBox> &state,
    const Box &domain_box, const eos_t &eos,
    const Tensor<2, double> &spatial_metric_LL,
    const Tensor<2, double> &spatial_metric_UU, int direction,
    const double &inverse_dx, const double &lapse,
    const Tensor<1, double> &shift_U,
    const AtmosphereOptions &atmosphere = AtmosphereOptions(),
    double limiter_theta = 1.5, bool use_reconstruction = true,
    const RecoveryOptions &recovery_options = RecoveryOptions())
{
    state.exchange();
    const DisjointBoxLayout &grids = state.disjointBoxLayout();
    DataIterator dit = grids.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        compute_directional_flux_rhs(
            rhs[dit], state[dit], grids[dit], state[dit].box(), domain_box,
            eos, spatial_metric_LL, spatial_metric_UU, direction, inverse_dx,
            lapse, shift_U, atmosphere, limiter_theta, use_reconstruction,
            recovery_options);
    }
}

inline void add_conserved_rhs(FArrayBox &rhs, const FArrayBox &directional_rhs,
                              const Box &interior_box)
{
    BoxIterator bit(interior_box);
    for (bit.begin(); bit.ok(); ++bit)
    {
        const IntVect iv = bit();
        rhs(iv, c_GRHD_D) += directional_rhs(iv, c_GRHD_D);
        rhs(iv, c_GRHD_tau) += directional_rhs(iv, c_GRHD_tau);
        rhs(iv, c_GRHD_S1) += directional_rhs(iv, c_GRHD_S1);
        rhs(iv, c_GRHD_S2) += directional_rhs(iv, c_GRHD_S2);
        rhs(iv, c_GRHD_S3) += directional_rhs(iv, c_GRHD_S3);
    }
}

inline void add_scaled_conserved_rhs(FArrayBox &state, const FArrayBox &rhs,
                                     const Box &interior_box, double scale)
{
    BoxIterator bit(interior_box);
    for (bit.begin(); bit.ok(); ++bit)
    {
        const IntVect iv = bit();
        state(iv, c_GRHD_D) += scale * rhs(iv, c_GRHD_D);
        state(iv, c_GRHD_tau) += scale * rhs(iv, c_GRHD_tau);
        state(iv, c_GRHD_S1) += scale * rhs(iv, c_GRHD_S1);
        state(iv, c_GRHD_S2) += scale * rhs(iv, c_GRHD_S2);
        state(iv, c_GRHD_S3) += scale * rhs(iv, c_GRHD_S3);
    }
}

inline void combine_ssprk2_conserved_state(FArrayBox &state,
                                           const FArrayBox &old_state,
                                           const FArrayBox &stage_state,
                                           const FArrayBox &stage_rhs,
                                           const Box &interior_box, double dt)
{
    BoxIterator bit(interior_box);
    for (bit.begin(); bit.ok(); ++bit)
    {
        const IntVect iv = bit();
        state(iv, c_GRHD_D) =
            0.5 * (old_state(iv, c_GRHD_D) + stage_state(iv, c_GRHD_D) +
                   dt * stage_rhs(iv, c_GRHD_D));
        state(iv, c_GRHD_tau) =
            0.5 * (old_state(iv, c_GRHD_tau) +
                   stage_state(iv, c_GRHD_tau) +
                   dt * stage_rhs(iv, c_GRHD_tau));
        state(iv, c_GRHD_S1) =
            0.5 * (old_state(iv, c_GRHD_S1) + stage_state(iv, c_GRHD_S1) +
                   dt * stage_rhs(iv, c_GRHD_S1));
        state(iv, c_GRHD_S2) =
            0.5 * (old_state(iv, c_GRHD_S2) + stage_state(iv, c_GRHD_S2) +
                   dt * stage_rhs(iv, c_GRHD_S2));
        state(iv, c_GRHD_S3) =
            0.5 * (old_state(iv, c_GRHD_S3) + stage_state(iv, c_GRHD_S3) +
                   dt * stage_rhs(iv, c_GRHD_S3));
    }
}

template <class eos_t>
void compute_leveldata_flux_rhs(
    LevelData<FArrayBox> &rhs, LevelData<FArrayBox> &state,
    const Box &domain_box, const eos_t &eos,
    const Tensor<2, double> &spatial_metric_LL,
    const Tensor<2, double> &spatial_metric_UU,
    const Tensor<1, double> &inverse_dx_U, const double &lapse,
    const Tensor<1, double> &shift_U,
    const AtmosphereOptions &atmosphere = AtmosphereOptions(),
    double limiter_theta = 1.5, bool use_reconstruction = true,
    const RecoveryOptions &recovery_options = RecoveryOptions())
{
    const DisjointBoxLayout &grids = state.disjointBoxLayout();
    DataIterator dit = grids.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
        rhs[dit].setVal(0.0);

    LevelData<FArrayBox> directional_rhs(grids, NUM_VARS, IntVect::Zero);
    FOR(direction)
    {
        compute_leveldata_directional_flux_rhs(
            directional_rhs, state, domain_box, eos, spatial_metric_LL,
            spatial_metric_UU, direction, inverse_dx_U[direction], lapse,
            shift_U, atmosphere, limiter_theta, use_reconstruction,
            recovery_options);

        DataIterator add_dit = grids.dataIterator();
        for (add_dit.begin(); add_dit.ok(); ++add_dit)
            add_conserved_rhs(rhs[add_dit], directional_rhs[add_dit],
                              grids[add_dit]);
    }
}

template <class eos_t>
void compute_leveldata_directional_flux_rhs_from_ccz4_geometry(
    LevelData<FArrayBox> &rhs, LevelData<FArrayBox> &state,
    const ProblemDomain &domain, const eos_t &eos, int direction,
    const double &inverse_dx,
    const AtmosphereOptions &atmosphere = AtmosphereOptions(),
    double limiter_theta = 1.5, bool use_reconstruction = true,
    const RecoveryOptions &recovery_options = RecoveryOptions())
{
    state.exchange();
    const DisjointBoxLayout &grids = state.disjointBoxLayout();
    DataIterator dit = grids.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        compute_directional_flux_rhs_from_ccz4_geometry(
            rhs[dit], state[dit], grids[dit], state[dit].box(), domain, eos,
            direction, inverse_dx, atmosphere, limiter_theta,
            use_reconstruction, recovery_options);
    }
}

template <class eos_t>
void compute_leveldata_directional_flux_rhs_from_ccz4_geometry(
    LevelData<FArrayBox> &rhs, LevelData<FArrayBox> &state,
    const Box &domain_box, const eos_t &eos, int direction,
    const double &inverse_dx,
    const AtmosphereOptions &atmosphere = AtmosphereOptions(),
    double limiter_theta = 1.5, bool use_reconstruction = true,
    const RecoveryOptions &recovery_options = RecoveryOptions())
{
    state.exchange();
    const DisjointBoxLayout &grids = state.disjointBoxLayout();
    DataIterator dit = grids.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        compute_directional_flux_rhs_from_ccz4_geometry(
            rhs[dit], state[dit], grids[dit], state[dit].box(), domain_box,
            eos, direction, inverse_dx, atmosphere, limiter_theta,
            use_reconstruction, recovery_options);
    }
}

template <class eos_t>
void compute_leveldata_flux_rhs_from_ccz4_geometry(
    LevelData<FArrayBox> &rhs, LevelData<FArrayBox> &state,
    const ProblemDomain &domain, const eos_t &eos,
    const Tensor<1, double> &inverse_dx_U,
    const AtmosphereOptions &atmosphere = AtmosphereOptions(),
    double limiter_theta = 1.5, bool use_reconstruction = true,
    bool use_static_metric_sources = true,
    const RecoveryOptions &recovery_options = RecoveryOptions())
{
    const DisjointBoxLayout &grids = state.disjointBoxLayout();
    DataIterator dit = grids.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
        rhs[dit].setVal(0.0);

    LevelData<FArrayBox> directional_rhs(grids, NUM_VARS, IntVect::Zero);
    FOR(direction)
    {
        compute_leveldata_directional_flux_rhs_from_ccz4_geometry(
            directional_rhs, state, domain, eos, direction,
            inverse_dx_U[direction], atmosphere, limiter_theta,
            use_reconstruction, recovery_options);

        DataIterator add_dit = grids.dataIterator();
        for (add_dit.begin(); add_dit.ok(); ++add_dit)
            add_conserved_rhs(rhs[add_dit], directional_rhs[add_dit],
                              grids[add_dit]);
    }

    if (!use_static_metric_sources)
        return;

    state.exchange();
    DataIterator source_dit = grids.dataIterator();
    for (source_dit.begin(); source_dit.ok(); ++source_dit)
    {
        fill_ccz4_geometry_outflow_boundaries(state[source_dit],
                                              grids[source_dit],
                                              state[source_dit].box(),
                                              domain);
        add_static_metric_sources_to_conserved_rhs(
            rhs[source_dit], state[source_dit], grids[source_dit], eos,
            inverse_dx_U, atmosphere, recovery_options);
    }
}

template <class eos_t>
void compute_leveldata_flux_rhs_from_ccz4_geometry(
    LevelData<FArrayBox> &rhs, LevelData<FArrayBox> &state,
    const Box &domain_box, const eos_t &eos,
    const Tensor<1, double> &inverse_dx_U,
    const AtmosphereOptions &atmosphere = AtmosphereOptions(),
    double limiter_theta = 1.5, bool use_reconstruction = true,
    bool use_static_metric_sources = true,
    const RecoveryOptions &recovery_options = RecoveryOptions())
{
    const DisjointBoxLayout &grids = state.disjointBoxLayout();
    DataIterator dit = grids.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
        rhs[dit].setVal(0.0);

    LevelData<FArrayBox> directional_rhs(grids, NUM_VARS, IntVect::Zero);
    FOR(direction)
    {
        compute_leveldata_directional_flux_rhs_from_ccz4_geometry(
            directional_rhs, state, domain_box, eos, direction,
            inverse_dx_U[direction], atmosphere, limiter_theta,
            use_reconstruction, recovery_options);

        DataIterator add_dit = grids.dataIterator();
        for (add_dit.begin(); add_dit.ok(); ++add_dit)
            add_conserved_rhs(rhs[add_dit], directional_rhs[add_dit],
                              grids[add_dit]);
    }

    if (!use_static_metric_sources)
        return;

    state.exchange();
    DataIterator source_dit = grids.dataIterator();
    for (source_dit.begin(); source_dit.ok(); ++source_dit)
    {
        fill_ccz4_geometry_outflow_boundaries(state[source_dit],
                                              grids[source_dit],
                                              state[source_dit].box(),
                                              domain_box);
        add_static_metric_sources_to_conserved_rhs(
            rhs[source_dit], state[source_dit], grids[source_dit], eos,
            inverse_dx_U, atmosphere, recovery_options);
    }
}


template <class eos_t>
double compute_max_inverse_dt_from_ccz4_geometry(
    const FArrayBox &state, const Box &interior_box, const eos_t &eos,
    const Tensor<1, double> &inverse_dx_U,
    const AtmosphereOptions &atmosphere = AtmosphereOptions(),
    const RecoveryOptions &recovery_options = RecoveryOptions())
{
    double max_inverse_dt = 0.0;
    BoxIterator bit(interior_box);
    for (bit.begin(); bit.ok(); ++bit)
    {
        const auto geometry = load_ccz4_geometry(state, bit());
        const auto recovered = recover_primitive(
            load_conserved(state, bit()), eos, geometry.spatial_metric_UU,
            recovery_options);
        if (!recovered.success)
        {
            std::ostringstream message;
            message << "GRHD primitive recovery failed at " << bit()
                    << " while computing CFL speed, residual "
                    << recovered.residual;
            throw std::runtime_error(message.str());
        }

        Primitive<double> primitive = recovered.primitive;
        enforce_primitive_floors(primitive, eos, geometry.spatial_metric_LL,
                                 atmosphere);
        if (is_atmosphere_primitive(primitive, atmosphere))
            continue;

        double cell_inverse_dt = 0.0;
        FOR(direction)
        {
            const auto speeds = compute_characteristic_speeds(
                primitive, eos, geometry.spatial_metric_LL,
                geometry.spatial_metric_UU, direction, geometry.lapse,
                geometry.shift_U);
            const double max_speed =
                std::max(std::abs(speeds.lower), std::abs(speeds.upper));
            cell_inverse_dt += max_speed * inverse_dx_U[direction];
        }
        max_inverse_dt = std::max(max_inverse_dt, cell_inverse_dt);
    }
    return max_inverse_dt;
}

template <class eos_t>
double compute_leveldata_max_inverse_dt_from_ccz4_geometry(
    const LevelData<FArrayBox> &state, const Box &domain_box,
    const eos_t &eos, const Tensor<1, double> &inverse_dx_U,
    const AtmosphereOptions &atmosphere = AtmosphereOptions(),
    const RecoveryOptions &recovery_options = RecoveryOptions())
{
    (void)domain_box;
    const DisjointBoxLayout &grids = state.disjointBoxLayout();
    double local_max_inverse_dt = 0.0;
    DataIterator dit = grids.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        local_max_inverse_dt = std::max(
            local_max_inverse_dt,
            compute_max_inverse_dt_from_ccz4_geometry(
                state[dit], grids[dit], eos, inverse_dx_U, atmosphere,
                recovery_options));
    }

#ifdef CH_MPI
    double global_max_inverse_dt = 0.0;
    MPI_Allreduce(&local_max_inverse_dt, &global_max_inverse_dt, 1,
                  MPI_DOUBLE, MPI_MAX, Chombo_MPI::comm);
    return global_max_inverse_dt;
#else
    return local_max_inverse_dt;
#endif
}

template <class eos_t> class LevelDataFiniteVolumeOperator
{
  public:
    LevelDataFiniteVolumeOperator(
        const eos_t &a_eos,
        const AtmosphereOptions &a_atmosphere = AtmosphereOptions(),
        double a_limiter_theta = 1.5, bool a_use_reconstruction = true,
        const RecoveryOptions &a_recovery_options = RecoveryOptions(),
        bool a_use_static_metric_sources = true)
        : m_eos(a_eos), m_atmosphere(a_atmosphere),
          m_limiter_theta(a_limiter_theta),
          m_use_reconstruction(a_use_reconstruction),
          m_recovery_options(a_recovery_options),
          m_use_static_metric_sources(a_use_static_metric_sources)
    {
    }

    void compute_rhs(LevelData<FArrayBox> &rhs, LevelData<FArrayBox> &state,
                     const ProblemDomain &domain,
                     const Tensor<1, double> &inverse_dx_U) const
    {
        compute_leveldata_flux_rhs_from_ccz4_geometry(
            rhs, state, domain, m_eos, inverse_dx_U, m_atmosphere,
            m_limiter_theta, m_use_reconstruction,
            m_use_static_metric_sources, m_recovery_options);
    }

    void compute_rhs(LevelData<FArrayBox> &rhs, LevelData<FArrayBox> &state,
                     const Box &domain_box,
                     const Tensor<1, double> &inverse_dx_U) const
    {
        compute_rhs(rhs, state, ProblemDomain(domain_box), inverse_dx_U);
    }

    void add_conserved_rhs_to(LevelData<FArrayBox> &rhs,
                              LevelData<FArrayBox> &state,
                              const ProblemDomain &domain,
                              const Tensor<1, double> &inverse_dx_U) const
    {
        const DisjointBoxLayout &grids = state.disjointBoxLayout();
        LevelData<FArrayBox> hydro_rhs(grids, NUM_VARS, IntVect::Zero);
        compute_rhs(hydro_rhs, state, domain, inverse_dx_U);

        DataIterator dit = grids.dataIterator();
        for (dit.begin(); dit.ok(); ++dit)
            add_conserved_rhs(rhs[dit], hydro_rhs[dit], grids[dit]);
    }

    void add_conserved_rhs_to(LevelData<FArrayBox> &rhs,
                              LevelData<FArrayBox> &state,
                              const Box &domain_box,
                              const Tensor<1, double> &inverse_dx_U) const
    {
        add_conserved_rhs_to(rhs, state, ProblemDomain(domain_box),
                             inverse_dx_U);
    }

    AtmosphereResetDiagnostics
    recover_primitives(LevelData<FArrayBox> &state) const
    {
        AtmosphereResetDiagnostics local_diagnostics;
        const DisjointBoxLayout &grids = state.disjointBoxLayout();
        DataIterator dit = grids.dataIterator();
        for (dit.begin(); dit.ok(); ++dit)
        {
            local_diagnostics.add(
                recover_primitives_and_apply_atmosphere_from_ccz4_geometry(
                    state[dit], grids[dit], m_eos, m_recovery_options,
                    m_atmosphere));
        }
        return reduce_atmosphere_reset_diagnostics(local_diagnostics);
    }

    void update_conserved(LevelData<FArrayBox> &state,
                          const LevelData<FArrayBox> &rhs, double dt,
                          bool recover_primitives_after_update = true) const
    {
        const DisjointBoxLayout &grids = state.disjointBoxLayout();
        DataIterator dit = grids.dataIterator();
        for (dit.begin(); dit.ok(); ++dit)
            add_scaled_conserved_rhs(state[dit], rhs[dit], grids[dit], dt);

        if (recover_primitives_after_update)
            recover_primitives(state);
    }

    double compute_max_inverse_dt(
        const LevelData<FArrayBox> &state, const Box &domain_box,
        const Tensor<1, double> &inverse_dx_U) const
    {
        return compute_leveldata_max_inverse_dt_from_ccz4_geometry(
            state, domain_box, m_eos, inverse_dx_U, m_atmosphere,
            m_recovery_options);
    }

    double compute_stable_dt(const LevelData<FArrayBox> &state,
                             const Box &domain_box,
                             const Tensor<1, double> &inverse_dx_U,
                             double cfl) const
    {
        if (cfl <= 0.0)
            throw std::invalid_argument("GRHD CFL number must be positive");

        const double max_inverse_dt =
            compute_max_inverse_dt(state, domain_box, inverse_dx_U);
        return cfl / std::max(max_inverse_dt, 1.0e-14);
    }

    void advance_ssprk2(LevelData<FArrayBox> &state,
                        const ProblemDomain &domain,
                        const Tensor<1, double> &inverse_dx_U,
                        double dt) const
    {
        const DisjointBoxLayout &grids = state.disjointBoxLayout();
        LevelData<FArrayBox> old_state(grids, NUM_VARS, state.ghostVect());
        LevelData<FArrayBox> stage_state(grids, NUM_VARS, state.ghostVect());
        LevelData<FArrayBox> rhs_initial(grids, NUM_VARS, IntVect::Zero);
        LevelData<FArrayBox> rhs_stage(grids, NUM_VARS, IntVect::Zero);

        state.copyTo(old_state);
        state.copyTo(stage_state);
        compute_rhs(rhs_initial, state, domain, inverse_dx_U);
        update_conserved(stage_state, rhs_initial, dt);
        compute_rhs(rhs_stage, stage_state, domain, inverse_dx_U);

        DataIterator dit = grids.dataIterator();
        for (dit.begin(); dit.ok(); ++dit)
        {
            combine_ssprk2_conserved_state(state[dit], old_state[dit],
                                           stage_state[dit], rhs_stage[dit],
                                           grids[dit], dt);
        }
        recover_primitives(state);
    }

    void advance_ssprk2(LevelData<FArrayBox> &state, const Box &domain_box,
                        const Tensor<1, double> &inverse_dx_U,
                        double dt) const
    {
        advance_ssprk2(state, ProblemDomain(domain_box), inverse_dx_U, dt);
    }

  private:
    eos_t m_eos;
    AtmosphereOptions m_atmosphere;
    double m_limiter_theta;
    bool m_use_reconstruction;
    RecoveryOptions m_recovery_options;
    bool m_use_static_metric_sources;
};

} // namespace GRHD

#endif /* GRHD_FARRAYBOX_FINITEVOLUME_HPP_ */
