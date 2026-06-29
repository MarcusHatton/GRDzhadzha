/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef GRHD_FIXED_BG_FARRAYBOX_FINITEVOLUME_HPP_
#define GRHD_FIXED_BG_FARRAYBOX_FINITEVOLUME_HPP_

#include "Coordinates.hpp"
#include "FArrayBoxFiniteVolume.hpp"
#include "FixedBGMetric.hpp"

namespace GRHD
{
template <class background_t>
CellGeometry load_fixed_background_geometry(
    const background_t &background, const IntVect &iv, double dx,
    const std::array<double, CH_SPACEDIM> &center)
{
    return make_cell_geometry_from_background(
        background, Coordinates<double>(iv, dx, center));
}

template <class background_t>
StaticMetricDerivatives<double> load_fixed_background_metric_derivatives(
    const background_t &background, const IntVect &iv, double dx,
    const std::array<double, CH_SPACEDIM> &center)
{
    return make_static_metric_derivatives_from_background(
        background, Coordinates<double>(iv, dx, center));
}

template <class eos_t, class background_t>
void recover_primitives_from_fixed_background(
    FArrayBox &primitives, const FArrayBox &state, const Box &box,
    const eos_t &eos, const background_t &background, double dx,
    const std::array<double, CH_SPACEDIM> &center,
    const RecoveryOptions &options = RecoveryOptions())
{
    BoxIterator bit(box);
    for (bit.begin(); bit.ok(); ++bit)
    {
        const auto geometry = load_fixed_background_geometry(
            background, bit(), dx, center);
        const auto recovered = recover_primitive(
            load_conserved(state, bit()), eos, geometry.spatial_metric_UU,
            options);
        if (!recovered.success)
        {
            std::ostringstream message;
            message << "GRHD primitive recovery failed at " << bit()
                    << " with residual " << recovered.residual
                    << " on fixed background";
            throw std::runtime_error(message.str());
        }
        store_primitive(primitives, bit(), recovered.primitive);
    }
}

template <class eos_t, class background_t>
AtmosphereResetDiagnostics
recover_primitives_and_apply_atmosphere_from_fixed_background(
    FArrayBox &state, const Box &box, const eos_t &eos,
    const background_t &background, double dx,
    const std::array<double, CH_SPACEDIM> &center,
    const RecoveryOptions &options = RecoveryOptions(),
    const AtmosphereOptions &atmosphere = AtmosphereOptions())
{
    AtmosphereResetDiagnostics diagnostics;
    BoxIterator bit(box);
    for (bit.begin(); bit.ok(); ++bit)
    {
        ++diagnostics.num_cells;
        const IntVect iv = bit();
        const auto geometry = load_fixed_background_geometry(
            background, iv, dx, center);
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
            primitive = make_atmosphere_primitive<double>(eos, atmosphere);
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


template <class eos_t, class background_t>
void fill_conserved_outflow_boundaries_from_fixed_background(
    FArrayBox &state, const Box &interior_box, const Box &ghosted_box,
    const ProblemDomain &domain, const background_t &background,
    const eos_t &eos, double dx,
    const std::array<double, CH_SPACEDIM> &center,
    const AtmosphereOptions &atmosphere = AtmosphereOptions(),
    const RecoveryOptions &recovery_options = RecoveryOptions())
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

        if (!is_physical_ghost)
            continue;

        const auto source_geometry = load_fixed_background_geometry(
            background, source_iv, dx, center);
        const auto recovered = recover_primitive(
            load_conserved(state, source_iv), eos,
            source_geometry.spatial_metric_UU, recovery_options);
        if (!recovered.success)
        {
            std::ostringstream message;
            message << "GRHD primitive recovery failed at outflow source "
                    << source_iv << " while filling fixed-background "
                    << "boundary, residual " << recovered.residual;
            throw std::runtime_error(message.str());
        }

        Primitive<double> primitive = recovered.primitive;
        const auto target_geometry = load_fixed_background_geometry(
            background, iv, dx, center);
        enforce_primitive_floors(primitive, eos,
                                 target_geometry.spatial_metric_LL,
                                 atmosphere);
        store_primitive(state, iv, primitive);
        store_conserved(state, iv,
                        compute_conserved(primitive, eos,
                                          target_geometry.spatial_metric_LL));
    }
}

template <class eos_t, class background_t>
Flux<double> compute_farraybox_muscl_hlle_flux_from_fixed_background(
    const FArrayBox &primitives, const FArrayBox &slopes,
    const background_t &background, const eos_t &eos,
    const IntVect &left_iv, int direction, double dx,
    const std::array<double, CH_SPACEDIM> &center,
    const AtmosphereOptions &atmosphere = AtmosphereOptions(),
    bool use_reconstruction = true)
{
    const IntVect right_iv = left_iv + direction_offset(direction);
    const auto geometry = average_geometry(
        load_fixed_background_geometry(background, left_iv, dx, center),
        load_fixed_background_geometry(background, right_iv, dx, center));
    return compute_muscl_hlle_flux(
        load_primitive(primitives, left_iv), load_primitive(primitives, right_iv),
        load_primitive(slopes, left_iv), load_primitive(slopes, right_iv), eos,
        geometry.spatial_metric_LL, geometry.spatial_metric_UU, direction,
        geometry.lapse, geometry.shift_U, atmosphere, use_reconstruction);
}

template <class eos_t, class background_t>
void compute_directional_flux_rhs_from_fixed_background(
    FArrayBox &rhs, FArrayBox &state, const Box &interior_box,
    const Box &ghosted_box, const ProblemDomain &domain,
    const background_t &background, const eos_t &eos, int direction,
    const double &inverse_dx, double dx,
    const std::array<double, CH_SPACEDIM> &center,
    const AtmosphereOptions &atmosphere = AtmosphereOptions(),
    double limiter_theta = 1.5, bool use_reconstruction = true,
    const RecoveryOptions &recovery_options = RecoveryOptions())
{
    fill_conserved_outflow_boundaries_from_fixed_background(
        state, interior_box, ghosted_box, domain, background, eos, dx,
        center, atmosphere, recovery_options);

    FArrayBox primitives(ghosted_box, NUM_VARS);
    FArrayBox slopes(ghosted_box, NUM_VARS);
    primitives.setVal(0.0);
    slopes.setVal(0.0);
    recover_primitives_from_fixed_background(
        primitives, state, ghosted_box, eos, background, dx, center,
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
        const auto flux_hi =
            compute_farraybox_muscl_hlle_flux_from_fixed_background(
                primitives, slopes, background, eos, iv, direction, dx,
                center, atmosphere, use_reconstruction);
        const auto flux_lo =
            compute_farraybox_muscl_hlle_flux_from_fixed_background(
                primitives, slopes, background, eos, iv - offset, direction,
                dx, center, atmosphere, use_reconstruction);

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

template <class eos_t, class background_t>
void add_static_metric_sources_from_fixed_background_to_conserved_rhs(
    FArrayBox &rhs, const FArrayBox &state, const Box &interior_box,
    const background_t &background, const eos_t &eos, double dx,
    const std::array<double, CH_SPACEDIM> &center,
    const AtmosphereOptions &atmosphere = AtmosphereOptions(),
    const RecoveryOptions &recovery_options = RecoveryOptions())
{
    BoxIterator bit(interior_box);
    for (bit.begin(); bit.ok(); ++bit)
    {
        const IntVect iv = bit();
        const auto geometry = load_fixed_background_geometry(
            background, iv, dx, center);
        const auto conserved = load_conserved(state, iv);
        const auto recovered = recover_primitive(
            conserved, eos, geometry.spatial_metric_UU, recovery_options);
        if (!recovered.success)
        {
            std::ostringstream message;
            message << "GRHD primitive recovery failed at " << iv
                    << " while computing fixed-background metric sources, "
                    << "residual " << recovered.residual;
            throw std::runtime_error(message.str());
        }

        Primitive<double> primitive = recovered.primitive;
        enforce_primitive_floors(primitive, eos, geometry.spatial_metric_LL,
                                 atmosphere);
        if (is_metric_source_atmosphere_primitive(primitive, atmosphere))
            continue;

        const auto derivatives = load_fixed_background_metric_derivatives(
            background, iv, dx, center);
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

template <class eos_t, class background_t>
void compute_leveldata_directional_flux_rhs_from_fixed_background(
    LevelData<FArrayBox> &rhs, LevelData<FArrayBox> &state,
    const ProblemDomain &domain, const background_t &background,
    const eos_t &eos, int direction, const double &inverse_dx, double dx,
    const std::array<double, CH_SPACEDIM> &center,
    const AtmosphereOptions &atmosphere = AtmosphereOptions(),
    double limiter_theta = 1.5, bool use_reconstruction = true,
    const RecoveryOptions &recovery_options = RecoveryOptions())
{
    state.exchange();
    const DisjointBoxLayout &grids = state.disjointBoxLayout();
    DataIterator dit = grids.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        compute_directional_flux_rhs_from_fixed_background(
            rhs[dit], state[dit], grids[dit], state[dit].box(), domain,
            background, eos, direction, inverse_dx, dx, center, atmosphere,
            limiter_theta, use_reconstruction, recovery_options);
    }
}

template <class eos_t, class background_t>
void compute_leveldata_flux_rhs_from_fixed_background(
    LevelData<FArrayBox> &rhs, LevelData<FArrayBox> &state,
    const ProblemDomain &domain, const background_t &background,
    const eos_t &eos, const Tensor<1, double> &inverse_dx_U, double dx,
    const std::array<double, CH_SPACEDIM> &center,
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
        compute_leveldata_directional_flux_rhs_from_fixed_background(
            directional_rhs, state, domain, background, eos, direction,
            inverse_dx_U[direction], dx, center, atmosphere, limiter_theta,
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
        add_static_metric_sources_from_fixed_background_to_conserved_rhs(
            rhs[source_dit], state[source_dit], grids[source_dit],
            background, eos, dx, center, atmosphere, recovery_options);
    }
}
template <class eos_t, class background_t>
double compute_max_inverse_dt_from_fixed_background(
    const FArrayBox &state, const Box &interior_box,
    const background_t &background, const eos_t &eos,
    const Tensor<1, double> &inverse_dx_U, double dx,
    const std::array<double, CH_SPACEDIM> &center,
    const AtmosphereOptions &atmosphere = AtmosphereOptions(),
    const RecoveryOptions &recovery_options = RecoveryOptions())
{
    double max_inverse_dt = 0.0;
    BoxIterator bit(interior_box);
    for (bit.begin(); bit.ok(); ++bit)
    {
        const IntVect iv = bit();
        const auto geometry = load_fixed_background_geometry(
            background, iv, dx, center);
        const auto recovered = recover_primitive(
            load_conserved(state, iv), eos, geometry.spatial_metric_UU,
            recovery_options);
        if (!recovered.success)
        {
            std::ostringstream message;
            message << "GRHD primitive recovery failed at " << iv
                    << " while computing fixed-background CFL speed, "
                    << "residual " << recovered.residual;
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

template <class eos_t, class background_t>
double compute_leveldata_max_inverse_dt_from_fixed_background(
    const LevelData<FArrayBox> &state, const Box &domain_box,
    const background_t &background, const eos_t &eos,
    const Tensor<1, double> &inverse_dx_U, double dx,
    const std::array<double, CH_SPACEDIM> &center,
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
            compute_max_inverse_dt_from_fixed_background(
                state[dit], grids[dit], background, eos, inverse_dx_U, dx,
                center, atmosphere, recovery_options));
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

template <class eos_t, class background_t>
class FixedBGLevelDataFiniteVolumeOperator
{
  public:
    FixedBGLevelDataFiniteVolumeOperator(
        const eos_t &a_eos, const background_t &a_background, double a_dx,
        const std::array<double, CH_SPACEDIM> &a_center,
        const AtmosphereOptions &a_atmosphere = AtmosphereOptions(),
        double a_limiter_theta = 1.5, bool a_use_reconstruction = true,
        const RecoveryOptions &a_recovery_options = RecoveryOptions(),
        bool a_use_static_metric_sources = true)
        : m_eos(a_eos), m_background(a_background), m_dx(a_dx),
          m_center(a_center), m_atmosphere(a_atmosphere),
          m_limiter_theta(a_limiter_theta),
          m_use_reconstruction(a_use_reconstruction),
          m_recovery_options(a_recovery_options),
          m_use_static_metric_sources(a_use_static_metric_sources)
    {
    }

    const background_t &background() const { return m_background; }

    void compute_rhs(LevelData<FArrayBox> &rhs, LevelData<FArrayBox> &state,
                     const ProblemDomain &domain,
                     const Tensor<1, double> &inverse_dx_U) const
    {
        compute_leveldata_flux_rhs_from_fixed_background(
            rhs, state, domain, m_background, m_eos, inverse_dx_U, m_dx,
            m_center, m_atmosphere, m_limiter_theta,
            m_use_reconstruction, m_use_static_metric_sources,
            m_recovery_options);
    }

    void compute_rhs(LevelData<FArrayBox> &rhs, LevelData<FArrayBox> &state,
                     const Box &domain_box,
                     const Tensor<1, double> &inverse_dx_U) const
    {
        compute_rhs(rhs, state, ProblemDomain(domain_box), inverse_dx_U);
    }

    void compute_rhs(LevelData<FArrayBox> &rhs, LevelData<FArrayBox> &state,
                     const ProblemDomain &domain) const
    {
        compute_rhs(rhs, state, domain, make_inverse_dx());
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

    AtmosphereResetDiagnostics
    recover_primitives(LevelData<FArrayBox> &state) const
    {
        AtmosphereResetDiagnostics local_diagnostics;
        const DisjointBoxLayout &grids = state.disjointBoxLayout();
        DataIterator dit = grids.dataIterator();
        for (dit.begin(); dit.ok(); ++dit)
        {
            local_diagnostics.add(
                recover_primitives_and_apply_atmosphere_from_fixed_background(
                    state[dit], grids[dit], m_eos, m_background, m_dx,
                    m_center, m_recovery_options, m_atmosphere));
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
        return compute_leveldata_max_inverse_dt_from_fixed_background(
            state, domain_box, m_background, m_eos, inverse_dx_U, m_dx,
            m_center, m_atmosphere, m_recovery_options);
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

    double compute_stable_dt(const LevelData<FArrayBox> &state,
                             const Box &domain_box, double cfl) const
    {
        return compute_stable_dt(state, domain_box, make_inverse_dx(), cfl);
    }

    void advance_ssprk2(LevelData<FArrayBox> &state,
                        const ProblemDomain &domain,
                        const Tensor<1, double> &inverse_dx_U,
                        double dt) const
    {
        const DisjointBoxLayout &grids = state.disjointBoxLayout();
        LevelData<FArrayBox> old_state(grids, NUM_VARS, state.ghostVect());
        LevelData<FArrayBox> stage_state(grids, NUM_VARS,
                                         state.ghostVect());
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

    void advance_ssprk2(LevelData<FArrayBox> &state,
                        const ProblemDomain &domain, double dt) const
    {
        advance_ssprk2(state, domain, make_inverse_dx(), dt);
    }

  private:
    Tensor<1, double> make_inverse_dx() const
    {
        Tensor<1, double> inverse_dx_U;
        FOR(dir) { inverse_dx_U[dir] = 1.0 / m_dx; }
        return inverse_dx_U;
    }

    eos_t m_eos;
    background_t m_background;
    double m_dx;
    std::array<double, CH_SPACEDIM> m_center;
    AtmosphereOptions m_atmosphere;
    double m_limiter_theta;
    bool m_use_reconstruction;
    RecoveryOptions m_recovery_options;
    bool m_use_static_metric_sources;
};

template <class background_factory_t>
double fixed_background_log_sqrt_det_time_derivative(
    const background_factory_t &background_factory, double time,
    const IntVect &iv, double dx,
    const std::array<double, CH_SPACEDIM> &center,
    double time_derivative_step)
{
    const double dt = std::max(time_derivative_step, 1.0e-12);
    const auto plus_background = background_factory(time + dt);
    const auto minus_background = background_factory(time - dt);
    const double plus_log_sqrt_det = spatial_metric_log_sqrt_det(
        load_fixed_background_geometry(plus_background, iv, dx, center));
    const double minus_log_sqrt_det = spatial_metric_log_sqrt_det(
        load_fixed_background_geometry(minus_background, iv, dx, center));
    return (plus_log_sqrt_det - minus_log_sqrt_det) / (2.0 * dt);
}

template <class background_factory_t>
void add_metric_volume_time_sources_from_fixed_background_to_conserved_rhs(
    FArrayBox &rhs, const FArrayBox &state, const Box &interior_box,
    const background_factory_t &background_factory, double time, double dx,
    const std::array<double, CH_SPACEDIM> &center,
    double time_derivative_step)
{
    BoxIterator bit(interior_box);
    for (bit.begin(); bit.ok(); ++bit)
    {
        const IntVect iv = bit();
        const double d_log_sqrt_det_dt =
            fixed_background_log_sqrt_det_time_derivative(
                background_factory, time, iv, dx, center,
                time_derivative_step);
        const auto source = compute_metric_volume_time_source_terms(
            load_conserved(state, iv), d_log_sqrt_det_dt);
        rhs(iv, c_GRHD_D) += source.D;
        rhs(iv, c_GRHD_tau) += source.tau;
        FOR(momentum_dir)
        {
            rhs(iv, c_GRHD_S1 + momentum_dir) += source.S_L[momentum_dir];
        }
    }
}

template <class background_factory_t>
void add_leveldata_metric_volume_time_sources_from_fixed_background(
    LevelData<FArrayBox> &rhs, const LevelData<FArrayBox> &state,
    const Box &domain_box, const background_factory_t &background_factory,
    double time, double dx, const std::array<double, CH_SPACEDIM> &center,
    double time_derivative_step)
{
    (void)domain_box;
    const DisjointBoxLayout &grids = state.disjointBoxLayout();
    DataIterator dit = grids.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        add_metric_volume_time_sources_from_fixed_background_to_conserved_rhs(
            rhs[dit], state[dit], grids[dit], background_factory, time, dx,
            center, time_derivative_step);
    }
}

template <class eos_t, class background_factory_t>
class TimeDependentFixedBGLevelDataFiniteVolumeOperator
{
  public:
    TimeDependentFixedBGLevelDataFiniteVolumeOperator(
        const eos_t &a_eos,
        const background_factory_t &a_background_factory, double a_dx,
        const std::array<double, CH_SPACEDIM> &a_center,
        const AtmosphereOptions &a_atmosphere = AtmosphereOptions(),
        double a_limiter_theta = 1.5, bool a_use_reconstruction = true,
        const RecoveryOptions &a_recovery_options = RecoveryOptions(),
        bool a_use_static_metric_sources = true,
        bool a_use_metric_volume_time_sources = true,
        double a_metric_time_derivative_step = 1.0e-3)
        : m_eos(a_eos), m_background_factory(a_background_factory),
          m_dx(a_dx), m_center(a_center), m_atmosphere(a_atmosphere),
          m_limiter_theta(a_limiter_theta),
          m_use_reconstruction(a_use_reconstruction),
          m_recovery_options(a_recovery_options),
          m_use_static_metric_sources(a_use_static_metric_sources),
          m_use_metric_volume_time_sources(a_use_metric_volume_time_sources),
          m_metric_time_derivative_step(a_metric_time_derivative_step)
    {
    }

    void compute_rhs(LevelData<FArrayBox> &rhs, LevelData<FArrayBox> &state,
                     const ProblemDomain &domain,
                     const Tensor<1, double> &inverse_dx_U,
                     double time) const
    {
        const auto background = m_background_factory(time);
        compute_leveldata_flux_rhs_from_fixed_background(
            rhs, state, domain, background, m_eos, inverse_dx_U, m_dx,
            m_center, m_atmosphere, m_limiter_theta,
            m_use_reconstruction, m_use_static_metric_sources,
            m_recovery_options);

        if (m_use_metric_volume_time_sources)
        {
            add_leveldata_metric_volume_time_sources_from_fixed_background(
                rhs, state, domain.domainBox(), m_background_factory, time,
                m_dx, m_center, m_metric_time_derivative_step);
        }
    }

    void compute_rhs(LevelData<FArrayBox> &rhs, LevelData<FArrayBox> &state,
                     const ProblemDomain &domain, double time) const
    {
        compute_rhs(rhs, state, domain, make_inverse_dx(), time);
    }

    AtmosphereResetDiagnostics recover_primitives(
        LevelData<FArrayBox> &state, double time) const
    {
        const auto background = m_background_factory(time);
        AtmosphereResetDiagnostics local_diagnostics;
        const DisjointBoxLayout &grids = state.disjointBoxLayout();
        DataIterator dit = grids.dataIterator();
        for (dit.begin(); dit.ok(); ++dit)
        {
            local_diagnostics.add(
                recover_primitives_and_apply_atmosphere_from_fixed_background(
                    state[dit], grids[dit], m_eos, background, m_dx,
                    m_center, m_recovery_options, m_atmosphere));
        }
        return reduce_atmosphere_reset_diagnostics(local_diagnostics);
    }

    void update_conserved(LevelData<FArrayBox> &state,
                          const LevelData<FArrayBox> &rhs, double dt,
                          double recovery_time,
                          bool recover_primitives_after_update = true) const
    {
        const DisjointBoxLayout &grids = state.disjointBoxLayout();
        DataIterator dit = grids.dataIterator();
        for (dit.begin(); dit.ok(); ++dit)
            add_scaled_conserved_rhs(state[dit], rhs[dit], grids[dit], dt);

        if (recover_primitives_after_update)
            recover_primitives(state, recovery_time);
    }

    double compute_max_inverse_dt(
        const LevelData<FArrayBox> &state, const Box &domain_box,
        const Tensor<1, double> &inverse_dx_U, double time) const
    {
        const auto background = m_background_factory(time);
        return compute_leveldata_max_inverse_dt_from_fixed_background(
            state, domain_box, background, m_eos, inverse_dx_U, m_dx,
            m_center, m_atmosphere, m_recovery_options);
    }

    double compute_stable_dt(const LevelData<FArrayBox> &state,
                             const Box &domain_box,
                             const Tensor<1, double> &inverse_dx_U,
                             double cfl, double time) const
    {
        if (cfl <= 0.0)
            throw std::invalid_argument("GRHD CFL number must be positive");

        const double max_inverse_dt =
            compute_max_inverse_dt(state, domain_box, inverse_dx_U, time);
        return cfl / std::max(max_inverse_dt, 1.0e-14);
    }

    void advance_ssprk2(LevelData<FArrayBox> &state,
                        const ProblemDomain &domain,
                        const Tensor<1, double> &inverse_dx_U,
                        double time, double dt) const
    {
        const DisjointBoxLayout &grids = state.disjointBoxLayout();
        LevelData<FArrayBox> old_state(grids, NUM_VARS, state.ghostVect());
        LevelData<FArrayBox> stage_state(grids, NUM_VARS,
                                         state.ghostVect());
        LevelData<FArrayBox> rhs_initial(grids, NUM_VARS, IntVect::Zero);
        LevelData<FArrayBox> rhs_stage(grids, NUM_VARS, IntVect::Zero);

        state.copyTo(old_state);
        state.copyTo(stage_state);
        compute_rhs(rhs_initial, state, domain, inverse_dx_U, time);
        update_conserved(stage_state, rhs_initial, dt, time + dt);
        compute_rhs(rhs_stage, stage_state, domain, inverse_dx_U,
                    time + dt);

        DataIterator dit = grids.dataIterator();
        for (dit.begin(); dit.ok(); ++dit)
        {
            combine_ssprk2_conserved_state(state[dit], old_state[dit],
                                           stage_state[dit], rhs_stage[dit],
                                           grids[dit], dt);
        }
        recover_primitives(state, time + dt);
    }

  private:
    Tensor<1, double> make_inverse_dx() const
    {
        Tensor<1, double> inverse_dx_U;
        FOR(dir) { inverse_dx_U[dir] = 1.0 / m_dx; }
        return inverse_dx_U;
    }

    eos_t m_eos;
    background_factory_t m_background_factory;
    double m_dx;
    std::array<double, CH_SPACEDIM> m_center;
    AtmosphereOptions m_atmosphere;
    double m_limiter_theta;
    bool m_use_reconstruction;
    RecoveryOptions m_recovery_options;
    bool m_use_static_metric_sources;
    bool m_use_metric_volume_time_sources;
    double m_metric_time_derivative_step;
};

} // namespace GRHD

#endif /* GRHD_FIXED_BG_FARRAYBOX_FINITEVOLUME_HPP_ */
