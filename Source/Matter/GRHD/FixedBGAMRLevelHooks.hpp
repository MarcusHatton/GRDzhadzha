/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef GRHD_FIXED_BG_AMR_LEVEL_HOOKS_HPP_
#define GRHD_FIXED_BG_AMR_LEVEL_HOOKS_HPP_

#include "FixedBGFArrayBoxFiniteVolume.hpp"
#include <algorithm>
#include <stdexcept>

namespace GRHD
{
//! GRAMRLevel-shaped hooks for GRHD on a static analytic fixed background.
/*!
 * A concrete fixed-background GRHD level can delegate its RHS, post-update
 * primitive recovery, CFL estimate, and standalone SSPRK2 advance to this
 * class. The underlying finite-volume work is handled by
 * `FixedBGLevelDataFiniteVolumeOperator`.
 */
template <class eos_t, class background_t> class FixedBGAMRLevelHooks
{
  public:
    FixedBGAMRLevelHooks(
        const eos_t &a_eos, const background_t &a_background, double a_dx,
        const std::array<double, CH_SPACEDIM> &a_center,
        const AtmosphereOptions &a_atmosphere = AtmosphereOptions(),
        double a_limiter_theta = 1.5, bool a_use_reconstruction = true,
        const RecoveryOptions &a_recovery_options = RecoveryOptions(),
        bool a_use_static_metric_sources = true)
        : m_level_operator(a_eos, a_background, a_dx, a_center,
                           a_atmosphere, a_limiter_theta,
                           a_use_reconstruction, a_recovery_options,
                           a_use_static_metric_sources),
          m_dx(a_dx)
    {
    }

    void specific_eval_rhs(LevelData<FArrayBox> &soln,
                           LevelData<FArrayBox> &rhs,
                           const ProblemDomain &domain,
                           const Tensor<1, double> &inverse_dx_U) const
    {
        m_level_operator.compute_rhs(rhs, soln, domain, inverse_dx_U);
    }

    void specific_eval_rhs(LevelData<FArrayBox> &soln,
                           LevelData<FArrayBox> &rhs,
                           const ProblemDomain &domain, double dx) const
    {
        specific_eval_rhs(soln, rhs, domain, make_inverse_dx(dx));
    }

    void specific_eval_rhs(LevelData<FArrayBox> &soln,
                           LevelData<FArrayBox> &rhs,
                           const ProblemDomain &domain,
                           const Tensor<1, double> &inverse_dx_U,
                           double time) const
    {
        (void)time;
        specific_eval_rhs(soln, rhs, domain, inverse_dx_U);
    }

    void specific_eval_rhs(LevelData<FArrayBox> &soln,
                           LevelData<FArrayBox> &rhs,
                           const ProblemDomain &domain, double dx,
                           double time) const
    {
        specific_eval_rhs(soln, rhs, domain, make_inverse_dx(dx), time);
    }

    AtmosphereResetDiagnostics specific_update_ode(
        LevelData<FArrayBox> &soln) const
    {
        return m_level_operator.recover_primitives(soln);
    }

    AtmosphereResetDiagnostics specific_update_ode(
        LevelData<FArrayBox> &soln, const LevelData<FArrayBox> &rhs,
        double dt) const
    {
        (void)rhs;
        (void)dt;
        return specific_update_ode(soln);
    }

    double compute_stable_dt(const LevelData<FArrayBox> &state,
                             const Box &domain_box,
                             const Tensor<1, double> &inverse_dx_U,
                             double cfl) const
    {
        return m_level_operator.compute_stable_dt(state, domain_box,
                                                  inverse_dx_U, cfl);
    }

    double compute_stable_dt(const LevelData<FArrayBox> &state,
                             const ProblemDomain &domain,
                             const Tensor<1, double> &inverse_dx_U,
                             double cfl) const
    {
        return compute_stable_dt(state, domain.domainBox(), inverse_dx_U,
                                 cfl);
    }

    double compute_stable_dt(const LevelData<FArrayBox> &state,
                             const ProblemDomain &domain, double dx,
                             double cfl) const
    {
        return compute_stable_dt(state, domain, make_inverse_dx(dx), cfl);
    }

    double compute_coupled_dt(const LevelData<FArrayBox> &state,
                              const ProblemDomain &domain,
                              const Tensor<1, double> &inverse_dx_U,
                              double cfl, double metric_dt) const
    {
        if (metric_dt <= 0.0)
            throw std::invalid_argument(
                "GRHD fixed-background coupled timestep must be positive");
        const double hydro_dt =
            compute_stable_dt(state, domain, inverse_dx_U, cfl);
        return std::min(metric_dt, hydro_dt);
    }

    double compute_coupled_dt(const LevelData<FArrayBox> &state,
                              const ProblemDomain &domain, double dx,
                              double cfl, double metric_dt) const
    {
        return compute_coupled_dt(state, domain, make_inverse_dx(dx), cfl,
                                  metric_dt);
    }

    void update_conserved(LevelData<FArrayBox> &state,
                          const LevelData<FArrayBox> &rhs, double dt,
                          bool recover_primitives_after_update = true) const
    {
        m_level_operator.update_conserved(state, rhs, dt,
                                          recover_primitives_after_update);
    }

    void advance_ssprk2(LevelData<FArrayBox> &state,
                        const ProblemDomain &domain,
                        const Tensor<1, double> &inverse_dx_U,
                        double dt) const
    {
        m_level_operator.advance_ssprk2(state, domain, inverse_dx_U, dt);
    }

    void advance_ssprk2(LevelData<FArrayBox> &state,
                        const ProblemDomain &domain, double dt) const
    {
        advance_ssprk2(state, domain, make_inverse_dx(m_dx), dt);
    }

  private:
    Tensor<1, double> make_inverse_dx(double dx) const
    {
        Tensor<1, double> inverse_dx_U;
        FOR(dir) { inverse_dx_U[dir] = 1.0 / dx; }
        return inverse_dx_U;
    }

    FixedBGLevelDataFiniteVolumeOperator<eos_t, background_t>
        m_level_operator;
    double m_dx;
};

//! GRAMRLevel-shaped hooks for GRHD on a time-dependent analytic background.
template <class eos_t, class background_factory_t>
class TimeDependentFixedBGAMRLevelHooks
{
  public:
    TimeDependentFixedBGAMRLevelHooks(
        const eos_t &a_eos,
        const background_factory_t &a_background_factory, double a_dx,
        const std::array<double, CH_SPACEDIM> &a_center,
        const AtmosphereOptions &a_atmosphere = AtmosphereOptions(),
        double a_limiter_theta = 1.5, bool a_use_reconstruction = true,
        const RecoveryOptions &a_recovery_options = RecoveryOptions(),
        bool a_use_static_metric_sources = true,
        bool a_use_metric_volume_time_sources = true,
        double a_metric_time_derivative_step = 1.0e-3)
        : m_level_operator(a_eos, a_background_factory, a_dx, a_center,
                           a_atmosphere, a_limiter_theta,
                           a_use_reconstruction, a_recovery_options,
                           a_use_static_metric_sources,
                           a_use_metric_volume_time_sources,
                           a_metric_time_derivative_step),
          m_dx(a_dx)
    {
    }

    void specific_eval_rhs(LevelData<FArrayBox> &soln,
                           LevelData<FArrayBox> &rhs,
                           const ProblemDomain &domain,
                           const Tensor<1, double> &inverse_dx_U,
                           double time) const
    {
        m_level_operator.compute_rhs(rhs, soln, domain, inverse_dx_U, time);
    }

    void specific_eval_rhs(LevelData<FArrayBox> &soln,
                           LevelData<FArrayBox> &rhs,
                           const ProblemDomain &domain, double dx,
                           double time) const
    {
        specific_eval_rhs(soln, rhs, domain, make_inverse_dx(dx), time);
    }

    AtmosphereResetDiagnostics specific_update_ode(
        LevelData<FArrayBox> &soln, double time) const
    {
        return m_level_operator.recover_primitives(soln, time);
    }

    AtmosphereResetDiagnostics specific_update_ode(
        LevelData<FArrayBox> &soln, const LevelData<FArrayBox> &rhs,
        double dt, double time) const
    {
        (void)rhs;
        (void)dt;
        return specific_update_ode(soln, time);
    }

    double compute_stable_dt(const LevelData<FArrayBox> &state,
                             const Box &domain_box,
                             const Tensor<1, double> &inverse_dx_U,
                             double cfl, double time) const
    {
        return m_level_operator.compute_stable_dt(
            state, domain_box, inverse_dx_U, cfl, time);
    }

    double compute_stable_dt(const LevelData<FArrayBox> &state,
                             const ProblemDomain &domain,
                             const Tensor<1, double> &inverse_dx_U,
                             double cfl, double time) const
    {
        return compute_stable_dt(state, domain.domainBox(), inverse_dx_U,
                                 cfl, time);
    }

    double compute_stable_dt(const LevelData<FArrayBox> &state,
                             const ProblemDomain &domain, double dx,
                             double cfl, double time) const
    {
        return compute_stable_dt(state, domain, make_inverse_dx(dx), cfl,
                                 time);
    }

    double compute_coupled_dt(const LevelData<FArrayBox> &state,
                              const ProblemDomain &domain,
                              const Tensor<1, double> &inverse_dx_U,
                              double cfl, double metric_dt,
                              double time) const
    {
        if (metric_dt <= 0.0)
            throw std::invalid_argument(
                "GRHD time-dependent fixed-background coupled timestep must be positive");
        const double hydro_dt =
            compute_stable_dt(state, domain, inverse_dx_U, cfl, time);
        return std::min(metric_dt, hydro_dt);
    }

    double compute_coupled_dt(const LevelData<FArrayBox> &state,
                              const ProblemDomain &domain, double dx,
                              double cfl, double metric_dt,
                              double time) const
    {
        return compute_coupled_dt(state, domain, make_inverse_dx(dx), cfl,
                                  metric_dt, time);
    }

    void update_conserved(LevelData<FArrayBox> &state,
                          const LevelData<FArrayBox> &rhs, double dt,
                          double recovery_time,
                          bool recover_primitives_after_update = true) const
    {
        m_level_operator.update_conserved(state, rhs, dt, recovery_time,
                                          recover_primitives_after_update);
    }

    void advance_ssprk2(LevelData<FArrayBox> &state,
                        const ProblemDomain &domain,
                        const Tensor<1, double> &inverse_dx_U,
                        double time, double dt) const
    {
        m_level_operator.advance_ssprk2(state, domain, inverse_dx_U, time,
                                        dt);
    }

    void advance_ssprk2(LevelData<FArrayBox> &state,
                        const ProblemDomain &domain, double time,
                        double dt) const
    {
        advance_ssprk2(state, domain, make_inverse_dx(m_dx), time, dt);
    }

  private:
    Tensor<1, double> make_inverse_dx(double dx) const
    {
        Tensor<1, double> inverse_dx_U;
        FOR(dir) { inverse_dx_U[dir] = 1.0 / dx; }
        return inverse_dx_U;
    }

    TimeDependentFixedBGLevelDataFiniteVolumeOperator<eos_t,
                                                      background_factory_t>
        m_level_operator;
    double m_dx;
};
} // namespace GRHD

#endif /* GRHD_FIXED_BG_AMR_LEVEL_HOOKS_HPP_ */
