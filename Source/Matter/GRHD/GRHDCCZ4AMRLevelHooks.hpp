/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef GRHD_CCZ4_AMR_LEVEL_HOOKS_HPP_
#define GRHD_CCZ4_AMR_LEVEL_HOOKS_HPP_

#include "ComputePack.hpp"
#include "GRHDCCZ4LevelOperator.hpp"
#include "PositiveChiAndAlpha.hpp"
#include "TraceARemoval.hpp"
#include <algorithm>
#include <stdexcept>

namespace GRHD
{
//! GRAMRLevel-shaped hooks for a coupled GRHD + CCZ4 single-level update.
/*!
 * A concrete GRHD level can delegate its overrides to this class:
 *
 * - `specific_eval_rhs` enforces the usual CCZ4 algebraic conditions, computes
 *   the matter-coupled CCZ4 RHS, and adds the finite-volume GRHD conserved RHS.
 * - `specific_update_ode` enforces trace-free A_ij after `soln += dt * rhs`
 *   and refreshes GRHD primitive variables from the updated conserved state.
 * - `compute_stable_dt` exposes the GRHD CFL timestep bound for use by a
 *   GRAMRLevel timestep override or a future combined NR/hydro timestep policy.
 * - `compute_coupled_dt` returns the conservative minimum of an externally
 *   supplied metric/AMR timestep and the GRHD CFL timestep.
 */
template <class gauge_t = MovingPunctureGauge,
          class deriv_t = FourthOrderDerivatives>
class GRHDCCZ4AMRLevelHooks
{
  public:
    using params_t = CCZ4_params_t<typename gauge_t::params_t>;

    GRHDCCZ4AMRLevelHooks(
        const IdealGasEOS &a_eos, const params_t &a_ccz4_params, double a_dx,
        double a_sigma, int a_formulation = CCZ4RHS<>::USE_CCZ4,
        double a_G_Newton = 1.0,
        const AtmosphereOptions &a_atmosphere = AtmosphereOptions(),
        double a_limiter_theta = 1.5, bool a_use_reconstruction = true,
        const RecoveryOptions &a_recovery_options = RecoveryOptions(),
        double a_min_chi = 1.0e-4, double a_min_lapse = 1.0e-4,
        bool a_use_static_metric_sources = true)
        : m_level_operator(a_eos, a_ccz4_params, a_dx, a_sigma,
                           a_formulation, a_G_Newton, a_atmosphere,
                           a_limiter_theta, a_use_reconstruction,
                           a_recovery_options,
                           a_use_static_metric_sources),
          m_min_chi(a_min_chi), m_min_lapse(a_min_lapse)
    {
    }

    void specific_eval_rhs(LevelData<FArrayBox> &soln,
                           LevelData<FArrayBox> &rhs,
                           const ProblemDomain &domain,
                           const Tensor<1, double> &inverse_dx_U) const
    {
        BoxLoops::loop(make_compute_pack(TraceARemoval(),
                                         PositiveChiAndAlpha(m_min_chi,
                                                             m_min_lapse)),
                       soln, soln, INCLUDE_GHOST_CELLS);
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

    AtmosphereResetDiagnostics specific_update_ode(LevelData<FArrayBox> &soln) const
    {
        BoxLoops::loop(TraceARemoval(), soln, soln, INCLUDE_GHOST_CELLS);
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
                             const ProblemDomain &domain,
                             const Tensor<1, double> &inverse_dx_U,
                             double cfl) const
    {
        return m_level_operator.compute_stable_dt(state, domain, inverse_dx_U,
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
                "GRHD coupled metric/AMR timestep must be positive");

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

    double compute_coupled_dt_from_multiplier(
        const LevelData<FArrayBox> &state, const ProblemDomain &domain,
        double dx, double cfl, double dt_multiplier) const
    {
        if (dt_multiplier <= 0.0)
            throw std::invalid_argument(
                "GRHD coupled timestep multiplier must be positive");

        return compute_coupled_dt(state, domain, dx, cfl, dx * dt_multiplier);
    }

  private:
    Tensor<1, double> make_inverse_dx(double dx) const
    {
        Tensor<1, double> inverse_dx_U;
        FOR(dir) { inverse_dx_U[dir] = 1.0 / dx; }
        return inverse_dx_U;
    }

    GRHDCCZ4LevelOperator<gauge_t, deriv_t> m_level_operator;
    double m_min_chi;
    double m_min_lapse;
};
} // namespace GRHD

#endif /* GRHD_CCZ4_AMR_LEVEL_HOOKS_HPP_ */
