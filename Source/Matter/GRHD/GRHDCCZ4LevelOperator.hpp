/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef GRHD_CCZ4_LEVEL_OPERATOR_HPP_
#define GRHD_CCZ4_LEVEL_OPERATOR_HPP_

#include "BoxLoops.hpp"
#include "FArrayBoxFiniteVolume.hpp"
#include "FourthOrderDerivatives.hpp"
#include "GRHDMatter.hpp"
#include "IdealGasEOS.hpp"
#include "MatterCCZ4RHS.hpp"
#include "MovingPunctureGauge.hpp"
#include "ProblemDomain.H"
#include "Tensor.hpp"

#include "UsingNamespace.H"

namespace GRHD
{
//! Coupled GRHD + matter-CCZ4 RHS helper for GRChombo level call sites.
/*!
 * GRHD uses a nonlocal finite-volume update for the conserved fluid variables,
 * while MatterCCZ4RHS supplies the local stress-energy source terms in the NR
 * variables. This wrapper composes those two pieces on the same level state:
 * first the MatterCCZ4 RHS is written for all variables, then the finite-volume
 * GRHD RHS is added into the conserved-fluid slots.
 */
template <class gauge_t = MovingPunctureGauge,
          class deriv_t = FourthOrderDerivatives>
class GRHDCCZ4LevelOperator
{
  public:
    using params_t = CCZ4_params_t<typename gauge_t::params_t>;

    GRHDCCZ4LevelOperator(
        const IdealGasEOS &a_eos, const params_t &a_ccz4_params, double a_dx,
        double a_sigma, int a_formulation = CCZ4RHS<>::USE_CCZ4,
        double a_G_Newton = 1.0,
        const AtmosphereOptions &a_atmosphere = AtmosphereOptions(),
        double a_limiter_theta = 1.5, bool a_use_reconstruction = true,
        const RecoveryOptions &a_recovery_options = RecoveryOptions(),
        bool a_use_static_metric_sources = true)
        : m_matter(a_eos, a_atmosphere),
          m_finite_volume_operator(a_eos, a_atmosphere, a_limiter_theta,
                                   a_use_reconstruction, a_recovery_options,
                                   a_use_static_metric_sources),
          m_ccz4_params(a_ccz4_params), m_dx(a_dx), m_sigma(a_sigma),
          m_formulation(a_formulation), m_G_Newton(a_G_Newton)
    {
    }

    void compute_matter_ccz4_rhs(LevelData<FArrayBox> &rhs,
                                 LevelData<FArrayBox> &state) const
    {
        MatterCCZ4RHS<GRHDMatter, gauge_t, deriv_t> matter_ccz4_rhs(
            m_matter, m_ccz4_params, m_dx, m_sigma, m_formulation, m_G_Newton);

        BoxLoops::loop(matter_ccz4_rhs, state, rhs, EXCLUDE_GHOST_CELLS,
                       disable_simd());
    }

    void add_grhd_conserved_rhs(LevelData<FArrayBox> &rhs,
                                LevelData<FArrayBox> &state,
                                const ProblemDomain &domain,
                                const Tensor<1, double> &inverse_dx_U) const
    {
        m_finite_volume_operator.add_conserved_rhs_to(rhs, state, domain,
                                                      inverse_dx_U);
    }

    void compute_rhs(LevelData<FArrayBox> &rhs, LevelData<FArrayBox> &state,
                     const ProblemDomain &domain,
                     const Tensor<1, double> &inverse_dx_U) const
    {
        compute_matter_ccz4_rhs(rhs, state);
        add_grhd_conserved_rhs(rhs, state, domain, inverse_dx_U);
    }

    void compute_rhs(LevelData<FArrayBox> &rhs, LevelData<FArrayBox> &state,
                     const ProblemDomain &domain, double dx) const
    {
        compute_rhs(rhs, state, domain, make_inverse_dx(dx));
    }

    double compute_stable_dt(const LevelData<FArrayBox> &state,
                             const ProblemDomain &domain,
                             const Tensor<1, double> &inverse_dx_U,
                             double cfl) const
    {
        return m_finite_volume_operator.compute_stable_dt(
            state, domain.domainBox(), inverse_dx_U, cfl);
    }

    double compute_stable_dt(const LevelData<FArrayBox> &state,
                             const ProblemDomain &domain, double dx,
                             double cfl) const
    {
        return compute_stable_dt(state, domain, make_inverse_dx(dx),
                                 cfl);
    }

    AtmosphereResetDiagnostics recover_primitives(LevelData<FArrayBox> &state) const
    {
        return m_finite_volume_operator.recover_primitives(state);
    }

  private:
    Tensor<1, double> make_inverse_dx(double dx) const
    {
        Tensor<1, double> inverse_dx_U;
        FOR(dir) { inverse_dx_U[dir] = 1.0 / dx; }
        return inverse_dx_U;
    }

    GRHDMatter m_matter;
    LevelDataFiniteVolumeOperator<IdealGasEOS> m_finite_volume_operator;
    params_t m_ccz4_params;
    double m_dx;
    double m_sigma;
    int m_formulation;
    double m_G_Newton;
};
} // namespace GRHD

#endif /* GRHD_CCZ4_LEVEL_OPERATOR_HPP_ */
