/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef GRHD_LEVELDATA_DRIVER_HPP_
#define GRHD_LEVELDATA_DRIVER_HPP_

#include "DisjointBoxLayout.H"
#include "FArrayBox.H"
#include "FArrayBoxFiniteVolume.hpp"
#include "LevelData.H"
#include "ProblemDomain.H"
#include "SPMD.H"
#include <array>
#include <cmath>

#include "UsingNamespace.H"

namespace GRHD
{
inline Conserved<double>
compute_leveldata_conserved_sum(const LevelData<FArrayBox> &state)
{
    std::array<double, 5> local_sums = {{0.0, 0.0, 0.0, 0.0, 0.0}};
    const DisjointBoxLayout &grids = state.disjointBoxLayout();
    DataIterator dit = grids.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        const FArrayBox &state_fab = state[dit];
        BoxIterator bit(grids[dit]);
        for (bit.begin(); bit.ok(); ++bit)
        {
            const auto conserved = load_conserved(state_fab, bit());
            local_sums[0] += conserved.D;
            local_sums[1] += conserved.S_L[0];
            local_sums[2] += conserved.S_L[1];
            local_sums[3] += conserved.S_L[2];
            local_sums[4] += conserved.tau;
        }
    }

    std::array<double, 5> global_sums = local_sums;
#ifdef CH_MPI
    MPI_Allreduce(local_sums.data(), global_sums.data(),
                  static_cast<int>(global_sums.size()), MPI_DOUBLE, MPI_SUM,
                  Chombo_MPI::comm);
#endif

    Conserved<double> sum = zero_conserved<double>();
    sum.D = global_sums[0];
    sum.S_L[0] = global_sums[1];
    sum.S_L[1] = global_sums[2];
    sum.S_L[2] = global_sums[3];
    sum.tau = global_sums[4];
    return sum;
}

inline bool conserved_is_finite(const Conserved<double> &conserved)
{
    bool is_finite = std::isfinite(conserved.D) &&
                     std::isfinite(conserved.tau);
    FOR(i) { is_finite = is_finite && std::isfinite(conserved.S_L[i]); }
    return is_finite;
}

inline bool
leveldata_conserved_state_is_finite(const LevelData<FArrayBox> &state)
{
    int local_ok = 1;
    const DisjointBoxLayout &grids = state.disjointBoxLayout();
    DataIterator dit = grids.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        const FArrayBox &state_fab = state[dit];
        BoxIterator bit(grids[dit]);
        for (bit.begin(); bit.ok(); ++bit)
        {
            if (!conserved_is_finite(load_conserved(state_fab, bit())))
                local_ok = 0;
        }
    }

#ifdef CH_MPI
    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN,
                  Chombo_MPI::comm);
    return global_ok == 1;
#else
    return local_ok == 1;
#endif
}

inline Tensor<1, double> make_uniform_inverse_dx(double dx)
{
    Tensor<1, double> inverse_dx_U;
    FOR(dir) { inverse_dx_U[dir] = 1.0 / dx; }
    return inverse_dx_U;
}

template <class eos_t> class LevelDataAMRHooks
{
  public:
    LevelDataAMRHooks(
        const eos_t &a_eos,
        const AtmosphereOptions &a_atmosphere = AtmosphereOptions(),
        double a_limiter_theta = 1.5, bool a_use_reconstruction = true,
        const RecoveryOptions &a_recovery_options = RecoveryOptions())
        : m_operator(a_eos, a_atmosphere, a_limiter_theta,
                     a_use_reconstruction, a_recovery_options)
    {
    }

    void compute_rhs(LevelData<FArrayBox> &rhs, LevelData<FArrayBox> &state,
                     const ProblemDomain &domain,
                     const Tensor<1, double> &inverse_dx_U) const
    {
        m_operator.compute_rhs(rhs, state, domain, inverse_dx_U);
    }

    void compute_rhs(LevelData<FArrayBox> &rhs, LevelData<FArrayBox> &state,
                     const ProblemDomain &domain, double dx) const
    {
        compute_rhs(rhs, state, domain, make_uniform_inverse_dx(dx));
    }

    void add_conserved_rhs_to(LevelData<FArrayBox> &rhs,
                              LevelData<FArrayBox> &state,
                              const ProblemDomain &domain,
                              const Tensor<1, double> &inverse_dx_U) const
    {
        m_operator.add_conserved_rhs_to(rhs, state, domain, inverse_dx_U);
    }

    void add_conserved_rhs_to(LevelData<FArrayBox> &rhs,
                              LevelData<FArrayBox> &state,
                              const ProblemDomain &domain, double dx) const
    {
        add_conserved_rhs_to(rhs, state, domain, make_uniform_inverse_dx(dx));
    }

    double compute_stable_dt(const LevelData<FArrayBox> &state,
                             const ProblemDomain &domain,
                             const Tensor<1, double> &inverse_dx_U,
                             double cfl) const
    {
        return m_operator.compute_stable_dt(state, domain.domainBox(),
                                            inverse_dx_U, cfl);
    }

    double compute_stable_dt(const LevelData<FArrayBox> &state,
                             const ProblemDomain &domain, double dx,
                             double cfl) const
    {
        return compute_stable_dt(state, domain, make_uniform_inverse_dx(dx),
                                 cfl);
    }

    AtmosphereResetDiagnostics recover_primitives(
        LevelData<FArrayBox> &state) const
    {
        return m_operator.recover_primitives(state);
    }

    void update_conserved(LevelData<FArrayBox> &state,
                          const LevelData<FArrayBox> &rhs, double dt,
                          bool recover_primitives_after_update = true) const
    {
        m_operator.update_conserved(state, rhs, dt,
                                    recover_primitives_after_update);
    }

    void advance_ssprk2(LevelData<FArrayBox> &state,
                        const ProblemDomain &domain,
                        const Tensor<1, double> &inverse_dx_U,
                        double dt) const
    {
        m_operator.advance_ssprk2(state, domain, inverse_dx_U, dt);
    }

    void advance_ssprk2(LevelData<FArrayBox> &state,
                        const ProblemDomain &domain, double dx,
                        double dt) const
    {
        advance_ssprk2(state, domain, make_uniform_inverse_dx(dx), dt);
    }

  private:
    LevelDataFiniteVolumeOperator<eos_t> m_operator;
};

template <class eos_t> class LevelDataDriverView
{
  public:
    LevelDataDriverView(
        LevelData<FArrayBox> &a_state, const ProblemDomain &a_domain,
        const Tensor<1, double> &a_inverse_dx_U, const eos_t &a_eos,
        const AtmosphereOptions &a_atmosphere = AtmosphereOptions(),
        double a_limiter_theta = 1.5, bool a_use_reconstruction = true,
        const RecoveryOptions &a_recovery_options = RecoveryOptions())
        : m_state(a_state), m_domain(a_domain),
          m_inverse_dx_U(a_inverse_dx_U),
          m_operator(a_eos, a_atmosphere, a_limiter_theta,
                     a_use_reconstruction, a_recovery_options),
          m_time(0.0), m_step(0)
    {
    }

    LevelDataDriverView(
        LevelData<FArrayBox> &a_state, const Box &a_domain_box,
        const Tensor<1, double> &a_inverse_dx_U, const eos_t &a_eos,
        const AtmosphereOptions &a_atmosphere = AtmosphereOptions(),
        double a_limiter_theta = 1.5, bool a_use_reconstruction = true,
        const RecoveryOptions &a_recovery_options = RecoveryOptions())
        : LevelDataDriverView(a_state, ProblemDomain(a_domain_box),
                              a_inverse_dx_U, a_eos, a_atmosphere,
                              a_limiter_theta, a_use_reconstruction,
                              a_recovery_options)
    {
    }

    LevelData<FArrayBox> &state() { return m_state; }
    const LevelData<FArrayBox> &state() const { return m_state; }

    const DisjointBoxLayout &grids() const
    {
        return m_state.disjointBoxLayout();
    }

    const Box &domain_box() const { return m_domain.domainBox(); }

    const ProblemDomain &problem_domain() const { return m_domain; }

    const Tensor<1, double> &inverse_dx_U() const { return m_inverse_dx_U; }

    double time() const { return m_time; }
    int step() const { return m_step; }

    void compute_rhs(LevelData<FArrayBox> &rhs)
    {
        m_operator.compute_rhs(rhs, m_state, m_domain, m_inverse_dx_U);
    }

    void add_conserved_rhs_to(LevelData<FArrayBox> &rhs)
    {
        m_operator.add_conserved_rhs_to(rhs, m_state, m_domain,
                                        m_inverse_dx_U);
    }

    double compute_stable_dt(double cfl) const
    {
        return m_operator.compute_stable_dt(m_state, m_domain.domainBox(),
                                            m_inverse_dx_U, cfl);
    }

    AtmosphereResetDiagnostics recover_primitives()
    {
        return m_operator.recover_primitives(m_state);
    }

    void update_conserved(const LevelData<FArrayBox> &rhs, double dt,
                          bool recover_primitives_after_update = true)
    {
        m_operator.update_conserved(m_state, rhs, dt,
                                    recover_primitives_after_update);
        m_time += dt;
        ++m_step;
    }

    void advance_ssprk2(double dt)
    {
        m_operator.advance_ssprk2(m_state, m_domain, m_inverse_dx_U, dt);
        m_time += dt;
        ++m_step;
    }

    Conserved<double> conserved_sum() const
    {
        return compute_leveldata_conserved_sum(m_state);
    }

    bool conserved_state_is_finite() const
    {
        return leveldata_conserved_state_is_finite(m_state);
    }

  private:
    LevelData<FArrayBox> &m_state;
    ProblemDomain m_domain;
    Tensor<1, double> m_inverse_dx_U;
    LevelDataFiniteVolumeOperator<eos_t> m_operator;
    double m_time;
    int m_step;
};

template <class eos_t> class LevelDataDriver
{
  public:
    LevelDataDriver(
        const DisjointBoxLayout &a_grids, const ProblemDomain &a_domain,
        const Tensor<1, double> &a_inverse_dx_U, int a_num_ghosts,
        const eos_t &a_eos,
        const AtmosphereOptions &a_atmosphere = AtmosphereOptions(),
        double a_limiter_theta = 1.5, bool a_use_reconstruction = true,
        const RecoveryOptions &a_recovery_options = RecoveryOptions())
        : m_state(a_grids, NUM_VARS, a_num_ghosts * IntVect::Unit),
          m_view(m_state, a_domain, a_inverse_dx_U, a_eos, a_atmosphere,
                 a_limiter_theta, a_use_reconstruction, a_recovery_options)
    {
    }

    LevelDataDriver(
        const DisjointBoxLayout &a_grids, const Box &a_domain_box,
        const Tensor<1, double> &a_inverse_dx_U, int a_num_ghosts,
        const eos_t &a_eos,
        const AtmosphereOptions &a_atmosphere = AtmosphereOptions(),
        double a_limiter_theta = 1.5, bool a_use_reconstruction = true,
        const RecoveryOptions &a_recovery_options = RecoveryOptions())
        : LevelDataDriver(a_grids, ProblemDomain(a_domain_box), a_inverse_dx_U,
                          a_num_ghosts, a_eos, a_atmosphere, a_limiter_theta,
                          a_use_reconstruction, a_recovery_options)
    {
    }

    LevelData<FArrayBox> &state() { return m_view.state(); }
    const LevelData<FArrayBox> &state() const { return m_view.state(); }

    const DisjointBoxLayout &grids() const { return m_view.grids(); }

    const Box &domain_box() const { return m_view.domain_box(); }

    const ProblemDomain &problem_domain() const
    {
        return m_view.problem_domain();
    }

    const Tensor<1, double> &inverse_dx_U() const
    {
        return m_view.inverse_dx_U();
    }

    double time() const { return m_view.time(); }
    int step() const { return m_view.step(); }

    void compute_rhs(LevelData<FArrayBox> &rhs) { m_view.compute_rhs(rhs); }

    void add_conserved_rhs_to(LevelData<FArrayBox> &rhs)
    {
        m_view.add_conserved_rhs_to(rhs);
    }

    double compute_stable_dt(double cfl) const
    {
        return m_view.compute_stable_dt(cfl);
    }

    AtmosphereResetDiagnostics recover_primitives()
    {
        return m_view.recover_primitives();
    }

    void update_conserved(const LevelData<FArrayBox> &rhs, double dt,
                          bool recover_primitives_after_update = true)
    {
        m_view.update_conserved(rhs, dt, recover_primitives_after_update);
    }

    void advance_ssprk2(double dt) { m_view.advance_ssprk2(dt); }

    Conserved<double> conserved_sum() const { return m_view.conserved_sum(); }

    bool conserved_state_is_finite() const
    {
        return m_view.conserved_state_is_finite();
    }

  private:
    LevelData<FArrayBox> m_state;
    LevelDataDriverView<eos_t> m_view;
};
} // namespace GRHD

#endif /* GRHD_LEVELDATA_DRIVER_HPP_ */
