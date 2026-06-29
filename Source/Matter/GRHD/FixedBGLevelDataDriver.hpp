/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef GRHD_FIXED_BG_LEVELDATA_DRIVER_HPP_
#define GRHD_FIXED_BG_LEVELDATA_DRIVER_HPP_

#include "FixedBGFArrayBoxFiniteVolume.hpp"
#include "LevelDataDriver.hpp"
#include <algorithm>
#include <array>
#include <stdexcept>

namespace GRHD
{
template <class eos_t, class background_t> class FixedBGLevelDataDriverView
{
  public:
    FixedBGLevelDataDriverView(
        LevelData<FArrayBox> &a_state, const ProblemDomain &a_domain,
        double a_dx, const std::array<double, CH_SPACEDIM> &a_center,
        const eos_t &a_eos, const background_t &a_background,
        const AtmosphereOptions &a_atmosphere = AtmosphereOptions(),
        double a_limiter_theta = 1.5, bool a_use_reconstruction = true,
        const RecoveryOptions &a_recovery_options = RecoveryOptions(),
        bool a_use_static_metric_sources = true)
        : m_state(a_state), m_domain(a_domain), m_dx(a_dx),
          m_center(a_center),
          m_inverse_dx_U(make_uniform_inverse_dx(a_dx)),
          m_operator(a_eos, a_background, a_dx, a_center, a_atmosphere,
                     a_limiter_theta, a_use_reconstruction,
                     a_recovery_options, a_use_static_metric_sources),
          m_time(0.0), m_step(0)
    {
        if (a_dx <= 0.0)
            throw std::invalid_argument(
                "GRHD fixed-background driver dx must be positive");
    }

    FixedBGLevelDataDriverView(
        LevelData<FArrayBox> &a_state, const Box &a_domain_box,
        double a_dx, const std::array<double, CH_SPACEDIM> &a_center,
        const eos_t &a_eos, const background_t &a_background,
        const AtmosphereOptions &a_atmosphere = AtmosphereOptions(),
        double a_limiter_theta = 1.5, bool a_use_reconstruction = true,
        const RecoveryOptions &a_recovery_options = RecoveryOptions(),
        bool a_use_static_metric_sources = true)
        : FixedBGLevelDataDriverView(
              a_state, ProblemDomain(a_domain_box), a_dx, a_center, a_eos,
              a_background, a_atmosphere, a_limiter_theta,
              a_use_reconstruction, a_recovery_options,
              a_use_static_metric_sources)
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
    double dx() const { return m_dx; }
    const std::array<double, CH_SPACEDIM> &center() const
    {
        return m_center;
    }
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

    double compute_coupled_dt(double cfl, double metric_dt) const
    {
        if (metric_dt <= 0.0)
            throw std::invalid_argument(
                "GRHD fixed-background coupled timestep must be positive");
        return std::min(metric_dt, compute_stable_dt(cfl));
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
    double m_dx;
    std::array<double, CH_SPACEDIM> m_center;
    Tensor<1, double> m_inverse_dx_U;
    FixedBGLevelDataFiniteVolumeOperator<eos_t, background_t> m_operator;
    double m_time;
    int m_step;
};

template <class eos_t, class background_t> class FixedBGLevelDataDriver
{
  public:
    FixedBGLevelDataDriver(
        const DisjointBoxLayout &a_grids, const ProblemDomain &a_domain,
        double a_dx, const std::array<double, CH_SPACEDIM> &a_center,
        int a_num_ghosts, const eos_t &a_eos,
        const background_t &a_background,
        const AtmosphereOptions &a_atmosphere = AtmosphereOptions(),
        double a_limiter_theta = 1.5, bool a_use_reconstruction = true,
        const RecoveryOptions &a_recovery_options = RecoveryOptions(),
        bool a_use_static_metric_sources = true)
        : m_state(a_grids, NUM_VARS, a_num_ghosts * IntVect::Unit),
          m_view(m_state, a_domain, a_dx, a_center, a_eos, a_background,
                 a_atmosphere, a_limiter_theta, a_use_reconstruction,
                 a_recovery_options, a_use_static_metric_sources)
    {
    }

    FixedBGLevelDataDriver(
        const DisjointBoxLayout &a_grids, const Box &a_domain_box,
        double a_dx, const std::array<double, CH_SPACEDIM> &a_center,
        int a_num_ghosts, const eos_t &a_eos,
        const background_t &a_background,
        const AtmosphereOptions &a_atmosphere = AtmosphereOptions(),
        double a_limiter_theta = 1.5, bool a_use_reconstruction = true,
        const RecoveryOptions &a_recovery_options = RecoveryOptions(),
        bool a_use_static_metric_sources = true)
        : FixedBGLevelDataDriver(
              a_grids, ProblemDomain(a_domain_box), a_dx, a_center,
              a_num_ghosts, a_eos, a_background, a_atmosphere,
              a_limiter_theta, a_use_reconstruction, a_recovery_options,
              a_use_static_metric_sources)
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
    double dx() const { return m_view.dx(); }
    const std::array<double, CH_SPACEDIM> &center() const
    {
        return m_view.center();
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
    double compute_coupled_dt(double cfl, double metric_dt) const
    {
        return m_view.compute_coupled_dt(cfl, metric_dt);
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
    FixedBGLevelDataDriverView<eos_t, background_t> m_view;
};

template <class eos_t, class background_factory_t>
class TimeDependentFixedBGLevelDataDriverView
{
  public:
    TimeDependentFixedBGLevelDataDriverView(
        LevelData<FArrayBox> &a_state, const ProblemDomain &a_domain,
        double a_dx, const std::array<double, CH_SPACEDIM> &a_center,
        const eos_t &a_eos,
        const background_factory_t &a_background_factory,
        const AtmosphereOptions &a_atmosphere = AtmosphereOptions(),
        double a_limiter_theta = 1.5, bool a_use_reconstruction = true,
        const RecoveryOptions &a_recovery_options = RecoveryOptions(),
        bool a_use_static_metric_sources = true,
        bool a_use_metric_volume_time_sources = true,
        double a_metric_time_derivative_step = 1.0e-3)
        : m_state(a_state), m_domain(a_domain), m_dx(a_dx),
          m_center(a_center),
          m_inverse_dx_U(make_uniform_inverse_dx(a_dx)),
          m_operator(a_eos, a_background_factory, a_dx, a_center,
                     a_atmosphere, a_limiter_theta, a_use_reconstruction,
                     a_recovery_options, a_use_static_metric_sources,
                     a_use_metric_volume_time_sources,
                     a_metric_time_derivative_step),
          m_time(0.0), m_step(0)
    {
        if (a_dx <= 0.0)
            throw std::invalid_argument(
                "GRHD fixed-background driver dx must be positive");
    }

    TimeDependentFixedBGLevelDataDriverView(
        LevelData<FArrayBox> &a_state, const Box &a_domain_box,
        double a_dx, const std::array<double, CH_SPACEDIM> &a_center,
        const eos_t &a_eos,
        const background_factory_t &a_background_factory,
        const AtmosphereOptions &a_atmosphere = AtmosphereOptions(),
        double a_limiter_theta = 1.5, bool a_use_reconstruction = true,
        const RecoveryOptions &a_recovery_options = RecoveryOptions(),
        bool a_use_static_metric_sources = true,
        bool a_use_metric_volume_time_sources = true,
        double a_metric_time_derivative_step = 1.0e-3)
        : TimeDependentFixedBGLevelDataDriverView(
              a_state, ProblemDomain(a_domain_box), a_dx, a_center, a_eos,
              a_background_factory, a_atmosphere, a_limiter_theta,
              a_use_reconstruction, a_recovery_options,
              a_use_static_metric_sources, a_use_metric_volume_time_sources,
              a_metric_time_derivative_step)
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
    double dx() const { return m_dx; }
    const std::array<double, CH_SPACEDIM> &center() const
    {
        return m_center;
    }
    const Tensor<1, double> &inverse_dx_U() const { return m_inverse_dx_U; }
    double time() const { return m_time; }
    int step() const { return m_step; }

    void set_time(double time) { m_time = time; }

    void compute_rhs(LevelData<FArrayBox> &rhs)
    {
        m_operator.compute_rhs(rhs, m_state, m_domain, m_inverse_dx_U,
                               m_time);
    }

    void compute_rhs(LevelData<FArrayBox> &rhs, double time)
    {
        m_operator.compute_rhs(rhs, m_state, m_domain, m_inverse_dx_U,
                               time);
    }

    double compute_stable_dt(double cfl) const
    {
        return m_operator.compute_stable_dt(m_state, m_domain.domainBox(),
                                            m_inverse_dx_U, cfl, m_time);
    }

    double compute_stable_dt(double cfl, double time) const
    {
        return m_operator.compute_stable_dt(m_state, m_domain.domainBox(),
                                            m_inverse_dx_U, cfl, time);
    }

    double compute_coupled_dt(double cfl, double metric_dt) const
    {
        if (metric_dt <= 0.0)
            throw std::invalid_argument(
                "GRHD fixed-background coupled timestep must be positive");
        return std::min(metric_dt, compute_stable_dt(cfl));
    }

    AtmosphereResetDiagnostics recover_primitives()
    {
        return m_operator.recover_primitives(m_state, m_time);
    }

    AtmosphereResetDiagnostics recover_primitives(double time)
    {
        return m_operator.recover_primitives(m_state, time);
    }

    void update_conserved(const LevelData<FArrayBox> &rhs, double dt,
                          bool recover_primitives_after_update = true)
    {
        const double recovery_time = m_time + dt;
        m_operator.update_conserved(m_state, rhs, dt, recovery_time,
                                    recover_primitives_after_update);
        m_time = recovery_time;
        ++m_step;
    }

    void advance_ssprk2(double dt)
    {
        m_operator.advance_ssprk2(m_state, m_domain, m_inverse_dx_U,
                                  m_time, dt);
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
    double m_dx;
    std::array<double, CH_SPACEDIM> m_center;
    Tensor<1, double> m_inverse_dx_U;
    TimeDependentFixedBGLevelDataFiniteVolumeOperator<eos_t,
                                                      background_factory_t>
        m_operator;
    double m_time;
    int m_step;
};

template <class eos_t, class background_factory_t>
class TimeDependentFixedBGLevelDataDriver
{
  public:
    TimeDependentFixedBGLevelDataDriver(
        const DisjointBoxLayout &a_grids, const ProblemDomain &a_domain,
        double a_dx, const std::array<double, CH_SPACEDIM> &a_center,
        int a_num_ghosts, const eos_t &a_eos,
        const background_factory_t &a_background_factory,
        const AtmosphereOptions &a_atmosphere = AtmosphereOptions(),
        double a_limiter_theta = 1.5, bool a_use_reconstruction = true,
        const RecoveryOptions &a_recovery_options = RecoveryOptions(),
        bool a_use_static_metric_sources = true,
        bool a_use_metric_volume_time_sources = true,
        double a_metric_time_derivative_step = 1.0e-3)
        : m_state(a_grids, NUM_VARS, a_num_ghosts * IntVect::Unit),
          m_view(m_state, a_domain, a_dx, a_center, a_eos,
                 a_background_factory, a_atmosphere, a_limiter_theta,
                 a_use_reconstruction, a_recovery_options,
                 a_use_static_metric_sources, a_use_metric_volume_time_sources,
                 a_metric_time_derivative_step)
    {
    }

    TimeDependentFixedBGLevelDataDriver(
        const DisjointBoxLayout &a_grids, const Box &a_domain_box,
        double a_dx, const std::array<double, CH_SPACEDIM> &a_center,
        int a_num_ghosts, const eos_t &a_eos,
        const background_factory_t &a_background_factory,
        const AtmosphereOptions &a_atmosphere = AtmosphereOptions(),
        double a_limiter_theta = 1.5, bool a_use_reconstruction = true,
        const RecoveryOptions &a_recovery_options = RecoveryOptions(),
        bool a_use_static_metric_sources = true,
        bool a_use_metric_volume_time_sources = true,
        double a_metric_time_derivative_step = 1.0e-3)
        : TimeDependentFixedBGLevelDataDriver(
              a_grids, ProblemDomain(a_domain_box), a_dx, a_center,
              a_num_ghosts, a_eos, a_background_factory, a_atmosphere,
              a_limiter_theta, a_use_reconstruction, a_recovery_options,
              a_use_static_metric_sources, a_use_metric_volume_time_sources,
              a_metric_time_derivative_step)
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
    double dx() const { return m_view.dx(); }
    const std::array<double, CH_SPACEDIM> &center() const
    {
        return m_view.center();
    }
    const Tensor<1, double> &inverse_dx_U() const
    {
        return m_view.inverse_dx_U();
    }
    double time() const { return m_view.time(); }
    int step() const { return m_view.step(); }
    void set_time(double time) { m_view.set_time(time); }
    void compute_rhs(LevelData<FArrayBox> &rhs) { m_view.compute_rhs(rhs); }
    void compute_rhs(LevelData<FArrayBox> &rhs, double time)
    {
        m_view.compute_rhs(rhs, time);
    }
    double compute_stable_dt(double cfl) const
    {
        return m_view.compute_stable_dt(cfl);
    }
    double compute_stable_dt(double cfl, double time) const
    {
        return m_view.compute_stable_dt(cfl, time);
    }
    double compute_coupled_dt(double cfl, double metric_dt) const
    {
        return m_view.compute_coupled_dt(cfl, metric_dt);
    }
    AtmosphereResetDiagnostics recover_primitives()
    {
        return m_view.recover_primitives();
    }
    AtmosphereResetDiagnostics recover_primitives(double time)
    {
        return m_view.recover_primitives(time);
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
    TimeDependentFixedBGLevelDataDriverView<eos_t, background_factory_t>
        m_view;
};
} // namespace GRHD

#endif /* GRHD_FIXED_BG_LEVELDATA_DRIVER_HPP_ */
