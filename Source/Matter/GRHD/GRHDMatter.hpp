/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo root directory.
 */

#ifndef GRHD_MATTER_HPP_
#define GRHD_MATTER_HPP_

#include "Atmosphere.hpp"
#include "GRHDVars.hpp"
#include "IdealGasEOS.hpp"
#include "StressEnergy.hpp"
#include "Tensor.hpp"
#include "UserVariables.hpp"
#include "VarsTools.hpp"

namespace GRHD
{
//! GRChombo-facing matter adapter for magnetic-field-free GRHD variables.
/*!
 * This supplies the local matter_t interface used by MatterCCZ4RHS and
 * ConstraintsMatter: variable mapping, stress-energy projections, and matter
 * RHS fields. The actual GRHD evolution is finite-volume/HRSC and must be
 * supplied by a separate nonlocal operator, so add_matter_rhs only initializes
 * the GRHD RHS slots to zero.
 */
class GRHDMatter
{
  public:
    explicit GRHDMatter(
        const IdealGasEOS &a_eos,
        const AtmosphereOptions &a_atmosphere = AtmosphereOptions())
        : m_eos(a_eos), m_atmosphere(a_atmosphere)
    {
    }

    template <class data_t> struct Vars
    {
        data_t GRHD_D;
        Tensor<1, data_t> GRHD_S_L;
        data_t GRHD_tau;

        data_t GRHD_rho;
        data_t GRHD_eps;
        data_t GRHD_pressure;
        Tensor<1, data_t> GRHD_velocity_U;

        template <typename mapping_function_t>
        void enum_mapping(mapping_function_t mapping_function)
        {
            VarsTools::define_enum_mapping(mapping_function, c_GRHD_D, GRHD_D);
            VarsTools::define_enum_mapping(
                mapping_function, GRInterval<c_GRHD_S1, c_GRHD_S3>(),
                GRHD_S_L);
            VarsTools::define_enum_mapping(mapping_function, c_GRHD_tau,
                                           GRHD_tau);
            VarsTools::define_enum_mapping(mapping_function, c_GRHD_rho,
                                           GRHD_rho);
            VarsTools::define_enum_mapping(mapping_function, c_GRHD_eps,
                                           GRHD_eps);
            VarsTools::define_enum_mapping(mapping_function, c_GRHD_pressure,
                                           GRHD_pressure);
            VarsTools::define_enum_mapping(
                mapping_function, GRInterval<c_GRHD_v1, c_GRHD_v3>(),
                GRHD_velocity_U);
        }
    };

    template <class data_t> struct Diff2Vars
    {
        template <typename mapping_function_t>
        void enum_mapping(mapping_function_t mapping_function)
        {
            (void)mapping_function;
        }
    };

    template <class data_t, template <typename> class vars_t>
    emtensor_t<data_t> compute_emtensor(
        const vars_t<data_t> &vars, const vars_t<Tensor<1, data_t>> &d1,
        const Tensor<2, data_t> &h_UU,
        const Tensor<3, data_t> &chris_ULL) const
    {
        (void)d1;
        (void)chris_ULL;

        Primitive<data_t> primitive;
        primitive.rho = vars.GRHD_rho;
        primitive.eps = vars.GRHD_eps;
        primitive.pressure = vars.GRHD_pressure;
        primitive.velocity_U = vars.GRHD_velocity_U;

        const auto spatial_metric_LL =
            compute_physical_spatial_metric(vars.h, vars.chi);
        Tensor<2, data_t> spatial_metric_UU;
        FOR(i, j) { spatial_metric_UU[i][j] = vars.chi * h_UU[i][j]; }

        enforce_primitive_floors(primitive, m_eos, spatial_metric_LL,
                                 m_atmosphere);
        return GRHD::compute_emtensor(primitive, m_eos, spatial_metric_LL,
                                      spatial_metric_UU);
    }

    template <class data_t, template <typename> class vars_t,
              template <typename> class diff2_vars_t,
              template <typename> class rhs_vars_t>
    void add_matter_rhs(
        rhs_vars_t<data_t> &total_rhs, const vars_t<data_t> &vars,
        const vars_t<Tensor<1, data_t>> &d1,
        const diff2_vars_t<Tensor<2, data_t>> &d2,
        const vars_t<data_t> &advec) const
    {
        (void)vars;
        (void)d1;
        (void)d2;
        (void)advec;

        total_rhs.GRHD_D = 0.0;
        total_rhs.GRHD_tau = 0.0;
        total_rhs.GRHD_rho = 0.0;
        total_rhs.GRHD_eps = 0.0;
        total_rhs.GRHD_pressure = 0.0;
        FOR(i)
        {
            total_rhs.GRHD_S_L[i] = 0.0;
            total_rhs.GRHD_velocity_U[i] = 0.0;
        }
    }

  private:
    IdealGasEOS m_eos;
    AtmosphereOptions m_atmosphere;
};
} // namespace GRHD

#endif /* GRHD_MATTER_HPP_ */
