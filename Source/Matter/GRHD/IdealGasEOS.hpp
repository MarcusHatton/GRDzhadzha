/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef GRHD_IDEALGASEOS_HPP_
#define GRHD_IDEALGASEOS_HPP_

//! Gamma-law ideal-gas equation of state for GRHD.
class IdealGasEOS
{
  public:
    struct params_t
    {
        double adiabatic_index = 5.0 / 3.0;
    };

    explicit IdealGasEOS(const params_t &a_params)
        : m_adiabatic_index(a_params.adiabatic_index)
    {
    }

    explicit IdealGasEOS(double a_adiabatic_index)
        : m_adiabatic_index(a_adiabatic_index)
    {
    }

    double adiabatic_index() const { return m_adiabatic_index; }

    template <class data_t>
    data_t compute_pressure(const data_t &rho, const data_t &eps) const
    {
        return (m_adiabatic_index - 1.0) * rho * eps;
    }

    template <class data_t>
    data_t compute_specific_enthalpy(const data_t &rho, const data_t &eps,
                                     const data_t &pressure) const
    {
        return 1.0 + eps + pressure / rho;
    }

    template <class data_t>
    data_t compute_sound_speed_squared(const data_t &rho, const data_t &eps,
                                       const data_t &pressure) const
    {
        const data_t h = compute_specific_enthalpy(rho, eps, pressure);
        return m_adiabatic_index * pressure / (rho * h);
    }

  private:
    double m_adiabatic_index;
};

#endif /* GRHD_IDEALGASEOS_HPP_ */
