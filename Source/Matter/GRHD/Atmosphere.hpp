/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef GRHD_ATMOSPHERE_HPP_
#define GRHD_ATMOSPHERE_HPP_

#include "GRHDVars.hpp"
#include "Valencia.hpp"
#include <algorithm>
#include <cmath>

namespace GRHD
{
struct AtmosphereOptions
{
    double rho_floor = 1.0e-12;
    double pressure_floor = 1.0e-12;
    double max_velocity_squared = 1.0 - 1.0e-12;
    double metric_source_atmosphere_factor = 1.0 + 1.0e-12;
};

template <class data_t, class eos_t>
void enforce_primitive_floors(Primitive<data_t> &primitive, const eos_t &eos,
                              const Tensor<2, data_t> &spatial_metric_LL,
                              const AtmosphereOptions &options =
                                  AtmosphereOptions())
{
    primitive.rho = std::max(primitive.rho, data_t(options.rho_floor));
    primitive.pressure =
        std::max(primitive.pressure, data_t(options.pressure_floor));

    const data_t v_squared =
        compute_velocity_squared(primitive.velocity_U, spatial_metric_LL);
    if (v_squared >= data_t(options.max_velocity_squared))
    {
        const data_t scale =
            std::sqrt(data_t(options.max_velocity_squared) / v_squared);
        FOR(i) { primitive.velocity_U[i] *= scale; }
    }

    primitive.eps = primitive.pressure /
                    ((data_t(eos.adiabatic_index()) - data_t(1.0)) *
                     primitive.rho);
}

template <class data_t, class eos_t>
Primitive<data_t> make_atmosphere_primitive(
    const eos_t &eos, const AtmosphereOptions &options = AtmosphereOptions())
{
    Primitive<data_t> primitive;
    primitive.rho = data_t(options.rho_floor);
    primitive.pressure = data_t(options.pressure_floor);
    primitive.eps = primitive.pressure /
                    ((data_t(eos.adiabatic_index()) - data_t(1.0)) *
                     primitive.rho);
    FOR(i) { primitive.velocity_U[i] = data_t(0.0); }
    return primitive;
}

template <class data_t, class eos_t>
Conserved<data_t> compute_atmosphere_conserved(
    const eos_t &eos, const Tensor<2, data_t> &spatial_metric_LL,
    const AtmosphereOptions &options = AtmosphereOptions())
{
    return compute_conserved(make_atmosphere_primitive<data_t>(eos, options),
                             eos, spatial_metric_LL);
}
} // namespace GRHD

#endif /* GRHD_ATMOSPHERE_HPP_ */
