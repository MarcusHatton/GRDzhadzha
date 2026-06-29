/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef GRHD_FINITEVOLUME_HPP_
#define GRHD_FINITEVOLUME_HPP_

#include "Atmosphere.hpp"
#include "DimensionDefinitions.hpp"
#include "GRHDVars.hpp"
#include "Reconstruction.hpp"
#include "RiemannSolvers.hpp"
#include "Valencia.hpp"
#include <array>

namespace GRHD
{
template <class data_t> Conserved<data_t> zero_conserved()
{
    Conserved<data_t> conserved;
    conserved.D = data_t(0.0);
    FOR(i) { conserved.S_L[i] = data_t(0.0); }
    conserved.tau = data_t(0.0);
    return conserved;
}

template <class data_t>
Conserved<data_t> add_scaled(const Conserved<data_t> &state,
                             const Conserved<data_t> &rhs,
                             const data_t &scale)
{
    Conserved<data_t> out = state;
    out.D += scale * rhs.D;
    out.tau += scale * rhs.tau;
    FOR(i) { out.S_L[i] += scale * rhs.S_L[i]; }
    return out;
}

template <class data_t>
Conserved<data_t> compute_flux_divergence(
    const std::array<Flux<data_t>, CH_SPACEDIM> &flux_hi,
    const std::array<Flux<data_t>, CH_SPACEDIM> &flux_lo,
    const Tensor<1, data_t> &inverse_dx_U)
{
    Conserved<data_t> rhs = zero_conserved<data_t>();
    FOR(dir)
    {
        rhs.D -= inverse_dx_U[dir] * (flux_hi[dir].D - flux_lo[dir].D);
        rhs.tau -= inverse_dx_U[dir] *
                   (flux_hi[dir].tau - flux_lo[dir].tau);
        FOR(momentum_dir)
        {
            rhs.S_L[momentum_dir] -=
                inverse_dx_U[dir] *
                (flux_hi[dir].S_L[momentum_dir] -
                 flux_lo[dir].S_L[momentum_dir]);
        }
    }
    return rhs;
}

template <class data_t, class eos_t>
Flux<data_t> compute_muscl_hlle_flux(
    const Primitive<data_t> &left_cell_primitive,
    const Primitive<data_t> &right_cell_primitive,
    const Primitive<data_t> &left_slope, const Primitive<data_t> &right_slope,
    const eos_t &eos, const Tensor<2, data_t> &spatial_metric_LL,
    const Tensor<2, data_t> &spatial_metric_UU, int direction,
    const data_t &lapse, const Tensor<1, data_t> &shift_U,
    const AtmosphereOptions &atmosphere = AtmosphereOptions(),
    bool use_reconstruction = true)
{
    Primitive<data_t> left_primitive = left_cell_primitive;
    Primitive<data_t> right_primitive = right_cell_primitive;

    if (use_reconstruction)
    {
        left_primitive = add_scaled(left_cell_primitive, left_slope,
                                    data_t(0.5));
        right_primitive = add_scaled(right_cell_primitive, right_slope,
                                     data_t(-0.5));
    }

    enforce_primitive_floors(left_primitive, eos, spatial_metric_LL,
                             atmosphere);
    enforce_primitive_floors(right_primitive, eos, spatial_metric_LL,
                             atmosphere);

    const auto left_conserved =
        compute_conserved(left_primitive, eos, spatial_metric_LL);
    const auto right_conserved =
        compute_conserved(right_primitive, eos, spatial_metric_LL);

    return compute_hlle_flux(left_primitive, right_primitive, left_conserved,
                             right_conserved, eos, spatial_metric_LL,
                             spatial_metric_UU, direction, lapse, shift_U);
}

template <class data_t, class eos_t>
Flux<data_t> compute_muscl_hlle_flux(
    const Primitive<data_t> &left_cell_primitive,
    const Primitive<data_t> &right_cell_primitive,
    const Primitive<data_t> &left_slope, const Primitive<data_t> &right_slope,
    const eos_t &eos, const Tensor<2, data_t> &spatial_metric_LL,
    const Tensor<2, data_t> &spatial_metric_UU, int direction,
    const AtmosphereOptions &atmosphere = AtmosphereOptions(),
    bool use_reconstruction = true)
{
    Tensor<1, data_t> zero_shift;
    FOR(i) { zero_shift[i] = 0.0; }
    return compute_muscl_hlle_flux(left_cell_primitive, right_cell_primitive,
                                   left_slope, right_slope, eos,
                                   spatial_metric_LL, spatial_metric_UU,
                                   direction, data_t(1.0), zero_shift,
                                   atmosphere, use_reconstruction);
}
} // namespace GRHD

#endif /* GRHD_FINITEVOLUME_HPP_ */
