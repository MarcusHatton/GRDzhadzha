/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef GRHD_STRESSENERGY_HPP_
#define GRHD_STRESSENERGY_HPP_

#include "CCZ4Geometry.hpp"
#include "DimensionDefinitions.hpp"
#include "GRHDVars.hpp"
#include "TensorAlgebra.hpp"
#include "Valencia.hpp"

namespace GRHD
{
//! Compute the 3+1 stress-energy projections for a perfect fluid.
template <class data_t, class eos_t>
emtensor_t<data_t>
compute_emtensor(const Primitive<data_t> &primitive, const eos_t &eos,
                 const Tensor<2, data_t> &spatial_metric_LL,
                 const Tensor<2, data_t> &spatial_metric_UU)
{
    emtensor_t<data_t> out;
    const Tensor<1, data_t> velocity_L =
        TensorAlgebra::lower_all(primitive.velocity_U, spatial_metric_LL);
    const data_t W =
        compute_lorentz_factor(primitive.velocity_U, spatial_metric_LL);
    const data_t h = eos.compute_specific_enthalpy(
        primitive.rho, primitive.eps, primitive.pressure);
    const data_t rho_h_W2 = primitive.rho * h * W * W;

    out.rho = rho_h_W2 - primitive.pressure;
    FOR(i)
    {
        out.Si[i] = rho_h_W2 * velocity_L[i];
        FOR(j)
        {
            out.Sij[i][j] = rho_h_W2 * velocity_L[i] * velocity_L[j] +
                            primitive.pressure * spatial_metric_LL[i][j];
        }
    }
    out.S = TensorAlgebra::compute_trace(out.Sij, spatial_metric_UU);
    return out;
}

//! Convert GRChombo's conformal spatial metric h_ij and chi to gamma_ij.
template <class data_t>
Tensor<2, data_t>
compute_physical_spatial_metric(const Tensor<2, data_t> &conformal_metric_LL,
                                const data_t &chi)
{
    Tensor<2, data_t> spatial_metric_LL;
    FOR(i, j) { spatial_metric_LL[i][j] = conformal_metric_LL[i][j] / chi; }
    return spatial_metric_LL;
}
} // namespace GRHD

#endif /* GRHD_STRESSENERGY_HPP_ */
