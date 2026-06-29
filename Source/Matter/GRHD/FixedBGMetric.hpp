/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef GRHD_FIXED_BG_METRIC_HPP_
#define GRHD_FIXED_BG_METRIC_HPP_

#include "ADMFixedBGVars.hpp"
#include "Coordinates.hpp"
#include "StaticMetric.hpp"
#include "TensorAlgebra.hpp"

namespace GRHD
{
//! Convert GRDzhadzha ADM fixed-background data into GRHD cell geometry.
template <class data_t, template <typename> class vars_t>
CellGeometryData<data_t>
make_cell_geometry_from_adm(const vars_t<data_t> &metric_vars)
{
    CellGeometryData<data_t> geometry;
    geometry.spatial_metric_LL = metric_vars.gamma;
    geometry.spatial_metric_UU =
        TensorAlgebra::compute_inverse_sym(metric_vars.gamma);
    geometry.lapse = metric_vars.lapse;
    geometry.shift_U = metric_vars.shift;
    geometry.extrinsic_curvature_LL = metric_vars.K_tensor;
    return geometry;
}

//! Convert ADM first derivatives into the GRHD static-metric source format.
template <class data_t, template <typename> class vars_t>
StaticMetricDerivatives<data_t>
make_static_metric_derivatives_from_adm(const vars_t<data_t> &metric_vars)
{
    StaticMetricDerivatives<data_t> derivatives;
    derivatives.lapse = metric_vars.d1_lapse;
    derivatives.shift_U = metric_vars.d1_shift;

    const auto gamma_UU =
        TensorAlgebra::compute_inverse_sym(metric_vars.gamma);

    FOR(direction)
    {
        FOR(i, j)
        {
            derivatives.spatial_metric_LL[direction][i][j] =
                metric_vars.d1_gamma[i][j][direction];
            derivatives.log_sqrt_det_spatial_metric[direction] +=
                0.5 * gamma_UU[i][j] *
                metric_vars.d1_gamma[i][j][direction];
        }
    }
    return derivatives;
}

//! Compute GRHD cell geometry directly from a GRDzhadzha analytic background.
template <class background_t, class data_t>
CellGeometryData<data_t> make_cell_geometry_from_background(
    const background_t &background, const Coordinates<data_t> &coords)
{
    ADMFixedBGVars::Vars<data_t> metric_vars;
    background.compute_metric_background(metric_vars, coords);
    return make_cell_geometry_from_adm(metric_vars);
}

//! Compute GRHD metric-source derivatives from a GRDzhadzha analytic background.
template <class background_t, class data_t>
StaticMetricDerivatives<data_t> make_static_metric_derivatives_from_background(
    const background_t &background, const Coordinates<data_t> &coords)
{
    ADMFixedBGVars::Vars<data_t> metric_vars;
    background.compute_metric_background(metric_vars, coords);
    return make_static_metric_derivatives_from_adm(metric_vars);
}
} // namespace GRHD

#endif /* GRHD_FIXED_BG_METRIC_HPP_ */
