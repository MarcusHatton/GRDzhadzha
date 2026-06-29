/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#include "ADMFixedBGVars.hpp"
#include "AnalyticSpacetime.hpp"
#include "CircularBinaryPN.hpp"
#include "Coordinates.hpp"
#include "DimensionDefinitions.hpp"
#include "FixedBGMetric.hpp"
#include "IdealGasEOS.hpp"
#include "StaticMetric.hpp"
#include "TensorAlgebra.hpp"
#include "UserVariables.hpp"
#include "Valencia.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

namespace
{
bool check_close(const std::string &name, double value, double expected,
                 double tolerance)
{
    if (!std::isfinite(value) || !std::isfinite(expected))
    {
        std::cout << name << " is not finite: got " << value
                  << ", expected " << expected << std::endl;
        return false;
    }
    const double scale = std::max(1.0, std::abs(expected));
    const double error = std::abs(value - expected) / scale;
    if (error > tolerance)
    {
        std::cout << name << " mismatch: got " << value << ", expected "
                  << expected << ", relative error " << error << std::endl;
        return false;
    }
    return true;
}

bool check_finite(const std::string &name, double value)
{
    if (!std::isfinite(value))
    {
        std::cout << name << " is not finite: " << value << std::endl;
        return false;
    }
    return true;
}

Tensor<1, double> position_from_coords(const Coordinates<double> &coords)
{
    Tensor<1, double> position;
    FOR(i) { position[i] = 0.0; }
    position[0] = coords.x;
    position[1] = coords.y;
    position[2] = coords.z;
    return position;
}

bool check_geometry_inverse(const GRHD::CellGeometry &geometry)
{
    bool passed = true;
    FOR(i, j)
    {
        double value = 0.0;
        FOR(k)
        {
            value += geometry.spatial_metric_LL[i][k] *
                     geometry.spatial_metric_UU[k][j];
        }
        passed &= check_close("gamma inverse", value,
                              TensorAlgebra::delta(i, j), 1.0e-11);
    }
    return passed;
}

bool check_sources_are_finite(const GRHD::Conserved<double> &source)
{
    bool passed = true;
    passed &= check_finite("source D", source.D);
    passed &= check_finite("source tau", source.tau);
    FOR(i) { passed &= check_finite("source S", source.S_L[i]); }
    return passed;
}
} // namespace

int main()
{
    bool passed = true;

    CircularBinaryPN::params_t bg_params;
    bg_params.mass_1 = 0.55;
    bg_params.mass_2 = 0.45;
    bg_params.separation = 9.0;
    bg_params.phase = 0.2;
    bg_params.time = 3.0;
    bg_params.softening_radius = 0.7;
    bg_params.center = {0.0, 0.0, 0.0};
    CircularBinaryPN background(bg_params, 0.25);

    const IntVect iv(D_DECL(60, 40, 24));
    const Coordinates<double> coords(iv, 0.25, bg_params.center);

    ADMFixedBGVars::Vars<double> metric_vars;
    background.compute_metric_background(metric_vars, coords);

    const auto geometry =
        GRHD::make_cell_geometry_from_background(background, coords);
    const auto derivatives =
        GRHD::make_static_metric_derivatives_from_background(background,
                                                             coords);

    GRHD::CircularBinaryPNParameters expected_params;
    expected_params.mass_1 = bg_params.mass_1;
    expected_params.mass_2 = bg_params.mass_2;
    expected_params.separation = bg_params.separation;
    expected_params.angular_velocity = bg_params.angular_velocity;
    expected_params.phase = bg_params.phase;
    expected_params.softening_radius = bg_params.softening_radius;

    const auto position = position_from_coords(coords);
    const auto expected_geometry = GRHD::make_circular_binary_pn_geometry(
        expected_params, bg_params.time, position);
    const auto expected_derivatives =
        GRHD::make_circular_binary_pn_derivatives(
            expected_params, bg_params.time, position);

    passed &= check_geometry_inverse(geometry);
    passed &= check_close("lapse", geometry.lapse, expected_geometry.lapse,
                          1.0e-12);

    double extrinsic_norm = 0.0;
    FOR(i, j)
    {
        passed &= check_close("extrinsic curvature",
                              geometry.extrinsic_curvature_LL[i][j],
                              metric_vars.K_tensor[i][j], 0.0);
        passed &= check_finite("extrinsic curvature",
                               geometry.extrinsic_curvature_LL[i][j]);
        extrinsic_norm = std::max(
            extrinsic_norm,
            std::abs(geometry.extrinsic_curvature_LL[i][j]));
    }
    if (extrinsic_norm <= 1.0e-14)
    {
        std::cout << "binary PN test expected nonzero K_ij"
                  << std::endl;
        passed = false;
    }

    CircularBinaryPN::params_t plus_params = bg_params;
    CircularBinaryPN::params_t minus_params = bg_params;
    const double time_derivative_step = 1.0e-5;
    plus_params.time += time_derivative_step;
    minus_params.time -= time_derivative_step;
    CircularBinaryPN plus_background(plus_params, 0.25);
    CircularBinaryPN minus_background(minus_params, 0.25);
    ADMFixedBGVars::Vars<double> plus_metric_vars;
    ADMFixedBGVars::Vars<double> minus_metric_vars;
    plus_background.compute_metric_background(plus_metric_vars, coords);
    minus_background.compute_metric_background(minus_metric_vars, coords);

    const auto gamma_UU =
        TensorAlgebra::compute_inverse_sym(metric_vars.gamma);
    const auto chris_phys =
        TensorAlgebra::compute_christoffel(metric_vars.d1_gamma, gamma_UU);
    FOR(i, j)
    {
        double covariant_shift_derivative = 0.0;
        FOR(k)
        {
            covariant_shift_derivative +=
                metric_vars.gamma[k][j] * metric_vars.d1_shift[k][i] +
                metric_vars.gamma[k][i] * metric_vars.d1_shift[k][j] +
                (metric_vars.d1_gamma[k][i][j] +
                 metric_vars.d1_gamma[k][j][i]) *
                    metric_vars.shift[k];
            FOR(m)
            {
                covariant_shift_derivative +=
                    -2.0 * chris_phys.ULL[k][i][j] *
                    metric_vars.gamma[k][m] * metric_vars.shift[m];
            }
        }
        const double d_gamma_dt =
            (plus_metric_vars.gamma[i][j] - minus_metric_vars.gamma[i][j]) /
            (2.0 * time_derivative_step);
        const double expected_K =
            (covariant_shift_derivative - d_gamma_dt) /
            (2.0 * metric_vars.lapse);
        passed &= check_close("ADM K time derivative identity",
                              metric_vars.K_tensor[i][j], expected_K,
                              1.0e-8);
    }
    FOR(i)
    {
        passed &= check_close("shift", geometry.shift_U[i],
                              expected_geometry.shift_U[i], 1.0e-12);
        passed &= check_close("d lapse", derivatives.lapse[i],
                              expected_derivatives.lapse[i], 1.0e-12);
        passed &= check_close(
            "d log sqrt det",
            derivatives.log_sqrt_det_spatial_metric[i],
            expected_derivatives.log_sqrt_det_spatial_metric[i], 1.0e-12);
        FOR(j)
        {
            passed &= check_close("metric LL",
                                  geometry.spatial_metric_LL[i][j],
                                  expected_geometry.spatial_metric_LL[i][j],
                                  1.0e-12);
            passed &= check_close("metric UU",
                                  geometry.spatial_metric_UU[i][j],
                                  expected_geometry.spatial_metric_UU[i][j],
                                  1.0e-12);
            passed &= check_close(
                "d shift", derivatives.shift_U[i][j],
                expected_derivatives.shift_U[i][j], 1.0e-12);
        }
    }

    IdealGasEOS eos(5.0 / 3.0);
    GRHD::Primitive<double> primitive;
    primitive.rho = 0.9;
    primitive.eps = 0.15;
    primitive.pressure = eos.compute_pressure(primitive.rho, primitive.eps);
    primitive.velocity_U[0] = 0.004;
    primitive.velocity_U[1] = -0.006;
    primitive.velocity_U[2] = 0.002;

    const auto conserved =
        GRHD::compute_conserved(primitive, eos, geometry.spatial_metric_LL);
    const auto recovered =
        GRHD::recover_primitive(conserved, eos, geometry.spatial_metric_UU);
    if (!recovered.success)
    {
        std::cout << "primitive recovery failed with residual "
                  << recovered.residual << std::endl;
        return 1;
    }

    passed &= check_close("rho", recovered.primitive.rho, primitive.rho,
                          1.0e-10);
    passed &= check_close("eps", recovered.primitive.eps, primitive.eps,
                          1.0e-10);
    passed &= check_close("pressure", recovered.primitive.pressure,
                          primitive.pressure, 1.0e-10);
    FOR(i)
    {
        passed &= check_close("velocity", recovered.primitive.velocity_U[i],
                              primitive.velocity_U[i], 1.0e-10);
    }

    const auto source = GRHD::compute_static_metric_source_terms(
        primitive, conserved, eos, geometry, derivatives);
    passed &= check_sources_are_finite(source);


    auto geometry_without_k = geometry;
    FOR(i, j)
    {
        geometry_without_k.extrinsic_curvature_LL[i][j] = 0.0;
    }
    const auto source_without_k = GRHD::compute_static_metric_source_terms(
        primitive, conserved, eos, geometry_without_k, derivatives);
    const auto stress_UU = GRHD::compute_spatial_stress_UU(
        primitive, eos, geometry);
    double expected_tau_delta = 0.0;
    FOR(i, j)
    {
        expected_tau_delta += geometry.lapse * stress_UU[i][j] *
                              geometry.extrinsic_curvature_LL[i][j];
    }
    passed &= check_close("extrinsic tau source",
                          source.tau - source_without_k.tau,
                          expected_tau_delta, 1.0e-12);
    passed &= check_close("extrinsic D source",
                          source.D - source_without_k.D, 0.0, 1.0e-14);
    FOR(i)
    {
        passed &= check_close("extrinsic momentum source",
                              source.S_L[i] - source_without_k.S_L[i],
                              0.0, 1.0e-14);
    }


    const double d_log_sqrt_det_dt = -0.037;
    const auto volume_time_source =
        GRHD::compute_metric_volume_time_source_terms(
            conserved, d_log_sqrt_det_dt);
    passed &= check_close("volume time D source", volume_time_source.D,
                          -conserved.D * d_log_sqrt_det_dt, 1.0e-14);
    passed &= check_close("volume time tau source", volume_time_source.tau,
                          -conserved.tau * d_log_sqrt_det_dt, 1.0e-14);
    FOR(i)
    {
        passed &= check_close("volume time momentum source",
                              volume_time_source.S_L[i],
                              -conserved.S_L[i] * d_log_sqrt_det_dt,
                              1.0e-14);
    }

    if (!passed)
        return 1;

    std::cout << "GRHD binary PN fixed-background test passed..."
              << std::endl;
    return 0;
}
