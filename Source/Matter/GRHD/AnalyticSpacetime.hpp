/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef GRHD_ANALYTIC_SPACETIME_HPP_
#define GRHD_ANALYTIC_SPACETIME_HPP_

#include "DimensionDefinitions.hpp"
#include "StaticMetric.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace GRHD
{
struct CircularBinaryPNParameters
{
    double mass_1 = 0.5;
    double mass_2 = 0.5;
    double separation = 10.0;
    double angular_velocity = 0.0;
    double phase = 0.0;
    double softening_radius = 0.0;
};

struct CircularBinaryPNState
{
    Tensor<1, double> position_1;
    Tensor<1, double> position_2;
    Tensor<1, double> velocity_1;
    Tensor<1, double> velocity_2;
    double angular_velocity = 0.0;
    double phase = 0.0;
};

struct CircularBinaryPNPotentials
{
    double scalar = 0.0;
    Tensor<1, double> scalar_derivative = 0.0;
    Tensor<1, double> vector = 0.0;
    Tensor<2, double> vector_derivative = 0.0;
};

inline double circular_binary_total_mass(
    const CircularBinaryPNParameters &parameters)
{
    return parameters.mass_1 + parameters.mass_2;
}

inline double circular_binary_newtonian_angular_velocity(
    const CircularBinaryPNParameters &parameters)
{
    const double total_mass = circular_binary_total_mass(parameters);
    if (total_mass <= 0.0 || parameters.separation <= 0.0)
    {
        throw std::runtime_error(
            "Circular binary PN parameters need positive mass and separation");
    }
    return std::sqrt(total_mass /
                     (parameters.separation * parameters.separation *
                      parameters.separation));
}

inline double circular_binary_angular_velocity(
    const CircularBinaryPNParameters &parameters)
{
    if (parameters.angular_velocity != 0.0)
        return parameters.angular_velocity;
    return circular_binary_newtonian_angular_velocity(parameters);
}

inline CircularBinaryPNState make_circular_binary_pn_state(
    const CircularBinaryPNParameters &parameters, double time)
{
    const double total_mass = circular_binary_total_mass(parameters);
    if (parameters.mass_1 <= 0.0 || parameters.mass_2 <= 0.0 ||
        total_mass <= 0.0 || parameters.separation <= 0.0 ||
        parameters.softening_radius < 0.0)
    {
        throw std::runtime_error("Invalid circular binary PN parameters");
    }

    CircularBinaryPNState state;
    state.angular_velocity = circular_binary_angular_velocity(parameters);
    state.phase = parameters.phase + state.angular_velocity * time;

    const double cos_phase = std::cos(state.phase);
    const double sin_phase = std::sin(state.phase);
    const double radius_1 = parameters.mass_2 / total_mass *
                            parameters.separation;
    const double radius_2 = parameters.mass_1 / total_mass *
                            parameters.separation;

    FOR(i)
    {
        state.position_1[i] = 0.0;
        state.position_2[i] = 0.0;
        state.velocity_1[i] = 0.0;
        state.velocity_2[i] = 0.0;
    }

    state.position_1[0] = radius_1 * cos_phase;
    state.position_1[1] = radius_1 * sin_phase;
    state.position_2[0] = -radius_2 * cos_phase;
    state.position_2[1] = -radius_2 * sin_phase;

    state.velocity_1[0] = -state.angular_velocity * radius_1 * sin_phase;
    state.velocity_1[1] = state.angular_velocity * radius_1 * cos_phase;
    state.velocity_2[0] = state.angular_velocity * radius_2 * sin_phase;
    state.velocity_2[1] = -state.angular_velocity * radius_2 * cos_phase;
    return state;
}

inline void add_softened_point_pn_potential(
    CircularBinaryPNPotentials &potentials, double mass,
    const Tensor<1, double> &source_position,
    const Tensor<1, double> &source_velocity,
    const Tensor<1, double> &field_position, double softening_radius)
{
    Tensor<1, double> displacement;
    FOR(i) { displacement[i] = 0.0; }
    double radius_squared = softening_radius * softening_radius;
    FOR(i)
    {
        displacement[i] = field_position[i] - source_position[i];
        radius_squared += displacement[i] * displacement[i];
    }

    if (radius_squared <= 0.0)
    {
        throw std::runtime_error(
            "Circular binary PN potential evaluated on an unsoftened source");
    }

    const double inverse_radius = 1.0 / std::sqrt(radius_squared);
    const double inverse_radius_cubed = inverse_radius * inverse_radius *
                                        inverse_radius;
    potentials.scalar += mass * inverse_radius;
    FOR(direction)
    {
        const double derivative = -mass * displacement[direction] *
                                  inverse_radius_cubed;
        potentials.scalar_derivative[direction] += derivative;
    }

    FOR(component)
    {
        potentials.vector[component] += mass * source_velocity[component] *
                                        inverse_radius;
        FOR(direction)
        {
            potentials.vector_derivative[component][direction] +=
                -mass * source_velocity[component] * displacement[direction] *
                inverse_radius_cubed;
        }
    }
}

inline CircularBinaryPNPotentials make_circular_binary_pn_potentials(
    const CircularBinaryPNParameters &parameters, double time,
    const Tensor<1, double> &field_position)
{
    const auto state = make_circular_binary_pn_state(parameters, time);
    CircularBinaryPNPotentials potentials;
    potentials.scalar = 0.0;
    FOR(i)
    {
        potentials.scalar_derivative[i] = 0.0;
        potentials.vector[i] = 0.0;
        FOR(j) { potentials.vector_derivative[i][j] = 0.0; }
    }
    add_softened_point_pn_potential(potentials, parameters.mass_1,
                                    state.position_1, state.velocity_1,
                                    field_position,
                                    parameters.softening_radius);
    add_softened_point_pn_potential(potentials, parameters.mass_2,
                                    state.position_2, state.velocity_2,
                                    field_position,
                                    parameters.softening_radius);
    return potentials;
}

inline CellGeometry make_circular_binary_pn_geometry(
    const CircularBinaryPNParameters &parameters, double time,
    const Tensor<1, double> &field_position)
{
    const auto potentials = make_circular_binary_pn_potentials(
        parameters, time, field_position);
    const double U = potentials.scalar;
    const double conformal_factor = 1.0 + 2.0 * U;
    if (conformal_factor <= 0.0 || !std::isfinite(conformal_factor))
    {
        throw std::runtime_error(
            "Circular binary PN spatial metric is not positive definite");
    }

    double vector_squared = 0.0;
    FOR(i) { vector_squared += potentials.vector[i] * potentials.vector[i]; }

    const double lapse_squared = 1.0 - 2.0 * U + 2.0 * U * U +
                                 16.0 * vector_squared / conformal_factor;
    if (lapse_squared <= 0.0 || !std::isfinite(lapse_squared))
    {
        throw std::runtime_error(
            "Circular binary PN lapse is not real at this field point");
    }

    CellGeometry geometry = make_flat_cell_geometry();
    FOR(i)
    {
        geometry.spatial_metric_LL[i][i] = conformal_factor;
        geometry.spatial_metric_UU[i][i] = 1.0 / conformal_factor;
        geometry.shift_U[i] = -4.0 * potentials.vector[i] /
                              conformal_factor;
    }
    geometry.lapse = std::sqrt(lapse_squared);
    return geometry;
}

inline StaticMetricDerivatives<double> make_circular_binary_pn_derivatives(
    const CircularBinaryPNParameters &parameters, double time,
    const Tensor<1, double> &field_position)
{
    const auto potentials = make_circular_binary_pn_potentials(
        parameters, time, field_position);
    const double U = potentials.scalar;
    const double conformal_factor = 1.0 + 2.0 * U;
    if (conformal_factor <= 0.0 || !std::isfinite(conformal_factor))
    {
        throw std::runtime_error(
            "Circular binary PN spatial metric is not positive definite");
    }

    double vector_squared = 0.0;
    FOR(i) { vector_squared += potentials.vector[i] * potentials.vector[i]; }
    const double lapse_squared = 1.0 - 2.0 * U + 2.0 * U * U +
                                 16.0 * vector_squared / conformal_factor;
    if (lapse_squared <= 0.0 || !std::isfinite(lapse_squared))
    {
        throw std::runtime_error(
            "Circular binary PN lapse is not real at this field point");
    }
    const double lapse = std::sqrt(lapse_squared);

    StaticMetricDerivatives<double> derivatives;
    FOR(direction)
    {
        const double dU = potentials.scalar_derivative[direction];
        const double d_conformal_factor = 2.0 * dU;
        derivatives.log_sqrt_det_spatial_metric[direction] =
            1.5 * d_conformal_factor / conformal_factor;

        FOR(metric_dir)
        {
            derivatives.spatial_metric_LL[direction][metric_dir][metric_dir] =
                d_conformal_factor;
        }

        double d_vector_squared = 0.0;
        FOR(component)
        {
            d_vector_squared += 2.0 * potentials.vector[component] *
                                potentials.vector_derivative[component]
                                                              [direction];
            derivatives.shift_U[component][direction] =
                -4.0 * (potentials.vector_derivative[component][direction] /
                            conformal_factor -
                        potentials.vector[component] * d_conformal_factor /
                            (conformal_factor * conformal_factor));
        }

        const double d_lapse_squared =
            (-2.0 + 4.0 * U) * dU +
            16.0 * (d_vector_squared / conformal_factor -
                    vector_squared * d_conformal_factor /
                        (conformal_factor * conformal_factor));
        derivatives.lapse[direction] = 0.5 * d_lapse_squared / lapse;
    }
    return derivatives;
}
} // namespace GRHD

#endif /* GRHD_ANALYTIC_SPACETIME_HPP_ */
