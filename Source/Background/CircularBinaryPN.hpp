/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef CIRCULARBINARYPN_HPP_
#define CIRCULARBINARYPN_HPP_

#include "ADMFixedBGVars.hpp"
#include "Cell.hpp"
#include "Coordinates.hpp"
#include "DimensionDefinitions.hpp"
#include "Tensor.hpp"
#include "TensorAlgebra.hpp"
#include "UserVariables.hpp"
#include "simd.hpp"
#include <array>
#include <cmath>

//! Softened circular binary post-Newtonian metric background.
/*!
 * This is an instantaneous analytic background intended as a stepping stone
 * toward BBH disk experiments. The spatial metric, lapse, shift, and their
 * spatial derivatives are built from a Newtonian scalar potential and vector
 * potential for two masses on a prescribed circular orbit. The extrinsic
 * curvature includes the explicit time derivative of the conformally flat
 * spatial metric for the prescribed circular motion.
 */
class CircularBinaryPN
{
  public:
    struct params_t
    {
        double mass_1 = 0.5;
        double mass_2 = 0.5;
        double separation = 10.0;
        double angular_velocity = 0.0;
        double phase = 0.0;
        double time = 0.0;
        double softening_radius = 0.5;
        std::array<double, CH_SPACEDIM> center = {0.0, 0.0, 0.0};
    };

    template <class data_t> using Vars = ADMFixedBGVars::Vars<data_t>;

    const params_t m_params;
    const double m_dx;

    CircularBinaryPN(params_t a_params, double a_dx)
        : m_params(a_params), m_dx(a_dx)
    {
        if (m_params.mass_1 <= 0.0 || m_params.mass_2 <= 0.0 ||
            m_params.separation <= 0.0 || m_params.softening_radius < 0.0)
        {
            MayDay::Error("Invalid CircularBinaryPN parameters");
        }
    }

    template <class data_t> void compute(Cell<data_t> current_cell) const
    {
        const Coordinates<data_t> coords(current_cell, m_dx, m_params.center);
        Vars<data_t> metric_vars;
        compute_metric_background(metric_vars, coords);

        data_t chi = TensorAlgebra::compute_determinant_sym(metric_vars.gamma);
        chi = pow(chi, -1.0 / 3.0);
        current_cell.store_vars(chi, c_chi);
    }

    template <class data_t, template <typename> class vars_t>
    void compute_metric_background(vars_t<data_t> &vars,
                                   const Coordinates<data_t> &coords) const
    {
        Tensor<1, data_t> position;
        FOR1(i) { position[i] = 0.0; }
        position[0] = coords.x;
        position[1] = coords.y;
        position[2] = coords.z;

        const auto state = compute_state();
        Potentials<data_t> potentials;
        potentials.scalar = 0.0;
        FOR1(i)
        {
            potentials.scalar_derivative[i] = 0.0;
            potentials.vector[i] = 0.0;
            FOR1(j) { potentials.vector_derivative[i][j] = 0.0; }
        }
        potentials.scalar_time_derivative = 0.0;
        add_point_potential(potentials, m_params.mass_1, state.position_1,
                            state.velocity_1, position);
        add_point_potential(potentials, m_params.mass_2, state.position_2,
                            state.velocity_2, position);

        const data_t U = potentials.scalar;
        const data_t d_conformal_factor_dt =
            2.0 * potentials.scalar_time_derivative;
        const data_t conformal_factor = 1.0 + 2.0 * U;

        data_t vector_squared = 0.0;
        FOR1(i)
        {
            vector_squared += potentials.vector[i] * potentials.vector[i];
        }

        const data_t lapse_squared =
            1.0 - 2.0 * U + 2.0 * U * U +
            16.0 * vector_squared / conformal_factor;

        FOR1(i)
        {
            vars.shift[i] = 0.0;
            vars.d1_lapse[i] = 0.0;
            FOR1(j)
            {
                vars.gamma[i][j] = 0.0;
                vars.d1_shift[i][j] = 0.0;
                vars.K_tensor[i][j] = 0.0;
                FOR1(k) { vars.d1_gamma[i][j][k] = 0.0; }
            }
        }

        vars.lapse = sqrt(lapse_squared);
        FOR1(i)
        {
            vars.gamma[i][i] = conformal_factor;
            vars.shift[i] = -4.0 * potentials.vector[i] / conformal_factor;
        }

        const auto gamma_UU = TensorAlgebra::compute_inverse_sym(vars.gamma);

        FOR1(direction)
        {
            const data_t dU = potentials.scalar_derivative[direction];
            const data_t d_conformal_factor = 2.0 * dU;
            FOR1(metric_dir)
            {
                vars.d1_gamma[metric_dir][metric_dir][direction] =
                    d_conformal_factor;
            }

            data_t d_vector_squared = 0.0;
            FOR1(component)
            {
                d_vector_squared +=
                    2.0 * potentials.vector[component] *
                    potentials.vector_derivative[component][direction];
                vars.d1_shift[component][direction] =
                    -4.0 *
                    (potentials.vector_derivative[component][direction] /
                         conformal_factor -
                     potentials.vector[component] * d_conformal_factor /
                         (conformal_factor * conformal_factor));
            }

            const data_t d_lapse_squared =
                (-2.0 + 4.0 * U) * dU +
                16.0 * (d_vector_squared / conformal_factor -
                        vector_squared * d_conformal_factor /
                            (conformal_factor * conformal_factor));
            vars.d1_lapse[direction] =
                0.5 * d_lapse_squared / vars.lapse;
        }

        const auto chris_phys =
            TensorAlgebra::compute_christoffel(vars.d1_gamma, gamma_UU);
        FOR2(i, j)
        {
            FOR1(k)
            {
                vars.K_tensor[i][j] +=
                    vars.gamma[k][j] * vars.d1_shift[k][i] +
                    vars.gamma[k][i] * vars.d1_shift[k][j] +
                    (vars.d1_gamma[k][i][j] +
                     vars.d1_gamma[k][j][i]) *
                        vars.shift[k];
                FOR1(m)
                {
                    vars.K_tensor[i][j] +=
                        -2.0 * chris_phys.ULL[k][i][j] *
                        vars.gamma[k][m] * vars.shift[m];
                }
            }
            if (i == j)
                vars.K_tensor[i][j] -= d_conformal_factor_dt;
            vars.K_tensor[i][j] *= 0.5 / vars.lapse;
        }
        vars.K = TensorAlgebra::compute_trace(gamma_UU, vars.K_tensor);
    }

  protected:
    struct state_t
    {
        Tensor<1, double> position_1;
        Tensor<1, double> position_2;
        Tensor<1, double> velocity_1;
        Tensor<1, double> velocity_2;
    };

    template <class data_t> struct Potentials
    {
        data_t scalar = 0.0;
        data_t scalar_time_derivative = 0.0;
        Tensor<1, data_t> scalar_derivative = data_t(0.0);
        Tensor<1, data_t> vector = data_t(0.0);
        Tensor<2, data_t> vector_derivative = data_t(0.0);
    };

    double angular_velocity() const
    {
        if (m_params.angular_velocity != 0.0)
            return m_params.angular_velocity;
        const double total_mass = m_params.mass_1 + m_params.mass_2;
        return sqrt(total_mass /
                    (m_params.separation * m_params.separation *
                     m_params.separation));
    }

    state_t compute_state() const
    {
        const double total_mass = m_params.mass_1 + m_params.mass_2;
        const double omega = angular_velocity();
        const double phase = m_params.phase + omega * m_params.time;
        const double cos_phase = cos(phase);
        const double sin_phase = sin(phase);
        const double radius_1 = m_params.mass_2 / total_mass *
                                m_params.separation;
        const double radius_2 = m_params.mass_1 / total_mass *
                                m_params.separation;

        state_t state;
        FOR1(i)
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

        state.velocity_1[0] = -omega * radius_1 * sin_phase;
        state.velocity_1[1] = omega * radius_1 * cos_phase;
        state.velocity_2[0] = omega * radius_2 * sin_phase;
        state.velocity_2[1] = -omega * radius_2 * cos_phase;
        return state;
    }

    template <class data_t>
    void add_point_potential(Potentials<data_t> &potentials, double mass,
                             const Tensor<1, double> &source_position,
                             const Tensor<1, double> &source_velocity,
                             const Tensor<1, data_t> &field_position) const
    {
        Tensor<1, data_t> displacement;
        FOR1(i) { displacement[i] = 0.0; }
        data_t radius_squared =
            m_params.softening_radius * m_params.softening_radius;
        FOR1(i)
        {
            displacement[i] = field_position[i] - source_position[i];
            radius_squared += displacement[i] * displacement[i];
        }

        const data_t inverse_radius = 1.0 / sqrt(radius_squared);
        const data_t inverse_radius_cubed =
            inverse_radius * inverse_radius * inverse_radius;
        potentials.scalar += mass * inverse_radius;
        data_t displacement_dot_velocity = 0.0;
        FOR1(i)
        {
            displacement_dot_velocity +=
                displacement[i] * source_velocity[i];
        }
        potentials.scalar_time_derivative +=
            mass * displacement_dot_velocity * inverse_radius_cubed;
        FOR1(direction)
        {
            potentials.scalar_derivative[direction] +=
                -mass * displacement[direction] * inverse_radius_cubed;
        }

        FOR2(component, direction)
        {
            potentials.vector_derivative[component][direction] +=
                -mass * source_velocity[component] *
                displacement[direction] * inverse_radius_cubed;
        }
        FOR1(component)
        {
            potentials.vector[component] +=
                mass * source_velocity[component] * inverse_radius;
        }
    }
};

#endif /* CIRCULARBINARYPN_HPP_ */
