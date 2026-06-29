/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef GRHD_VARS_HPP_
#define GRHD_VARS_HPP_

#include "Tensor.hpp"

namespace GRHD
{
//! Primitive variables: rest-mass density, specific internal energy, pressure,
//! and Eulerian three-velocity with an upper spatial index.
template <class data_t> struct Primitive
{
    data_t rho;
    data_t eps;
    data_t pressure;
    Tensor<1, data_t> velocity_U;
};

//! Valencia conserved variables without magnetic fields.
template <class data_t> struct Conserved
{
    data_t D;
    Tensor<1, data_t> S_L;
    data_t tau;
};

//! Valencia flux in one coordinate direction.
template <class data_t> struct Flux
{
    data_t D;
    Tensor<1, data_t> S_L;
    data_t tau;
};
} // namespace GRHD

#endif /* GRHD_VARS_HPP_ */
