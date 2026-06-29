/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef USERVARIABLES_HPP
#define USERVARIABLES_HPP

#include "ArrayTools.hpp"
#include "CCZ4UserVariables.hpp"
#include <array>
#include <string>

enum
{
    NUM_DIAGNOSTIC_VARS
};

namespace DiagnosticVariables
{
static const std::array<std::string, NUM_DIAGNOSTIC_VARS> variable_names = {};
}

enum
{
    c_GRHD_D = NUM_CCZ4_VARS,
    c_GRHD_S1,
    c_GRHD_S2,
    c_GRHD_S3,
    c_GRHD_tau,

    c_GRHD_rho,
    c_GRHD_eps,
    c_GRHD_pressure,
    c_GRHD_v1,
    c_GRHD_v2,
    c_GRHD_v3,

    NUM_VARS
};

namespace UserVariables
{
static const std::array<std::string, NUM_VARS - NUM_CCZ4_VARS>
    user_variable_names = {"GRHD_D",        "GRHD_S1",       "GRHD_S2",
                           "GRHD_S3",       "GRHD_tau",      "GRHD_rho",
                           "GRHD_eps",      "GRHD_pressure", "GRHD_v1",
                           "GRHD_v2",       "GRHD_v3"};

static const std::array<std::string, NUM_VARS> variable_names =
    ArrayTools::concatenate(ccz4_variable_names, user_variable_names);
} // namespace UserVariables

#endif /* USERVARIABLES_HPP */
