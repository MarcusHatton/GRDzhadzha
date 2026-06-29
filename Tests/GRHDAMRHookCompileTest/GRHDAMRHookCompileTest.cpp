/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#include "FourthOrderDerivatives.hpp"
#include "GRHDCCZ4AMRLevelHooks.hpp"
#include "GRHDCCZ4LevelOperator.hpp"
#include "IdealGasEOS.hpp"
#include "MovingPunctureGauge.hpp"
#include <iostream>

// Explicitly instantiate the GRChombo-facing GRHD level helpers. This is a
// compile-time guard for the AMR/CCZ4 coupling surface while the fixed-
// background executable tests exercise the standalone finite-volume path.
template class GRHD::GRHDCCZ4LevelOperator<MovingPunctureGauge,
                                           FourthOrderDerivatives>;
template class GRHD::GRHDCCZ4AMRLevelHooks<MovingPunctureGauge,
                                           FourthOrderDerivatives>;

int main()
{
    CCZ4_params_t<MovingPunctureGauge::params_t> ccz4_params;
    IdealGasEOS eos(5.0 / 3.0);

    GRHD::GRHDCCZ4LevelOperator<> level_operator(eos, ccz4_params, 0.25,
                                                 0.1);
    GRHD::GRHDCCZ4AMRLevelHooks<> level_hooks(eos, ccz4_params, 0.25, 0.1);
    (void)level_operator;
    (void)level_hooks;

    std::cout << "GRHD AMR hook compile test passed..." << std::endl;
    return 0;
}
