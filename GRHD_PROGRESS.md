# GRHD Progress Log

Date: 2026-06-24

## Goal

Assess whether GRDzhadzha is an appropriate base for fixed-background GRHD
simulations of gas disks around black holes, with a long-term target of binary
black-hole merger backgrounds and an initial stepping stone of a disk around a
single black hole.

## Initial Assessment

- GRDzhadzha itself currently exposes fixed-background scalar-field examples
  (`Examples/KerrBHScalarField`, `Examples/BoostedBHComplexScalar`) and matching
  tests. It does not currently contain a GRHD application or a disk simulation.
- The fixed-background architecture is a good conceptual match: matter evolution
  is separated from analytic ADM background data through `MatterEvolution` and
  background classes such as `KerrSchild` and `BoostedBH`.
- The local GRChombo checkout at `/home/mjhatto/GRChombo` already contains a
  `Source/Matter/GRHD` module with Valencia variables, ideal-gas EOS,
  primitive recovery, atmosphere handling, reconstruction, Riemann solvers,
  finite-volume updates, static-metric source terms, and analytic spacetime
  helpers.
- The same GRChombo checkout includes GRHD tests/examples for Sod shocks,
  smooth convergence, blast waves, Michel accretion, Schwarzschild disk, and
  Schwarzschild Keplerian disk. These are directly relevant stepping stones.

## Suitability Decision

GRDzhadzha is appropriate as a host for the target project, but not sufficient
out of the box. The fixed analytic-background design is aligned with the goal,
and the underlying local GRChombo tree already has much of the needed GRHD
numerics. The required work is to port or expose those GRHD components through
GRDzhadzha-style fixed-background applications, then extend the analytic
background support from single-hole/static tests toward a binary black-hole
background suitable for disk evolution.

## Near-Term Technical Milestones

- Verify that the existing GRDzhadzha scalar-field build/test path works in this
  environment.
- Verify that the local GRChombo GRHD unit and disk-related executables run.
- Add GRDzhadzha-facing GRHD source/include paths and a small fixed-background
  GRHD target once the upstream GRHD tests are confirmed runnable.
- Start with a single-black-hole accretion/disk test, then generalize the
  metric interface toward time-dependent binary backgrounds.

## Progress

- 2026-06-24: Inspected repository layout and README. Found GRDzhadzha is a
  thin fixed-background layer with scalar-field examples/tests.
- 2026-06-24: Located local GRChombo and Chombo installations. `CHOMBO_HOME`
  is set to `/home/mjhatto/Chombo/lib`; `GRCHOMBO_SOURCE` is not set in the
  default shell but `/home/mjhatto/GRChombo/Source` exists.
- 2026-06-24: Found local GRChombo GRHD module and existing GRHD test
  executables, including Michel accretion and Schwarzschild disk variants.
- 2026-06-24: Built `Tests/KerrBHScalarTest` in GRDzhadzha using
  `/home/mjhatto/Chombo/env_grchombo.sh` and
  `GRCHOMBO_SOURCE=/home/mjhatto/GRChombo/Source`.
- 2026-06-24: Ran
  `Tests/KerrBHScalarTest/KerrBHScalarTest3d.Linux.64.mpicxx.gfortran.OPTHIGH.MPI.ex`.
  It reported a minimum convergence factor of 16 and
  `Fixed Background test passed...`.
- 2026-06-24: Ran local GRChombo GRHD core test
  `/home/mjhatto/GRChombo/Tests/GRHDTest/GRHDTest3d.Linux.64.mpicxx.gfortran.OPTHIGH.MPI.ex`;
  it reported `GRHD test passed...`.
- 2026-06-24: Ran local GRChombo Michel accretion test from
  `/home/mjhatto/GRChombo`; it passed with mass-flux relative error
  `2.11758e-16`, Bernoulli relative error `4.79408e-14`, and primitive
  recovery relative error `7.61464e-07`.
- 2026-06-24: Ran local GRChombo fixed Schwarzschild disk test. It passed
  on a `72^2` grid to `t = 120` with 560 steps, no atmosphere resets, and
  diagnostics under
  `/home/mjhatto/GRChombo/Tests/GRHDTest/output`.
- 2026-06-24: Ran local GRChombo fixed Schwarzschild Keplerian disk test.
  It passed on a `72^2` grid to `t = 200` with 789 steps and 393
  atmosphere resets, writing diagnostics and frames under the same output
  directory.
- 2026-06-24: Added `Examples/GRHDSchwarzschildDisk` to GRDzhadzha by
  bringing in the proven fixed Schwarzschild disk driver and a narrow
  makefile that builds against `$(GRCHOMBO_SOURCE)/Matter/GRHD`.
- 2026-06-24: Fresh-built the new GRDzhadzha example after cleaning stale
  objects. The successful compile used `utils`, `simd`, `BoxUtils`, and
  `Matter/GRHD` source paths, avoiding unnecessary `GRChomboCore`
  dependencies.
- 2026-06-24: Ran
  `Examples/GRHDSchwarzschildDisk/Main_GRHDSchwarzschildDisk3d.Linux.64.mpicxx.gfortran.OPTHIGH.MPI.ex`.
  It passed on a `72^2` grid to `t = 120` with 560 steps, 81 frames,
  final disk mass `0.00436951`, max velocity `0.832251`, and zero
  atmosphere resets. Outputs are under
  `Examples/GRHDSchwarzschildDisk/output`.
- 2026-06-24: Ported the local GRChombo GRHD header module into
  `Source/Matter/GRHD`.
- 2026-06-24: Added `Source/Matter/GRHD/FixedBGMetric.hpp`, which converts
  GRDzhadzha `ADMFixedBGVars` analytic-background data into the GRHD
  `CellGeometryData` and `StaticMetricDerivatives` formats.
- 2026-06-24: Added `Tests/GRHDFixedBGTest`, a direct GRHD/GRDzhadzha coupling
  test using `KerrSchild` analytic metric data, Valencia conserved recovery,
  and fixed-background metric source terms.
- 2026-06-24: Built and ran `Tests/GRHDFixedBGTest`; it reported
  `GRHD fixed-background coupling test passed...`.
- 2026-06-24: Rebuilt `Examples/GRHDSchwarzschildDisk` after cleaning so it
  uses `Source/Matter/GRHD` from this repository rather than
  `/home/mjhatto/GRChombo/Source/Matter/GRHD`. The rebuilt executable
  passed the same fixed Schwarzschild disk run.
- 2026-06-24: Added `Examples/GRHDKerrSchildDisk`, a second disk driver that
  uses GRDzhadzha `KerrSchild` analytic background data through
  `FixedBGMetric.hpp` for geometry and metric-source derivatives. This keeps
  the finite-volume evolution loop close to the Schwarzschild benchmark while
  replacing the standalone Schwarzschild-isotropic metric helper.
- 2026-06-24: Built and ran `Examples/GRHDKerrSchildDisk`. The smoke test
  passed to `t = 45` on a `72^2` grid with 395 steps, 31 frames, final disk
  mass `0.0126274`, max velocity `0.516582`, and 57380 atmosphere resets.
  This proves end-to-end GRHD evolution on a GRDzhadzha analytic metric, but
  the reset count shows this is still a stability/initial-data tuning target.
- 2026-06-24: Reran `Tests/GRHDFixedBGTest` after adding the Kerr-Schild disk
  example; it still reported `GRHD fixed-background coupling test passed...`.
- 2026-06-24: Added a low-density atmosphere-state precheck to
  `Examples/GRHDKerrSchildDisk` before primitive recovery. Cells whose
  conserved variables remain within `1000` times the local atmosphere state
  are repaired as atmosphere rather than counted as failed primitive
  recoveries.
- 2026-06-24: Rebuilt and reran `Examples/GRHDKerrSchildDisk` after the
  atmosphere repair change. The smoke test still passed to `t = 45` on a
  `72^2` grid with 390 steps, final disk mass `0.0126265`, max velocity
  `0.328439`, and failed recovery resets reduced from the original `57380`
  to `30612`.
- 2026-06-24: Reran `Tests/GRHDFixedBGTest` after the atmosphere repair
  change; it still reported `GRHD fixed-background coupling test passed...`.
- 2026-06-24: Added `Source/Background/CircularBinaryPN.hpp`, a softened
  circular-binary post-Newtonian analytic background with the standard
  GRDzhadzha `compute_metric_background` interface. It provides ADM spatial
  metric, lapse, shift, spatial derivatives, and a frozen-time extrinsic
  curvature approximation.
- 2026-06-24: Added `Tests/GRHDBinaryPNTest`, which compares the new
  GRDzhadzha binary PN background against the ported GRHD circular-binary PN
  geometry/derivative helper, then exercises Valencia primitive recovery and
  fixed-background source terms on that metric.
- 2026-06-24: Built and ran `Tests/GRHDBinaryPNTest`; it reported
  `GRHD binary PN fixed-background test passed...`.
- 2026-06-24: Reran `Tests/GRHDFixedBGTest` after adding the binary PN
  background; it still reported `GRHD fixed-background coupling test passed...`.

- 2026-06-24: Added `Examples/GRHDBinaryPNDisk`, a fixed-background
  GRHD smoke example using `CircularBinaryPN` through `FixedBGMetric.hpp`.
  The driver evolves a circumbinary torus on an instantaneous softened PN
  binary metric with central excision disabled by default.
- 2026-06-24: Updated the binary-PN disk initial data to use the PN metric
  shift in the torus potential and primitive velocity. The driver now solves
  for the constant specific angular momentum that makes the radial derivative
  of the PN torus potential vanish at the requested pressure maximum, instead
  of using only the Schwarzschild total-mass estimate.
- 2026-06-24: Built and ran `Examples/GRHDBinaryPNDisk` from its example
  directory. The smoke test passed on an `80^2` grid to `t = 15` with 38
  steps and 16 frames. Diagnostics: initial disk mass `0.0025287`, final disk
  mass `0.00117602`, relative disk-mass change `-0.534931`, max velocity
  `0.374285`, max Lorentz factor `1.07838`, and 1124 atmosphere resets.
  Outputs are under `Examples/GRHDBinaryPNDisk/output`.
- 2026-06-24: Reran `Tests/GRHDBinaryPNTest` and `Tests/GRHDFixedBGTest`
  after adding the binary-PN disk example; both still reported their pass
  messages.
- 2026-06-24: Current binary-PN disk caveats: the background is frozen at a
  single binary phase/time, the `CircularBinaryPN` extrinsic curvature still
  omits explicit time derivatives of the spatial metric, and the smoke torus
  loses about half of its disk mass by `t = 15`. This is now a useful
  end-to-end BBH-path integration test, not yet a stable production
  circumbinary-disk setup.

- 2026-06-24: Added `Tests/GRHDMichelAccretion`, porting the local
  GRChombo Michel accretion profile benchmark into GRDzhadzha. The test now
  samples the GRDzhadzha `KerrSchild` analytic background through
  `FixedBGMetric.hpp` on the radial x-axis, rather than using the old direct
  Schwarzschild radial-line geometry helper.
- 2026-06-24: Adapted the Michel primitive velocity for the nonzero
  Kerr-Schild shift using the ADM relation `v^i = u^i/W + beta^i/alpha` and
  the invariant Michel `u_t`. This keeps the analytic mass-flux and Bernoulli
  checks while exercising primitive-to-conserved and recovery on the actual
  GRDzhadzha fixed-background metric adapter.
- 2026-06-24: Built and ran `Tests/GRHDMichelAccretion`. It reported
  `GRHD Michel accretion profile test passed...` with
  `max_mass_flux_rel_error = 2.11758e-16`,
  `max_bernoulli_rel_error = 4.79408e-14`,
  `max_recovery_rel_error = 1.55437e-07`, and
  `max_velocity_squared = 0.0355675`. The test writes
  `grhd_michel_profile.csv` under its own `output` directory and passes both
  from the repository root and from `Tests/GRHDMichelAccretion`.
- 2026-06-24: Reran `Tests/GRHDFixedBGTest` and `Tests/GRHDBinaryPNTest`
  after adding the Michel benchmark; both still reported their pass messages.

- 2026-06-24: Extended `GRHD::CellGeometryData` to carry physical
  `extrinsic_curvature_LL`. `FixedBGMetric.hpp` now copies GRDzhadzha
  `ADMFixedBGVars::K_tensor` into GRHD geometry, and the CCZ4 FArrayBox helper
  reconstructs physical `K_ij = A_ij / chi + gamma_ij K / 3` for the
  NR-coupled path.
- 2026-06-24: Added the `alpha S^{ij} K_ij` contribution to the GRHD
  fixed-background tau source. `Tests/GRHDFixedBGTest` and
  `Tests/GRHDBinaryPNTest` now explicitly verify that nonzero `K_ij` is
  carried through the adapter and changes only the tau source by the expected
  contraction.
- 2026-06-24: Fixed several unsafe GRHD `Tensor` initializations. GRChombo
  `Tensor()` is uninitialized and scalar construction such as `Tensor<1> x =
  0.0` is not a reliable full-vector zero. Added explicit zeroing for
  `CellGeometryData`, `StaticMetricDerivatives`, atmosphere velocities,
  zero-conserved momenta, zero-shift flux helpers, disk-driver basis and
  inverse-dx vectors, artificial z-flux momenta, and the circular-binary PN
  state/potential accumulators. The binary PN regression now also rejects
  non-finite `check_close` inputs.
- 2026-06-24: Rebuilt and reran `Tests/GRHDFixedBGTest`,
  `Tests/GRHDBinaryPNTest`, and `Tests/GRHDMichelAccretion`; all passed. The
  Michel diagnostics remain `max_mass_flux_rel_error = 2.11758e-16`,
  `max_bernoulli_rel_error = 4.79408e-14`,
  `max_recovery_rel_error = 1.55437e-07`, and
  `max_velocity_squared = 0.0355675`.
- 2026-06-24: Rebuilt and reran the disk smoke examples after the `K_ij` source
  and tensor-initialization fixes. `Examples/GRHDSchwarzschildDisk` remains
  stable to `t = 120` with final disk mass `0.00436951`, max velocity
  `0.832251`, and zero atmosphere resets. `Examples/GRHDKerrSchildDisk` now
  passes to `t = 45` with final disk mass `0.0195337`, max velocity
  `0.857755`, and zero atmosphere resets. `Examples/GRHDBinaryPNDisk` now
  passes to `t = 15` with final disk mass `0.00245159`, relative disk-mass
  change `-0.0304924`, max velocity `0.289659`, and zero atmosphere resets.
- 2026-06-24: Current caveat after the source-term fix: the Kerr-Schild disk
  smoke run no longer has primitive-recovery resets, but it gains about 39%
  disk mass by `t = 45`, so that setup still needs initial-data/source/boundary
  tuning. The binary PN smoke run is much cleaner than before, but it is still
  a frozen-phase softened PN stepping stone rather than a merger disk model.

- 2026-06-24: Added `GRHD::compute_metric_volume_time_source_terms`, the
  non-densitized conserved-variable source `-q d_t log(sqrt(gamma))` needed
  when an analytic background has a time-varying spatial volume element. The
  binary PN coupling test now has a direct regression for this helper.
- 2026-06-24: Updated `Examples/GRHDBinaryPNDisk` to evolve the prescribed
  circular-binary PN phase during the hydro run. The RHS now evaluates the
  analytic background at the current SSPRK stage time and, when enabled, adds
  the finite-difference `d_t log(sqrt(gamma))` volume source using
  `metric_time_derivative_step = 1.0e-3`. The initial torus is still seeded at
  the base phase.
- 2026-06-24: Rebuilt and ran `Tests/GRHDBinaryPNTest`,
  `Tests/GRHDFixedBGTest`, and `Tests/GRHDMichelAccretion`; all passed after
  the time-volume-source addition.
- 2026-06-24: Rebuilt and reran `Examples/GRHDBinaryPNDisk` with moving binary
  phase enabled. The smoke run advanced from `binary_time = 0` to
  `final_binary_time = 15` on an `80^2` grid with 30 steps and 16 frames.
  Diagnostics: initial disk mass `0.0025287`, final disk mass `0.00245176`,
  relative disk-mass change `-0.0304254`, max velocity `0.293462`, max Lorentz
  factor `1.04606`, and zero atmosphere resets.
- 2026-06-24: Superseded moving-binary caveat: at this point the driver had
  the time-dependent determinant source and stage-time metric evaluation, but
  `CircularBinaryPN::K_tensor` still used the frozen-time ADM approximation
  and omitted explicit `d_t gamma_ij`.

- 2026-06-24: Updated `CircularBinaryPN::K_tensor` to include the explicit
  time derivative of the conformally flat spatial metric for the prescribed
  circular motion. The binary PN regression now checks the ADM identity
  `K_ij = (D_i beta_j + D_j beta_i - d_t gamma_ij) / (2 alpha)` using a
  centered finite difference in background time.
- 2026-06-24: Rebuilt and reran `Tests/GRHDBinaryPNTest`,
  `Tests/GRHDFixedBGTest`, and `Tests/GRHDMichelAccretion`; all passed. The
  Michel diagnostics remain `max_mass_flux_rel_error = 2.11758e-16`,
  `max_bernoulli_rel_error = 4.79408e-14`,
  `max_recovery_rel_error = 1.55437e-07`, and
  `max_velocity_squared = 0.0355675`.
- 2026-06-24: Rebuilt and reran `Examples/GRHDBinaryPNDisk` from its example
  directory after the explicit `d_t gamma_ij` correction. The moving-phase
  smoke run advanced from `binary_time = 0` to `final_binary_time = 15` on an
  `80^2` grid with 30 steps and 16 frames. Diagnostics: initial disk mass
  `0.0025287`, final disk mass `0.00245168`, relative disk-mass change
  `-0.0304577`, max velocity `0.29276`, max Lorentz factor `1.04582`, and
  zero atmosphere resets.
- 2026-06-24: Current caveat: the GRHD path now uses GRDzhadzha analytic
  fixed-background metrics with nonzero `K_ij` and a moving binary phase, but
  the binary metric is still a softened circular PN stepping stone. It is not
  yet a merger spacetime, and the disk setup still needs longer convergence
  and boundary/initial-data tuning before it should be treated as production
  circumbinary-disk evolution.

- 2026-06-24: Replaced the remaining scalar zero assignment to a rank-2 `Tensor` in `Tests/GRHDBinaryPNTest` with an explicit component loop and reran the binary PN test; it still reported `GRHD binary PN fixed-background test passed...`.

- 2026-06-24: Added `Tests/GRHDAMRHookCompileTest`, a compile-time regression for the GRChombo-facing GRHD CCZ4 level operator and AMR hook classes. The test explicitly instantiates `GRHDCCZ4LevelOperator<MovingPunctureGauge, FourthOrderDerivatives>` and `GRHDCCZ4AMRLevelHooks<MovingPunctureGauge, FourthOrderDerivatives>` against the current GRDzhadzha/GRChombo headers.
- 2026-06-24: Built and ran `Tests/GRHDAMRHookCompileTest`; it reported `GRHD AMR hook compile test passed...`. The test makefile had to include `$(GRCHOMBO_SOURCE)/Matter` because `MatterCCZ4RHS.hpp` is supplied by GRChombo's Matter directory, not by the CCZ4 include directory.

- 2026-06-24: Added `Source/Matter/GRHD/FixedBGFArrayBoxFiniteVolume.hpp`, a reusable finite-volume bridge for GRDzhadzha analytic fixed backgrounds. It supplies FArrayBox recovery, MUSCL-HLLE directional flux RHS, static metric source RHS, atmosphere recovery, and LevelData RHS helpers that call `background.compute_metric_background` directly instead of reading CCZ4 metric variables from the hydro state.
- 2026-06-24: Extended `Tests/GRHDFixedBGTest` to fill matching fixed-background and CCZ4-geometry FArrayBox states from `KerrSchild`, compare the new fixed-background directional flux RHS against the existing CCZ4-geometry flux RHS in all spatial directions, compare the fixed-background source RHS against the direct cell-level source calculation, and explicitly instantiate the new fixed-background LevelData RHS helper for `KerrSchild`.
- 2026-06-24: Rebuilt and reran `Tests/GRHDFixedBGTest`; it reported `GRHD fixed-background coupling test passed...` after the fixed-background FArrayBox/LevelData helper additions.

- 2026-06-24: Extended `FixedBGFArrayBoxFiniteVolume.hpp` with `compute_max_inverse_dt_from_fixed_background`, `compute_leveldata_max_inverse_dt_from_fixed_background`, and `FixedBGLevelDataFiniteVolumeOperator`. The operator now exposes the Chombo-style fixed-background GRHD surface: `compute_rhs`, `add_conserved_rhs_to`, primitive recovery with atmosphere repair, CFL timestep estimation, conserved update, and SSPRK2 advance on a supplied analytic background.
- 2026-06-24: Extended `Tests/GRHDFixedBGTest` with a one-box `LevelData<FArrayBox>` regression for the new fixed-background operator. The test initializes MPI for the LevelData layout, fills state from `KerrSchild`, checks recovery diagnostics, verifies positive finite CFL quantities, compares operator RHS against the lower-level fixed-background LevelData RHS helper, and runs both a conserved update and an SSPRK2 step with finite-state checks.
- 2026-06-24: Rebuilt and reran `Tests/GRHDFixedBGTest`; it reported `GRHD fixed-background coupling test passed...` with the fixed-background LevelData operator enabled.

- 2026-06-24: Added reusable time-dependent fixed-background GRHD support to `FixedBGFArrayBoxFiniteVolume.hpp`. New helpers finite-difference `d_t log(sqrt(gamma))`, add the corresponding metric-volume source to FArrayBox/LevelData RHS objects, and `TimeDependentFixedBGLevelDataFiniteVolumeOperator` evaluates analytic backgrounds through a time-indexed factory for RHS, recovery, CFL, and SSPRK2 stage times.
- 2026-06-24: Extended `Tests/GRHDFixedBGTest` with a `CircularBinaryPNTimeFactory` regression for the time-dependent operator. The test initializes a one-box Chombo `LevelData` state from the binary PN background, verifies a nonzero finite determinant time derivative, compares the operator RHS to the lower-level fixed-background RHS plus volume-time source, checks finite positive timestep output, and advances one SSPRK2 step.
- 2026-06-24: Rebuilt and reran `Tests/GRHDFixedBGTest`; it reported `GRHD fixed-background coupling test passed...` with the time-dependent binary-PN LevelData operator path enabled.

- 2026-06-24: Added `Tests/GRHDLevelDataBinaryPNSmoke`, a moving
  `CircularBinaryPN` LevelData smoke test that initializes a smooth low-density
  gas ring, recovers primitives on the analytic background, estimates the CFL
  timestep, and advances four SSPRK2 steps through
  `TimeDependentFixedBGLevelDataFiniteVolumeOperator`.
- 2026-06-24: Fixed fixed-background physical ghost handling in
  `FixedBGFArrayBoxFiniteVolume.hpp`. Outflow ghost cells now recover the
  primitive state on the interior metric and then recompute conserved variables
  on the target ghost-cell metric, avoiding metric-inconsistent conserved copies
  when the analytic background varies across the boundary.
- 2026-06-24: Rebuilt and reran `Tests/GRHDFixedBGTest`; it still reported
  `GRHD fixed-background coupling test passed...` after the boundary helper
  change.
- 2026-06-24: Built and ran `Tests/GRHDLevelDataBinaryPNSmoke`; it reported
  `GRHD LevelData binary PN smoke test passed...` with final time `0.2`,
  initial mass `0.00296122`, final mass `0.00103393`, max density
  `1.15766e-05`, and max speed squared `0.00223438`.

- 2026-06-24: Added `Examples/GRHDLevelDataBinaryPNDisk`, a Chombo
  `LevelData<FArrayBox>` binary PN disk driver using
  `TimeDependentFixedBGLevelDataFiniteVolumeOperator` rather than the older
  hand-rolled vector finite-volume loop. The example evolves a smooth
  circumbinary gas ring on the moving `CircularBinaryPN` background and writes
  diagnostics, a final midplane slice, and a summary under its `output/`
  directory.
- 2026-06-24: Built and ran `Examples/GRHDLevelDataBinaryPNDisk` with a
  `56^2 x 9` single-box grid, `dx = 1.4285714285714286`, `max_dt = 0.05`, and
  final time `t = 2`. It completed 40 SSPRK2 steps with zero failed
  recoveries, zero floored primitives, and zero conserved resets. Diagnostics:
  initial disk mass `0.098471975732142877`, final disk mass
  `0.09847193894796033`, relative disk-mass change `-3.7354975640190207e-07`,
  max density `1.999572520121098e-05`, and max velocity
  `0.084521235761513006`.
- 2026-06-24: Superseded caveat: at this point the new LevelData disk example
  was still a deliberately mild smooth-ring stability check for the shared
  fixed-background Chombo operator, not yet torus initial data, AMR-enabled,
  or a true BBH merger spacetime.

- 2026-06-24: Extended `Tests/GRHDLevelDataBinaryPNSmoke` from a
  single-box LevelData check to a two-box x-split layout. The test now poisons
  internal ghost cells, recomputes the RHS through
  `TimeDependentFixedBGLevelDataFiniteVolumeOperator`, and compares it against
  a clean RHS to verify that the fixed-background LevelData path repairs
  internal ghosts through Chombo `exchange()` before flux/source evaluation.
- 2026-06-24: Rebuilt and ran `Tests/GRHDLevelDataBinaryPNSmoke` on one and
  two MPI ranks. Both runs reported `GRHD LevelData binary PN smoke test
  passed...` with `boxes = 2`, `exchange rhs max relative error = 0`, final
  time `0.2`, initial mass `0.00296122`, final mass `0.00103393`, max density
  `1.15766e-05`, and max speed squared `0.00223438`.

- 2026-06-24: Replaced the smooth Gaussian-ring initial data in
  `Examples/GRHDLevelDataBinaryPNDisk` with the constant-specific-angular-
  momentum torus construction from the earlier hand-rolled binary PN disk
  driver. The LevelData example now computes the torus potential from the
  analytic `CircularBinaryPN` geometry, solves for the specific angular
  momentum that places the pressure maximum at the configured radius, records
  the torus parameters in the summary file, and still evolves through
  `TimeDependentFixedBGLevelDataFiniteVolumeOperator`.
- 2026-06-24: Built and ran the torus-enabled
  `Examples/GRHDLevelDataBinaryPNDisk` with a `56^2 x 9` single-box grid,
  `dx = 1.4285714285714286`, `max_dt = 0.05`, and final time `t = 2`. It
  completed 40 SSPRK2 steps with zero failed recoveries, zero floored
  primitives, and zero conserved resets. Torus parameters: `ell =
  4.9353189499884671`, surface potential `-0.023380221245975233`, max
  enthalpy `1.0014653367693895`, and polytropic constant
  `0.79550718821504551`. Diagnostics: initial disk mass
  `0.019469376019477257`, final disk mass `0.019469695978953833`, relative
  disk-mass change `1.6433987214412347e-05`, max density
  `1.9894710361761495e-05`, max pressure `1.162435841077777e-08`, and max
  velocity `0.28737021599042162`.
- 2026-06-24: Current caveat for the LevelData binary PN disk example: the
  initial data is now a torus-like constant-ell disk rather than a smooth
  transport ring, but it is still an instantaneous softened circular PN
  background with a short smoke evolution. It is not yet AMR-enabled and not a
  true BBH merger spacetime.

- 2026-06-24: Converted `Examples/GRHDLevelDataBinaryPNDisk` from a
  single-box LevelData layout to a two-box x-split layout. With more than one
  MPI rank, the second box is assigned to rank 1. The example now records
  `num_boxes`, `mpi_ranks`, and `slice_output` in its summary file, and writes
  rank-local final midplane slices as
  `leveldata_binary_pn_disk_final_slice_rankNNNN.csv` so parallel runs do not
  silently omit non-root boxes.
- 2026-06-24: Rebuilt and ran the torus-enabled
  `Examples/GRHDLevelDataBinaryPNDisk` on one and two MPI ranks. The two-rank
  run completed 40 SSPRK2 steps to `t = 2` with `num_boxes = 2`,
  `mpi_ranks = 2`, zero failed recoveries, zero floored primitives, and zero
  conserved resets. Diagnostics: initial disk mass `0.019469376019477229`,
  final disk mass `0.019469695978953892`, relative disk-mass change
  `1.6433987218867374e-05`, max density `1.9894710361761495e-05`, max pressure
  `1.162435841077777e-08`, and max velocity `0.28737021599042162`. The output
  directory contains both `leveldata_binary_pn_disk_final_slice_rank0000.csv`
  and `leveldata_binary_pn_disk_final_slice_rank0001.csv`, each with 1569 CSV
  lines including the header.

- 2026-06-24: Added `Examples/GRHDLevelDataKerrSchildDisk`, a single-black-hole
  Kerr-Schild torus disk stepping stone that uses the same fixed-background
  `LevelData<FArrayBox>` GRHD operator path as the binary PN disk example. The
  driver uses a static `KerrSchild` background factory, constant-ell torus
  initial data, a two-box x-split layout, MPI-aware diagnostics, and rank-local
  final midplane slice output.
- 2026-06-24: Built and ran `Examples/GRHDLevelDataKerrSchildDisk` on one and
  two MPI ranks. The two-rank run completed 40 SSPRK2 steps to `t = 2` with
  `num_boxes = 2`, `mpi_ranks = 2`, zero failed recoveries, zero floored
  primitives, and zero conserved resets. Torus parameters: `ell =
  3.952846210518592`, surface potential `-0.038999967542758737`, max enthalpy
  `1.0058220833394507`, and polytropic constant `3.1607131163777216`.
  Diagnostics: initial disk mass `0.028799123650408626`, final disk mass
  `0.028801497749415979`, relative disk-mass change `8.2436501755154781e-05`,
  max density `1.9556944512350192e-05`, max pressure
  `4.5283557131844508e-08`, max velocity `0.51157954453700194`, and max
  Lorentz factor `1.1638247126610648`. The output directory contains both
  `leveldata_kerr_schild_disk_final_slice_rank0000.csv` and
  `leveldata_kerr_schild_disk_final_slice_rank0001.csv`, each with 1569 CSV
  lines including the header.
- 2026-06-24: Current caveat for the LevelData Kerr-Schild disk example: this
  is a short, coarse, non-AMR fixed-background smoke evolution. It gives the
  single-BH disk stepping stone through the same Chombo operator path intended
  for BBH disks, but it is not yet a long-duration production accretion-disk
  model or convergence study.

- 2026-06-24: Added `Tests/GRHDLevelDataMichelAccretion`, a finite-volume
  fixed-background LevelData smoke test for radial Michel accretion on a static
  Schwarzschild/Kerr-Schild background. The test initializes a 3D Michel inflow
  with an inner atmosphere region, recovers primitives through
  `FixedBGLevelDataFiniteVolumeOperator`, advances five SSPRK2 steps to
  `t = 0.1`, and checks finite positive diagnostics, bounded velocity, and
  modest mass drift.
- 2026-06-24: Built and ran `Tests/GRHDLevelDataMichelAccretion` on one and
  two MPI ranks. Both runs reported `GRHD LevelData Michel accretion test
  passed...` with `boxes = 2`, `steps = 5`, final time `0.1`, initial mass
  `0.00172117`, final mass `0.00172335`, relative mass change `0.00126778`,
  max density `2.06653e-06`, max pressure `1.75256e-07`, and max speed squared
  `0.0357728`.
- 2026-06-24: Current caveat for the LevelData Michel test: it is a short
  Cartesian-grid finite-volume smoke around a Michel profile with an inner
  atmosphere region, not a high-resolution steady-state accretion convergence
  test. It does, however, exercise the same fixed-background LevelData operator
  and MPI two-box exchange path as the disk stepping stones.

- 2026-06-24: Added `run_grhd_fixed_bg_smoke_suite.sh`, a repository-root
  runner that sources the local GRChombo/Chombo environment by default,
  builds each fixed-background GRHD smoke target, discovers the Chombo
  executable suffix automatically, and runs the current core suite. The suite
  covers `Tests/GRHDBinaryPNTest`, `Tests/GRHDFixedBGTest`,
  `Tests/GRHDMichelAccretion`, `Tests/GRHDLevelDataMichelAccretion` on two MPI
  ranks, `Tests/GRHDLevelDataBinaryPNSmoke` on two MPI ranks,
  `Tests/GRHDAMRHookCompileTest`, `Examples/GRHDLevelDataKerrSchildDisk` on two
  MPI ranks, and `Examples/GRHDLevelDataBinaryPNDisk` on two MPI ranks.
- 2026-06-24: Added `GRHD_FIXED_BACKGROUND.md`, a short runbook documenting the
  build environment, smoke-suite command, current single-BH accretion/disk and
  binary-PN disk stepping stones, and main non-production caveats.
- 2026-06-24: Ran `./run_grhd_fixed_bg_smoke_suite.sh` from the repository root.
  The first attempt exposed an environment-loading bug in the runner
  (`mpicxx`/`gfortran` and `libopenblas.so.0` unavailable when it skipped
  sourcing the Chombo environment). The runner now sources
  `/home/mjhatto/Chombo/env_grchombo.sh` by default unless
  `GRHD_SOURCE_ENV=0` is set.
- 2026-06-24: Reran `./run_grhd_fixed_bg_smoke_suite.sh`; it completed
  successfully and ended with `GRHD fixed-background smoke suite passed.` The
  run rebuilt or checked all suite targets and reran the two-rank LevelData
  Michel, binary-PN exchange, Kerr-Schild disk, and binary-PN disk smoke paths.

- 2026-06-24: Added `Source/Matter/GRHD/FixedBGAMRLevelHooks.hpp`, which provides
  GRAMRLevel-shaped wrappers for static and time-dependent analytic
  fixed-background GRHD evolution. The wrappers expose `specific_eval_rhs`,
  post-update primitive recovery, CFL and coupled-timestep helpers,
  `update_conserved`, and SSPRK2 advancement while delegating the actual
  finite-volume work to `FixedBGLevelDataFiniteVolumeOperator` and
  `TimeDependentFixedBGLevelDataFiniteVolumeOperator`.
- 2026-06-24: Added `Tests/GRHDFixedBGAMRHookTest`, a two-box LevelData
  regression for the new hooks. It compares static Kerr-Schild hook RHS,
  timestep, coupled timestep, recovery, and SSPRK2 advancement against the
  underlying static fixed-background LevelData operator, and performs the same
  comparisons for the time-dependent `CircularBinaryPN` hook with metric volume
  time sources enabled.
- 2026-06-24: Built and ran `Tests/GRHDFixedBGAMRHookTest` on one and two MPI
  ranks; both runs reported `GRHD fixed-background AMR hook test passed...`.
- 2026-06-24: Added `Tests/GRHDFixedBGAMRHookTest` to
  `run_grhd_fixed_bg_smoke_suite.sh` and documented the hook in
  `GRHD_FIXED_BACKGROUND.md`. Reran the full smoke suite; it completed
  successfully and ended with `GRHD fixed-background smoke suite passed.`

- 2026-06-24: Added `Source/Matter/GRHD/FixedBGTorus.hpp`, a reusable
  fixed-background constant-specific-angular-momentum torus helper. It now owns
  the azimuthal metric/shift projections, coordinate angular velocity,
  constant-ell potential, pressure-maximum angular-momentum solve,
  polytropic scaling, primitive construction, and `LevelData<FArrayBox>` torus
  fill routine previously duplicated in the LevelData Kerr-Schild and binary-PN
  disk examples.
- 2026-06-24: Refactored `Examples/GRHDLevelDataKerrSchildDisk` and
  `Examples/GRHDLevelDataBinaryPNDisk` to use `FixedBGTorus.hpp` while
  preserving their existing radii, density scale, velocity cap, diagnostics,
  and summary outputs.
- 2026-06-24: Rebuilt both refactored disk examples and reran
  `./run_grhd_fixed_bg_smoke_suite.sh`; the full suite passed. The two-rank
  Kerr-Schild disk run again completed 40 steps to `t = 2` with initial/final
  disk masses `0.0287991`/`0.0288015` and zero conserved resets. The two-rank
  binary-PN disk run completed 40 steps to `t = 2` with initial/final disk
  masses `0.0194694`/`0.0194697` and zero conserved resets.

- 2026-06-24: Added opt-in density-frame output to
  `Tests/GRHDLevelDataMichelAccretion`. When
  `GRHD_MICHEL_WRITE_FRAMES=1` is set, the test writes x-y midplane density CSV
  frames for the initial state and each SSPRK2 step. Added
  `plot_density_frames.py`, a dependency-free renderer that converts those CSV
  frames into PNG images plus an HTML flipbook. Generated six frames for the
  current short run under `Tests/GRHDLevelDataMichelAccretion/output/images`.


- 2026-06-25: Hardened `Tests/GRHDLevelDataMichelAccretion/plot_density_frames.py`.
  The renderer now groups frame CSVs by timestep so two-rank
  `GRHD_MICHEL_WRITE_FRAMES=1` outputs such as
  `grhd_leveldata_michel_density_frame_0000_rank0000.csv` and
  `..._rank0001.csv` render as one combined image instead of overwriting
  duplicate step numbers. It also has optional MP4 encoding through `ffmpeg`,
  with a clean PNG/HTML fallback when no encoder is installed.
- 2026-06-25: Verified the Michel visualization script on the existing
  single-file CSV frames and on synthetic rank-sharded CSV fixtures under
  `/tmp`. Both produced six PNG frames plus `density_movie.html`. A movie probe
  correctly reported `MP4 encoding skipped: ffmpeg not found: ffmpeg` in the
  current environment.

- 2026-06-25: Reran the actual two-rank
  `Tests/GRHDLevelDataMichelAccretion` executable with
  `GRHD_MICHEL_WRITE_FRAMES=1`. It passed with the same smoke diagnostics
  (`steps = 5`, `final time = 0.1`, relative mass change `0.00126778`) and
  wrote rank-local frame CSVs under `Tests/GRHDLevelDataMichelAccretion/output`.
  Rendered those rank-local CSVs into six PNGs and `density_movie.html`; MP4
  encoding was skipped because `ffmpeg` is not installed in the current PATH.

- 2026-06-25: Checked `/home/mjhatto/GRChombo` for the previously playable MP4
  workflow. The stable path used the static binary
  `/home/mjhatto/GRChombo/tools/ffmpeg-imageio-v4.2.2` with an image-sequence
  input and `-c:v libx264 -pix_fmt yuv420p -movflags +faststart`, rather than
  system `ffmpeg` or matplotlib's writer.
- 2026-06-25: Updated the GRDzhadzha Michel density renderer to default to that
  GRChombo static ffmpeg binary and to encode from `density_%04d.png` using the
  same H.264/yuv420p/faststart options. Regenerated
  `Tests/GRHDLevelDataMichelAccretion/output/images/grhd_leveldata_michel_density.mp4`
  from the actual two-rank rank-local CSV frames with `--require-movie`; the
  encode completed successfully and a follow-up ffmpeg decode pass returned
  cleanly.
