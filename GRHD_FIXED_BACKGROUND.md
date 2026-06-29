# GRHD fixed-background runbook

This tree contains a GRHD port coupled to GRDzhadzha analytic fixed-background
metrics. The current verified path is finite-volume GRHD on Chombo
`LevelData<FArrayBox>` with analytic Kerr-Schild and circular-binary PN
backgrounds.

## Environment

On the current workstation the GRChombo/Chombo environment is loaded with:

```bash
source /home/mjhatto/Chombo/env_grchombo.sh
export GRCHOMBO_SOURCE=/home/mjhatto/GRChombo/Source
```

The smoke-suite runner sources `/home/mjhatto/Chombo/env_grchombo.sh`
automatically when that file exists. Set `GRHD_SOURCE_ENV=0` to disable that
behavior.

## Smoke suite

Run the currently useful fixed-background GRHD checks from the repository root:

```bash
./run_grhd_fixed_bg_smoke_suite.sh
```

Optional controls:

```bash
JOBS=4 ./run_grhd_fixed_bg_smoke_suite.sh
RUNNER=mpirun ./run_grhd_fixed_bg_smoke_suite.sh
```

The suite builds and runs:

- `Tests/GRHDBinaryPNTest`
- `Tests/GRHDFixedBGTest`
- `Tests/GRHDMichelAccretion`
- `Tests/GRHDLevelDataMichelAccretion` on two MPI ranks
- `Tests/GRHDLevelDataBinaryPNSmoke` on two MPI ranks
- `Tests/GRHDAMRHookCompileTest`
- `Tests/GRHDFixedBGAMRHookTest` on two MPI ranks
- `Examples/GRHDLevelDataKerrSchildDisk` on two MPI ranks
- `Examples/GRHDLevelDataBinaryPNDisk` on two MPI ranks

## Current stepping stones

`Examples/GRHDLevelDataKerrSchildDisk` is the single-black-hole disk stepping
stone. It uses a static `KerrSchild` analytic background, constant-ell torus
initial data, a two-box Chombo layout, and rank-local final slice output.

`Tests/GRHDLevelDataMichelAccretion` is the single-black-hole accretion
stepping stone through the same fixed-background LevelData finite-volume
operator. It initializes a Cartesian Michel inflow and advances a short smoke
run. Set `GRHD_MICHEL_WRITE_FRAMES=1` when running the executable to write
midplane density CSV frames, then run
`python3 Tests/GRHDLevelDataMichelAccretion/plot_density_frames.py` from the
repository root to render PNG frames and an HTML flipbook under the test output
directory. The renderer now merges MPI rank-local CSV shards by timestep. It
also writes an H.264 MP4 using the local GRChombo static ffmpeg binary at
`/home/mjhatto/GRChombo/tools/ffmpeg-imageio-v4.2.2` by default, falling back
cleanly to PNG/HTML output if no encoder is available.

`Examples/GRHDLevelDataBinaryPNDisk` is the binary stepping stone. It uses a
moving softened circular-binary PN background, constant-ell torus initial data,
nonzero `K_ij`, the metric volume time source, a two-box Chombo layout, and
rank-local final slice output.

`Source/Matter/GRHD/FixedBGTorus.hpp` provides the shared
fixed-background constant-ell torus initial-data helper used by both LevelData
disk examples. It computes the torus potential and pressure-maximum angular
momentum on the selected analytic background, then fills primitive and conserved
GRHD state on a Chombo `LevelData<FArrayBox>`.

`Source/Matter/GRHD/FixedBGAMRLevelHooks.hpp` provides GRAMRLevel-shaped hooks
for static and time-dependent analytic fixed backgrounds. The hook regression
compares those wrappers against the underlying LevelData operators on a two-box
layout.

## Main caveats

These are smoke/regression-scale runs. They are not yet AMR production runs,
long-duration disk evolutions, or true BBH merger spacetimes. The current
binary background is a prescribed softened circular PN metric, intended as an
intermediate target for the GRHD fixed-background coupling.
