# KITP circumbinary disk driver

This fixed-background GRHD driver initializes the KITP circumbinary disk,
including alpha viscosity and binary sink terms. The current defaults use a
binary separation of 8, `alpha = 0.03`, sink size 1, and sink rate `1e4`.

The first positional argument is the absolute final simulation time:

```sh
srun -n 16 ./Main_GRHDLevelDataBinaryPNKITPDisk3d.Linux.64.mpicxx.gfortran.OPTHIGH.MPI.ex 500
```

Checkpoints are written every 50 time units and at the final timestep under
`output/checkpoints`. A restart restores the complete state, physical ghost
zones, simulation time, timestep and frame counters, diagnostics baseline,
and primitive-recovery counters. The MPI rank count may differ from the run
that wrote the checkpoint.

```sh
srun -n 16 ./Main_GRHDLevelDataBinaryPNKITPDisk3d.Linux.64.mpicxx.gfortran.OPTHIGH.MPI.ex 750 \
  --restart output/checkpoints/checkpoint_step_00025000.bin
```

Useful options are `--checkpoint-interval`, `--output-dir`, `--num-cells`,
`--num-z-cells`, `--domain-length`, and `--no-frames`. Grid and physical
parameters that affect evolution are validated against checkpoint metadata;
the requested final time and MPI rank count are intentionally allowed to
change.
