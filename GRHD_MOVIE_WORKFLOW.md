# GRHD Movie Workflow

Date: 2026-06-25

Use the bundled GRChombo static ffmpeg binary. It is not on `PATH`, so `which ffmpeg` can fail even though MP4 encoding works.

```bash
FFMPEG=/home/mjhatto/GRChombo/tools/ffmpeg-imageio-v4.2.2
$FFMPEG -version
```

## Binary PN Mini-Disks

The one-rank mini-disk driver writes PPM density frames here:

```text
Examples/GRHDLevelDataBinaryPNMiniDisks/output/leveldata_binary_pn_minidisks_frames/frame_%04d.ppm
```

Encode those frames directly to H.264 MP4 with the same options that produced the earlier playable movie:

```bash
/home/mjhatto/GRChombo/tools/ffmpeg-imageio-v4.2.2 \
  -y \
  -framerate 10 \
  -i Examples/GRHDLevelDataBinaryPNMiniDisks/output/leveldata_binary_pn_minidisks_frames/frame_%04d.ppm \
  -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -movflags +faststart \
  Examples/GRHDLevelDataBinaryPNMiniDisks/output/leveldata_binary_pn_minidisks_density.mp4
```

The `pad` filter keeps H.264/yuv420p happy if a future frame size is odd. The current 96-cell, scale-4 frames are 384x384, so the filter is harmless.

To verify the finished movie decodes:

```bash
/home/mjhatto/GRChombo/tools/ffmpeg-imageio-v4.2.2 \
  -v error \
  -i Examples/GRHDLevelDataBinaryPNMiniDisks/output/leveldata_binary_pn_minidisks_density.mp4 \
  -f null -
```

## Michel Density Renderer

`Tests/GRHDLevelDataMichelAccretion/plot_density_frames.py` already defaults to the same ffmpeg binary and writes an MP4 when movie output is enabled. Use `--require-movie` when the encode must fail loudly instead of falling back to PNG/HTML only.

## Long Mini-Disk Runs

For long one-rank movie-producing runs, request wall time explicitly. A plain `srun -n 1 ... 100` was killed by the interactive job limit before `t=100`.

```bash
source /home/mjhatto/Chombo/env_grchombo.sh
srun -t 04:00:00 -n 1 \
  Examples/GRHDLevelDataBinaryPNMiniDisks/Main_GRHDLevelDataBinaryPNMiniDisks3d.Linux.64.mpicxx.gfortran.OPTHIGH.MPI.ex \
  100
```

The site may still route the command to the `interact` partition. Check the actual allocation with:

```bash
squeue -j JOBID -o '%i %P %t %M %l %R'
```

Use one MPI rank for this driver when a movie is needed, because the current PPM density-frame writer is guarded by `mpi_size() == 1`.
