#!/usr/bin/env python3
import argparse
import csv
import glob
import math
from pathlib import Path
import shutil
import struct
import subprocess
import zlib


PALETTE = [
    (68, 1, 84),
    (59, 82, 139),
    (33, 145, 140),
    (94, 201, 98),
    (253, 231, 37),
]


def interpolate_color(value):
    value = min(1.0, max(0.0, value))
    scaled = value * (len(PALETTE) - 1)
    index = int(math.floor(scaled))
    if index >= len(PALETTE) - 1:
        return PALETTE[-1]
    frac = scaled - index
    c0 = PALETTE[index]
    c1 = PALETTE[index + 1]
    return tuple(int(round(c0[i] + frac * (c1[i] - c0[i]))) for i in range(3))


def read_frame_rows(path):
    rows = []
    with open(path, newline="") as stream:
        for row in csv.DictReader(stream):
            rows.append({key: float(value) for key, value in row.items()})
    if not rows:
        raise ValueError(f"empty frame: {path}")
    return rows


def make_frame(rows, paths):
    x_values = sorted({row["x"] for row in rows})
    y_values = sorted({row["y"] for row in rows})
    rho = [[float("nan") for _ in x_values] for _ in y_values]
    x_index = {value: index for index, value in enumerate(x_values)}
    y_index = {value: index for index, value in enumerate(y_values)}
    for row in rows:
        rho[y_index[row["y"]]][x_index[row["x"]]] = row["rho"]
    return {
        "paths": [Path(path) for path in paths],
        "step": int(rows[0]["step"]),
        "time": rows[0]["time"],
        "z": rows[0]["z"],
        "x": x_values,
        "y": y_values,
        "rho": rho,
    }


def read_frames(paths):
    grouped_rows = {}
    grouped_paths = {}
    for path in paths:
        rows = read_frame_rows(path)
        step = int(rows[0]["step"])
        grouped_rows.setdefault(step, []).extend(rows)
        grouped_paths.setdefault(step, []).append(path)
    return [
        make_frame(grouped_rows[step], grouped_paths[step])
        for step in sorted(grouped_rows)
    ]


def png_chunk(kind, data):
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def write_png(path, width, height, pixels):
    raw = bytearray()
    stride = width * 3
    for y in range(height):
        raw.append(0)
        start = y * stride
        raw.extend(pixels[start:start + stride])
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    with open(path, "wb") as stream:
        stream.write(b"\x89PNG\r\n\x1a\n")
        stream.write(png_chunk(b"IHDR", ihdr))
        stream.write(png_chunk(b"IDAT", zlib.compress(bytes(raw), 9)))
        stream.write(png_chunk(b"IEND", b""))


def set_pixel(pixels, width, height, x, y, color):
    if 0 <= x < width and 0 <= y < height:
        offset = (y * width + x) * 3
        pixels[offset:offset + 3] = bytes(color)


def draw_projected_radius(pixels, width, height, extent, z, spherical_radius, color):
    projected_squared = spherical_radius * spherical_radius - z * z
    if projected_squared <= 0.0:
        return
    radius = math.sqrt(projected_squared)
    xmin, xmax, ymin, ymax = extent
    for sample in range(1440):
        angle = 2.0 * math.pi * sample / 1440.0
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        px = int(round((x - xmin) / (xmax - xmin) * (width - 1)))
        py = int(round((ymax - y) / (ymax - ymin) * (height - 1)))
        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                set_pixel(pixels, width, height, px + ox, py + oy, color)


def render_frame(frame, output_path, log_min, log_max, scale):
    nx = len(frame["x"])
    ny = len(frame["y"])
    width = nx * scale
    height = ny * scale
    pixels = bytearray([0, 0, 0] * width * height)
    log_span = max(1.0e-300, log_max - log_min)

    for j, row in enumerate(frame["rho"]):
        for i, rho in enumerate(row):
            if math.isfinite(rho) and rho > 0.0:
                value = (math.log10(rho) - log_min) / log_span
                color = interpolate_color(value)
            else:
                color = (0, 0, 0)
            x0 = i * scale
            y0 = (ny - 1 - j) * scale
            for py in range(y0, y0 + scale):
                start = (py * width + x0) * 3
                for px in range(scale):
                    offset = start + 3 * px
                    pixels[offset:offset + 3] = bytes(color)

    dx = min(frame["x"][k + 1] - frame["x"][k] for k in range(nx - 1)) if nx > 1 else 1.0
    dy = min(frame["y"][k + 1] - frame["y"][k] for k in range(ny - 1)) if ny > 1 else 1.0
    extent = (
        min(frame["x"]) - 0.5 * dx,
        max(frame["x"]) + 0.5 * dx,
        min(frame["y"]) - 0.5 * dy,
        max(frame["y"]) + 0.5 * dy,
    )
    draw_projected_radius(pixels, width, height, extent, frame["z"], 2.0, (255, 255, 255))
    draw_projected_radius(pixels, width, height, extent, frame["z"], 4.0, (220, 220, 220))
    draw_projected_radius(pixels, width, height, extent, frame["z"], 8.0, (180, 180, 180))
    write_png(output_path, width, height, pixels)


def write_metadata(output_dir, frames, log_min, log_max):
    path = output_dir / "density_frames_readme.txt"
    with open(path, "w") as stream:
        stream.write("GRHD LevelData Michel density frames\n")
        stream.write("background single black hole: Schwarzschild/Kerr-Schild, mass=1\n")
        stream.write("slice upper Cartesian midplane from the 18^3 test grid\n")
        stream.write("white/gray projected circles: horizon r=2M, atmosphere r=4M, sonic r=8M\n")
        stream.write(f"log10 rho range: {log_min:.17g} {log_max:.17g}\n")
        for frame in frames:
            inputs = " ".join(path.name for path in frame["paths"])
            stream.write(
                f"step {frame['step']} time {frame['time']:.17g} "
                f"z {frame['z']:.17g} inputs {inputs}\n"
            )
    return path


def write_html(output_dir, image_paths, movie_path=None):
    path = output_dir / "density_movie.html"
    names = [image.name for image in image_paths]
    with open(path, "w") as stream:
        stream.write("<!doctype html><meta charset='utf-8'>\n")
        stream.write("<title>GRHD Michel density frames</title>\n")
        stream.write("<style>body{margin:24px;background:#111;color:#eee;font-family:sans-serif}img{image-rendering:pixelated;max-width:90vw;max-height:80vh;border:1px solid #555}button,input{font:inherit}video{max-width:90vw;max-height:80vh;display:block;margin-bottom:16px}</style>\n")
        stream.write("<h1>GRHD Michel Density</h1><p>Single Schwarzschild/Kerr-Schild black hole, density midplane frames.</p>\n")
        if movie_path is not None:
            stream.write("<p><video controls loop src='" + movie_path.name + "'></video></p>\n")
        stream.write("<p><button id='play'>Pause</button> <input id='slider' type='range' min='0' max='" + str(len(names) - 1) + "' value='0'> <span id='label'></span></p>\n")
        stream.write("<img id='frame' src='" + names[0] + "' alt='density frame'>\n")
        stream.write("<script>\n")
        stream.write("const frames = " + repr(names) + "; let i = 0; let playing = true; const img = document.getElementById('frame'); const slider = document.getElementById('slider'); const label = document.getElementById('label'); function show(n){i=n; img.src=frames[i]; slider.value=i; label.textContent=frames[i];} document.getElementById('play').onclick=()=>{playing=!playing; document.getElementById('play').textContent=playing?'Pause':'Play';}; slider.oninput=()=>show(Number(slider.value)); show(0); setInterval(()=>{if(playing) show((i+1)%frames.length);}, 350);\n")
        stream.write("</script>\n")
    return path


def resolve_ffmpeg(ffmpeg):
    candidate = Path(ffmpeg)
    if candidate.exists():
        return candidate
    if not candidate.is_absolute():
        path = shutil.which(ffmpeg)
        if path is not None:
            return Path(path)
    return None


def encode_mp4(image_paths, movie_path, fps, ffmpeg):
    ffmpeg_path = resolve_ffmpeg(ffmpeg)
    if ffmpeg_path is None:
        return None

    movie_path = Path(movie_path)
    movie_path.parent.mkdir(parents=True, exist_ok=True)
    frame_pattern = image_paths[0].parent / "density_%04d.png"
    command = [
        str(ffmpeg_path),
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_pattern),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(movie_path),
    ]
    subprocess.run(command, check=True)
    return movie_path


def main():
    parser = argparse.ArgumentParser(
        description="Render GRHD LevelData Michel density CSV frames to PNGs."
    )
    parser.add_argument(
        "--input-glob",
        default="Tests/GRHDLevelDataMichelAccretion/output/grhd_leveldata_michel_density_frame_*.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="Tests/GRHDLevelDataMichelAccretion/output/images",
    )
    parser.add_argument("--scale", type=int, default=32, help="Pixels per grid cell.")
    parser.add_argument("--fps", type=float, default=4.0, help="MP4 frames per second.")
    parser.add_argument(
        "--movie-path",
        default=None,
        help="MP4 output path. Defaults to OUTPUT_DIR/grhd_leveldata_michel_density.mp4.",
    )
    parser.add_argument(
        "--ffmpeg",
        default="/home/mjhatto/GRChombo/tools/ffmpeg-imageio-v4.2.2",
        help="ffmpeg executable to use for optional MP4 encoding.",
    )
    parser.add_argument(
        "--no-movie",
        action="store_true",
        help="Only write PNG frames and the HTML flipbook.",
    )
    parser.add_argument(
        "--require-movie",
        action="store_true",
        help="Fail if ffmpeg is unavailable or MP4 encoding fails.",
    )
    args = parser.parse_args()

    frame_paths = sorted(glob.glob(args.input_glob))
    if not frame_paths:
        raise SystemExit(f"no frames matched {args.input_glob}")
    frames = read_frames(frame_paths)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    positive = []
    for frame in frames:
        for row in frame["rho"]:
            positive.extend(value for value in row if math.isfinite(value) and value > 0.0)
    if not positive:
        raise SystemExit("all density values are non-positive or non-finite")
    log_min = math.log10(min(positive))
    log_max = math.log10(max(positive))

    image_paths = []
    for frame in frames:
        output_path = output_dir / f"density_{frame['step']:04d}.png"
        render_frame(frame, output_path, log_min, log_max, max(1, args.scale))
        image_paths.append(output_path)
        print(output_path)
    print(write_metadata(output_dir, frames, log_min, log_max))

    movie_path = None
    if not args.no_movie:
        requested_movie_path = (
            Path(args.movie_path)
            if args.movie_path is not None
            else output_dir / "grhd_leveldata_michel_density.mp4"
        )
        try:
            movie_path = encode_mp4(
                image_paths, requested_movie_path, args.fps, args.ffmpeg
            )
        except (OSError, subprocess.CalledProcessError) as error:
            if args.require_movie:
                raise SystemExit(f"MP4 encoding failed: {error}")
            print(f"MP4 encoding skipped: {error}")
        if movie_path is None:
            message = f"ffmpeg not found: {args.ffmpeg}"
            if args.require_movie:
                raise SystemExit(message)
            print(f"MP4 encoding skipped: {message}")
        else:
            print(movie_path)

    print(write_html(output_dir, image_paths, movie_path))


if __name__ == "__main__":
    main()
