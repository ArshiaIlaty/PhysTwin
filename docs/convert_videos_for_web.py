#!/usr/bin/env python3
"""Re-encode mp4v OpenCV outputs to H.264 for browser playback (Chrome/Safari/GitHub Pages)."""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    for name in ("ffmpeg",):
        from shutil import which

        path = which(name)
        if path:
            return path
    raise SystemExit(
        "Need ffmpeg: pip install imageio-ffmpeg  OR  yum install ffmpeg"
    )


def convert_one(src: str, dst: str, ffmpeg: str, overwrite: bool) -> bool:
    if os.path.isfile(dst) and not overwrite:
        print(f"  skip (exists) {dst}")
        return False
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    tmp = dst + ".part.mp4"
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        src,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        tmp,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.replace(tmp, dst)
    print(f"  ok {os.path.basename(dst)}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dir",
        default="scenario_sweeps/forward_force_results/videos",
    )
    parser.add_argument(
        "--out_dir",
        default="docs/assets/scenario_sweeps",
        help="Write browser-ready copies here (same filenames).",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ffmpeg = ffmpeg_exe()
    pattern = os.path.join(args.in_dir, "*.mp4")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No mp4 in {args.in_dir}")
        sys.exit(1)

    print(f"Converting {len(files)} videos → {args.out_dir}/ (H.264)")
    for src in files:
        name = os.path.basename(src)
        dst = os.path.join(args.out_dir, name)
        convert_one(src, dst, ffmpeg, args.overwrite)
    print("Done.")


if __name__ == "__main__":
    main()
