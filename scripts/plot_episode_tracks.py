"""Reconstruct & plot recorded square-loop tracks from saved parquet episodes.

Integrates the body-frame command [vx, vy, wz] starting from the known route
start pose (lm = (-half, 0) for both ccw and cw) so we can visually verify the
expert teacher actually drove the square. Outputs one PNG with all episodes
overlaid on the square outline.
"""

from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def integrate(commands: np.ndarray, ts: np.ndarray, start_xy=(-0.45, 0.0), start_yaw=0.0):
    n = len(commands)
    xs = np.empty(n + 1)
    ys = np.empty(n + 1)
    xs[0], ys[0] = start_xy
    yaw = start_yaw
    x, y = start_xy
    for i in range(n):
        dt = ts[i + 1] - ts[i] if i + 1 < n else (ts[-1] - ts[-2] if n > 1 else 0.1)
        vx, vy, wz = commands[i]
        yaw_mid = yaw + 0.5 * wz * dt
        x += (vx * math.cos(yaw_mid) - vy * math.sin(yaw_mid)) * dt
        y += (vx * math.sin(yaw_mid) + vy * math.cos(yaw_mid)) * dt
        yaw = (yaw + wz * dt + math.pi) % (2 * math.pi) - math.pi
        xs[i + 1] = x
        ys[i + 1] = y
    return xs, ys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--parquet_glob",
        nargs="+",
        default=[
            "/workspace/turbopi_standalone/outputs/episodes/session_simple_main/episode_*/data.parquet",
            "/workspace/turbopi_standalone/outputs/episodes/session_simple_val/episode_*/data.parquet",
        ],
    )
    ap.add_argument("--half", type=float, default=0.45)
    ap.add_argument("--out", default="/workspace/turbopi_standalone/outputs/episodes/tracks_overview.png")
    ap.add_argument("--max_episodes", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    files = []
    for pattern in args.parquet_glob:
        files.extend(p for p in glob.glob(pattern) if Path(p).exists())
    files = sorted(set(files))
    if args.max_episodes:
        files = files[: args.max_episodes]
    if not files:
        raise SystemExit(f"No parquet files at {args.parquet_glob}")

    fig, ax = plt.subplots(figsize=(7, 7))
    h = args.half
    sq = np.array([[-h, -h], [h, -h], [h, h], [-h, h], [-h, -h]])
    ax.plot(sq[:, 0], sq[:, 1], "k--", lw=2, label="square track")
    ax.scatter([-h], [0.0], c="green", s=80, zorder=5, label="start (lm)")

    color_map = {"counterclockwise": "tab:blue", "clockwise": "tab:red"}
    seen_dir = set()
    summary = []

    for f in files:
        df = pd.read_parquet(f)
        commands = np.stack(df["command"].to_numpy())
        ts = df["timestamp"].to_numpy()
        direction = str(df["direction"].iloc[0])
        # First segment for ccw is lm->bl (face -y); for cw it is lm->tl (face +y).
        start_yaw = -math.pi / 2 if direction == "counterclockwise" else math.pi / 2
        xs, ys = integrate(commands, ts, start_xy=(-h, 0.0), start_yaw=start_yaw)
        label = direction if direction not in seen_dir else None
        seen_dir.add(direction)
        ax.plot(xs, ys, color=color_map.get(direction, "gray"), alpha=0.5, lw=1.0, label=label)
        summary.append((Path(f).stem, direction, len(df), float(df["lap_progress"].iloc[-1])))

    ax.set_aspect("equal")
    pad = 0.25
    ax.set_xlim(-h - pad, h + pad)
    ax.set_ylim(-h - pad, h + pad)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"Reconstructed expert tracks — {len(files)} episodes")
    plt.tight_layout()
    plt.savefig(args.out, dpi=110)
    print(f"[OK] Wrote {args.out}")
    print()
    print(f"{'episode':<48} {'dir':<18} {'frames':>6} {'lap_prog':>9}")
    for name, d, n, lp in summary:
        print(f"{name:<48} {d:<18} {n:>6} {lp:>9.3f}")


if __name__ == "__main__":
    main()
