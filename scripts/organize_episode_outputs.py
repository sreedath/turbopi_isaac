"""Mirror per-episode video.mp4 and data.parquet under flat outputs folders.

The recorder writes one folder per episode (video + parquet + info). This helper
adds two flat directories for easy inspection while keeping the original layout
intact (the dataset loader still reads from the per-episode dirs).

Outputs:
    outputs/episodes/videos/<session>__episode_XXXXX.mp4
    outputs/episodes/parquet/<session>__episode_XXXXX.parquet
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-root", default="/workspace/turbopi_standalone/outputs/episodes")
    parser.add_argument("--symlink", action="store_true", help="symlink instead of copy")
    args = parser.parse_args()

    root = Path(args.episodes_root)
    videos_dir = root / "videos"
    parquet_dir = root / "parquet"
    videos_dir.mkdir(parents=True, exist_ok=True)
    parquet_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for episode_dir in sorted(root.glob("*/episode_*")):
        if not episode_dir.is_dir():
            continue
        session_name = episode_dir.parent.name
        ep_name = episode_dir.name
        video_src = episode_dir / "video.mp4"
        parquet_src = episode_dir / "data.parquet"
        video_dst = videos_dir / f"{session_name}__{ep_name}.mp4"
        parquet_dst = parquet_dir / f"{session_name}__{ep_name}.parquet"

        for src, dst in ((video_src, video_dst), (parquet_src, parquet_dst)):
            if not src.exists():
                continue
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if args.symlink:
                dst.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dst)
        count += 1

    print(f"[organize] mirrored {count} episodes -> {videos_dir} and {parquet_dir}")


if __name__ == "__main__":
    main()
