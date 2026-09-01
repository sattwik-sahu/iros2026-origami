#!/usr/bin/env python3
"""
01_organize_seasons.py

Downloads one or more "seasons" of a season-organized LeRobot dataset (like
SharpaIT/Robotic_Origami_Challenge) from the Hugging Face Hub and lays each
one out on your disk as a *proper, self-contained LeRobot dataset root*
(meta/, data/, videos/) that StreamingLeRobotDataset can read directly.

Why this is needed: some Hub datasets are published as

    <repo_root>/
      season_A_train/
        lerobot3.0/{meta,data,videos}/
        lerobotv2.1/{meta,data,videos}/
      season_B_train/
        lerobot3.0/{meta,data,videos}/
      ...

instead of a flat `meta/ data/ videos/` at the repo root. StreamingLeRobotDataset
expects the latter, so pointing it straight at the repo_id fails with a
"meta/info.json not found" style error. This script downloads just the
season(s) + format you want and leaves each one as its own valid dataset root:

    <out_dir>/<season>/<fmt>/{meta,data,videos}/

No re-encoding, no re-chunking, no copying beyond what the Hub SDK does for
you -- this is the disk- and time-cheap option. Use 02_stream_seasons.py to
read one or many of these season roots (they get virtually chained together,
nothing is merged on disk). If you *do* want a single physically-merged
dataset (e.g. to re-publish as one clean repo), use 03_merge_seasons.py
afterwards -- that's an optional, disk-heavier step.

Usage
-----
# See what seasons/formats exist in the repo
python 01_organize_seasons.py --repo-id SharpaIT/Robotic_Origami_Challenge --list

# Download two seasons in lerobot3.0 format
python 01_organize_seasons.py \\
    --repo-id SharpaIT/Robotic_Origami_Challenge \\
    --seasons season_A_train season_B_train \\
    --format lerobot3.0 \\
    --out-dir /data/lerobot_seasons

# Download *all* seasons (careful -- this can be hundreds of GB)
python 01_organize_seasons.py --repo-id SharpaIT/Robotic_Origami_Challenge --all --out-dir /data/lerobot_seasons

Requires: huggingface_hub (pulled in automatically by `pip install lerobot`).
Make sure you've run `hf auth login` first if the dataset is gated.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def discover_seasons(repo_id: str, token: str | bool | None = None) -> dict[str, set[str]]:
    """Return {season_name: {formats found}} by inspecting the repo's file list.

    Works for the "<season>/<format>/meta|data|videos/..." layout. Falls back
    to treating the whole repo as a single "season" named "" if no such
    nested layout is detected (i.e. it's already a flat dataset).
    """
    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type="dataset", token=token)

    season_formats: dict[str, set[str]] = {}
    pattern = re.compile(r"^([^/]+)/([^/]+)/(meta|data|videos)/")
    for f in files:
        m = pattern.match(f)
        if m:
            season, fmt = m.group(1), m.group(2)
            season_formats.setdefault(season, set()).add(fmt)

    if not season_formats:
        # Flat dataset already -- no season nesting.
        flat_pattern = re.compile(r"^(meta|data|videos)/")
        if any(flat_pattern.match(f) for f in files):
            season_formats[""] = {"<flat>"}

    return season_formats


def download_season(
    repo_id: str,
    season: str,
    fmt: str,
    out_dir: Path,
    token: str | bool | None = None,
    max_workers: int = 8,
) -> Path:
    """Download one season/format into out_dir, return the resulting dataset root."""
    allow_patterns = [f"{season}/{fmt}/**"] if season else [f"{fmt}/**", "meta/**", "data/**", "videos/**"]
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        local_dir=out_dir,
        max_workers=max_workers,
        token=token,
    )
    root = out_dir / season / fmt if season else out_dir
    if not (root / "meta" / "info.json").exists():
        raise FileNotFoundError(
            f"Expected {root}/meta/info.json after download but it's missing. "
            f"Double check --format matches what's actually published for this season."
        )
    return root


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", required=True, help="Source Hub dataset repo, e.g. SharpaIT/Robotic_Origami_Challenge")
    p.add_argument("--out-dir", type=Path, default=Path("./lerobot_seasons"), help="Where to write season roots")
    p.add_argument("--format", default="lerobot3.0", help="Which per-season format folder to pull (default: lerobot3.0)")
    p.add_argument("--seasons", nargs="*", default=None, help="Specific season names to download")
    p.add_argument("--all", action="store_true", help="Download every season found")
    p.add_argument("--list", action="store_true", help="Just print discovered seasons/formats and exit")
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--token", default=None, help="HF token (defaults to your cached `hf auth login` token)")
    args = p.parse_args()

    season_formats = discover_seasons(args.repo_id, token=args.token)
    if not season_formats:
        print(f"Could not find any season-shaped or flat lerobot layout in {args.repo_id}.", file=sys.stderr)
        sys.exit(1)

    print(f"Discovered {len(season_formats)} season(s) in {args.repo_id}:")
    for season, fmts in sorted(season_formats.items()):
        print(f"  - {season or '(repo root)'}: formats = {sorted(fmts)}")

    if args.list:
        return

    if args.all:
        chosen = list(season_formats.keys())
    elif args.seasons:
        unknown = set(args.seasons) - set(season_formats.keys())
        if unknown:
            print(f"Unknown season(s): {sorted(unknown)}", file=sys.stderr)
            sys.exit(1)
        chosen = args.seasons
    else:
        print("\nNothing downloaded -- pass --seasons <names...> or --all (see --list output above).")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    roots = []
    for season in chosen:
        fmt = args.format if args.format in season_formats[season] else next(iter(season_formats[season]))
        if fmt != args.format:
            print(f"  [{season}] '{args.format}' not available, using '{fmt}' instead")
        print(f"Downloading {args.repo_id}::{season}/{fmt} ...")
        root = download_season(args.repo_id, season, fmt, args.out_dir, token=args.token, max_workers=args.max_workers)
        roots.append(root)
        print(f"  -> ready at {root}")

    print("\nDone. Each of these is a valid, independent LeRobot dataset root:")
    for r in roots:
        print(f"  {r}")
    print("\nNext: use 02_stream_seasons.py to stream from one or all of them,")
    print("or 03_merge_seasons.py to physically merge them into a single dataset.")


if __name__ == "__main__":
    main()
