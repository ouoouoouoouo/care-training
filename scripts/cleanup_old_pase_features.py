"""Find and delete PASE+ feature .npy files saved in the OLD layout (T, 256).

Background: an early version of `extract_msp_pase_features.py` saved
already-downsampled features as `(T_50hz, 256)`. CARE's `dataset_pase.py`
expects `(256, T_100hz)` because it does its own `.T[::2, :]` downsample.

This script scans the output dir, identifies files whose first dim != 256,
and (optionally) deletes them so the extractor can re-process those utts.

Usage:
    # Dry-run (just count) — safe
    python cleanup_old_pase_features.py --pase-dir /home/ouo/care_training/data/pase_features

    # Actually delete
    python cleanup_old_pase_features.py --pase-dir /home/ouo/care_training/data/pase_features --delete
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pase-dir", required=True, type=str)
    parser.add_argument("--expected-dim", default=256, type=int)
    parser.add_argument("--delete", action="store_true",
                        help="Actually delete files. Without this flag, it's a dry run.")
    args = parser.parse_args()

    pase_dir = Path(args.pase_dir)
    files = sorted(pase_dir.glob("*.npy"))
    if not files:
        print(f"No .npy files in {pase_dir}")
        return
    print(f"Scanning {len(files)} files under {pase_dir} ...")

    new_format = []      # shape (256, T) - correct
    old_format = []      # shape (T, 256) - needs re-extraction
    weird = []           # unknown shape

    for p in tqdm(files, desc="scan"):
        try:
            arr = np.load(p, mmap_mode='r')   # mmap = fast, no full read
            if arr.ndim != 2:
                weird.append((p, arr.shape))
                continue
            if arr.shape[0] == args.expected_dim:
                new_format.append(p)
            elif arr.shape[1] == args.expected_dim:
                old_format.append(p)
            else:
                weird.append((p, arr.shape))
        except Exception as e:  # noqa: BLE001
            weird.append((p, f"read-error: {e}"))

    print(f"\n=== Summary ===")
    print(f"  ✅ NEW format (256, T):   {len(new_format)}")
    print(f"  ❌ OLD format (T, 256):   {len(old_format)}")
    print(f"  ⚠️  weird:                {len(weird)}")

    if weird:
        print("\nWeird files (first 10):")
        for p, info in weird[:10]:
            print(f"  {p.name}  →  {info}")

    if old_format:
        print(f"\nSample OLD files (first 5):")
        for p in old_format[:5]:
            arr = np.load(p, mmap_mode='r')
            print(f"  {p.name}  shape={arr.shape}")

    if args.delete and old_format:
        confirm = input(f"\nDelete {len(old_format)} OLD-format files? [y/N]: ").strip().lower()
        if confirm == "y":
            for p in tqdm(old_format, desc="delete"):
                p.unlink()
            print(f"Deleted {len(old_format)} files. Re-run extract_msp_pase_features.py to re-extract.")
        else:
            print("Aborted, no files deleted.")
    elif old_format:
        print(f"\n→ Re-run with --delete to remove {len(old_format)} OLD-format files.")
    else:
        print("\nAll good — no OLD-format files to clean.")


if __name__ == "__main__":
    main()
