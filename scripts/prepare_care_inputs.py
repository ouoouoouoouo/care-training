"""Prepare the JSON + pickle inputs that CARE training expects.

CARE's `dataset_pase.py` reads:
    podcast_transcripts -> JSON: {filename_with_ext: transcript_text}
    train_files / valid_files -> pickle: list of filenames matching the JSON keys
    podcast_wavlm_tokens -> text file: one dict per line {"audio": path, "wavlm": "1 2 3 ..."}
                            (LOADED but never used in __getitem__ — safe to provide a placeholder)

This script:
  1. Converts your Whisper transcripts CSV → whisper_transcripts.json
     (keys = "{utt_id}.wav" to match dataset_pase.py's `name.split(os.sep)[-1]`)
  2. Builds train/val pickle (80/20 random split with fixed seed)
  3. Optionally writes empty wavlm_tokens.txt placeholders so init doesn't crash
     (only do this if you can also patch CARE/pretraining/dataset_pase.py to skip
      the actual `self.wavlm_tokens[name]` lookup — we'll do that in Phase 6)

Usage:
    python prepare_care_inputs.py \
        --transcripts-csv /home/ouo/merits-l-text/data/manifests/msp_podcast/transcripts.csv \
        --out-dir /home/ouo/care_training/data \
        --val-fraction 0.20 \
        --seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcripts-csv", required=True, type=str,
                        help="Whisper transcripts CSV with at least (utt_id, text) columns")
    parser.add_argument("--out-dir", required=True, type=str,
                        help="Output dir (e.g. /home/ouo/care_training/data); files written:\n"
                             "  whisper_transcripts.json, trainlist.pkl, vallist.pkl, wavlm_tokens.txt")
    parser.add_argument("--val-fraction", default=0.20, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--include-wavlm-tokens-stub", action="store_true",
                        help="Also write empty wavlm_tokens.txt placeholder.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.transcripts_csv)

    # 1) Load transcripts CSV → dict {filename.wav: text}
    transcripts: Dict[str, str] = {}
    skipped = 0
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            uid = (r.get("utt_id") or "").strip()
            txt = (r.get("text") or "").strip()
            if not uid:
                continue
            if not txt:
                skipped += 1
                continue
            transcripts[f"{uid}.wav"] = txt
    print(f"Loaded {len(transcripts)} transcripts (skipped {skipped} empty).")

    # 2) Write whisper_transcripts.json
    json_path = out_dir / "whisper_transcripts.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(transcripts, f, ensure_ascii=False)
    json_mb = json_path.stat().st_size / (1024 * 1024)
    print(f"✅ Wrote {json_path}  ({json_mb:.1f} MB, {len(transcripts)} entries)")

    # 3) Random 80/20 split → trainlist.pkl, vallist.pkl
    rng = np.random.default_rng(args.seed)
    files = sorted(transcripts.keys())                   # deterministic input order
    idx = np.arange(len(files))
    rng.shuffle(idx)
    n_val = int(round(len(files) * args.val_fraction))
    val_set = set(idx[:n_val].tolist())
    train_list = [files[i] for i in idx if i not in val_set]
    val_list = [files[i] for i in idx if i in val_set]

    train_pkl = out_dir / "trainlist.pkl"
    val_pkl = out_dir / "vallist.pkl"
    with train_pkl.open("wb") as f:
        pickle.dump(train_list, f)
    with val_pkl.open("wb") as f:
        pickle.dump(val_list, f)
    print(f"✅ Wrote {train_pkl}  ({len(train_list)} entries)")
    print(f"✅ Wrote {val_pkl}    ({len(val_list)} entries)")

    # 4) Optional placeholder for wavlm_tokens.txt
    if args.include_wavlm_tokens_stub:
        tokens_path = out_dir / "wavlm_tokens.txt"
        with tokens_path.open("w", encoding="utf-8") as f:
            # One entry per file with a single token "0" — enough so init can parse,
            # but dataset_pase.py must be patched in Phase 6 to skip actually using these.
            for fname in files:
                f.write('{"audio": "' + fname + '", "wavlm": "0"}\n')
        print(f"✅ Wrote {tokens_path} (placeholder — Phase 6 will patch dataset_pase.py to ignore)")

    print("\nSummary:")
    print(f"  Total utts: {len(files)}")
    print(f"  Train:      {len(train_list)} ({100*len(train_list)/len(files):.1f}%)")
    print(f"  Val:        {len(val_list)} ({100*len(val_list)/len(files):.1f}%)")
    print(f"  Seed:       {args.seed}")


if __name__ == "__main__":
    main()
