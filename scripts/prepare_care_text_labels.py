"""Generate the `text_labels.json` that CARE's dataset_pase.py expects.

CARE dataset filters files by `if x in self.labels_dict` where labels_dict is:
    {filename.wav: int_label}   with int_label in {0, 1, 2}

We reuse the GPT-3.5 pseudo-labels you already produced for MERITS-L:
    negative → 0
    neutral  → 1
    positive → 2

Input  : pseudo_labels.csv  (utt_id, text, label)  — from merits-l-text Phase 1c
Output : text_labels.json  ({utt_id.wav: 0/1/2})

Usage:
    python prepare_care_text_labels.py \
        --pseudo-labels-csv /home/ouo/merits-l-text/data/manifests/msp_podcast/pseudo_labels.csv \
        --out-json /home/ouo/care_training/data/text_labels.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pseudo-labels-csv", required=True, type=str)
    parser.add_argument("--out-json", required=True, type=str)
    args = parser.parse_args()

    csv_path = Path(args.pseudo_labels_csv)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Did you finish MERITS-L Phase 1c?")

    label_dict = {}
    bad_rows = 0
    counter = Counter()

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            uid = (r.get("utt_id") or "").strip()
            lab = (r.get("label") or "").strip().lower()
            if not uid:
                continue
            if lab not in LABEL_MAP:
                bad_rows += 1
                continue
            label_dict[f"{uid}.wav"] = LABEL_MAP[lab]
            counter[lab] += 1

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(label_dict, f, ensure_ascii=False)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"✅ Wrote {out_path}  ({size_mb:.2f} MB, {len(label_dict)} entries)")
    print(f"   Label distribution: {dict(counter)}")
    if bad_rows:
        print(f"   ⚠️ {bad_rows} rows skipped (label not in negative/neutral/positive)")


if __name__ == "__main__":
    main()
