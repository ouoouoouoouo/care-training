"""Extract RoBERTa-base mean-pool features for Whisper transcripts of MSP-PODCAST.

CARE paper Sec III-B (semantic supervision):
    "we extract contextual word-level embeddings from the transcripts using a
     pre-trained RoBERTa model and **mean-pool** these embeddings to obtain a
     single feature vector representing the entire transcript. These
     utterance-level embeddings serve as the supervisory signal (y_text)..."

Input  : a CSV with at minimum (utt_id, text) — the Whisper transcripts you
         already produced in your merits-l-text pipeline.
Output : per-utterance .npy files at:
             {out_dir}/{utt_id}_text.npy   shape (768,), float32
         (CARE's dataset_pase.py loads via:
             np.load(f"{folder}/{wav_name.replace('.wav', '_text.npy')}")
          so the `_text.npy` suffix is required.)

Usage:
    python extract_msp_roberta_mean.py \
        --transcripts-csv /home/ouo/merits-l-text/data/manifests/msp_podcast/transcripts.csv \
        --out-dir /home/ouo/care_training/data/roberta_features \
        --model roberta-base \
        --batch-size 64
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def _load_rows(csv_path: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            uid = (r.get("utt_id") or "").strip()
            txt = (r.get("text") or "").strip()
            if uid and txt:
                rows.append((uid, txt))
    return rows


@torch.no_grad()
def _extract_batch(
    model,
    tokenizer,
    texts: List[str],
    device: torch.device,
    max_length: int = 128,
) -> torch.Tensor:
    """Returns (B, 768) mean-pooled (with attention mask) RoBERTa features."""
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True,
    )
    enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
    out = model(**enc)
    hidden = out.last_hidden_state                        # (B, T, H)
    mask = enc["attention_mask"].unsqueeze(-1).float()    # (B, T, 1)
    summed = (hidden * mask).sum(dim=1)                   # (B, H)
    denom = mask.sum(dim=1).clamp(min=1e-6)               # (B, 1)
    pooled = summed / denom                               # (B, H)
    return pooled.detach().cpu().to(torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcripts-csv", required=True, type=str)
    parser.add_argument("--out-dir", required=True, type=str)
    parser.add_argument("--model", default="roberta-base", type=str,
                        help="HuggingFace name; CARE paper uses roberta-base for the semantic target.")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--max-length", default=128, type=int)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    csv_path = Path(args.transcripts_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(csv_path)
    print(f"Loaded {len(rows)} (utt_id, text) rows from {csv_path}")

    # Resume-friendly: skip files already extracted
    done = {p.stem.replace("_text", "") for p in out_dir.glob("*_text.npy")}
    todo = [(u, t) for u, t in rows if u not in done]
    print(f"Already done: {len(done)} | To process: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    n_ok, n_fail = 0, 0
    t0 = time.time()
    with torch.no_grad():
        for i in tqdm(range(0, len(todo), args.batch_size), desc="extract"):
            batch = todo[i : i + args.batch_size]
            uids = [b[0] for b in batch]
            texts = [b[1] for b in batch]
            try:
                pooled = _extract_batch(model, tokenizer, texts, device, args.max_length)
                for uid, vec in zip(uids, pooled):
                    np.save(out_dir / f"{uid}_text.npy", vec.numpy())
                    n_ok += 1
            except Exception as e:  # noqa: BLE001
                print(f"\n[warn] batch {uids[0]}: {e}")
                n_fail += len(batch)

    elapsed = time.time() - t0
    print(f"\nDONE in {elapsed/60:.1f} min")
    print(f"  ok={n_ok} | failed={n_fail}")
    print(f"  output dir: {out_dir.resolve()}")
    if n_ok:
        sample = next(out_dir.glob("*_text.npy"))
        arr = np.load(sample)
        print(f"  sample {sample.name}: shape={arr.shape}, dtype={arr.dtype}, "
              f"range [{arr.min():.3f}, {arr.max():.3f}]")


if __name__ == "__main__":
    main()
