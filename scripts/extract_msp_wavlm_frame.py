"""(Optional / Phase 3) Extract WavLM-base last-layer frame features for MSP-PODCAST.

CARE's `dataset_pase.py` loads `wavlm_tokens` (DISCRETE codes via k-means on
WavLM, like HuBERT) but the codes are NEVER used downstream in `__getitem__`
(the line that retrieves them is essentially dead code). So strictly speaking
you do NOT need continuous WavLM features for CARE training.

This script is here for completeness — e.g. if you want to:
  - Compute your own k-means tokens to mimic CARE's tokens
  - Use WavLM features for some auxiliary objective
  - Generally inspect what WavLM extracts for MSP-PODCAST audios

It mirrors the `(D, T_50hz)` save layout used by PASE+ extractor, so CARE's
`.T[::2, :]` loader would work transparently:
    {out_dir}/{utt_id}.npy   shape (768, T_50hz), float32   (after the loader)

Usage:
    python extract_msp_wavlm_frame.py \
        --audio-root /home/ouo/dataset/MSP_Podcast/Audios \
        --reference-dir /home/ouo/dataset/MSP_Podcast/Transcripts \
        --out-dir /home/ouo/care_training/data/wavlm_features \
        --model microsoft/wavlm-base \
        --batch-size 8
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Set

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel

AUDIO_EXTS = (".wav", ".flac")


def _load_reference_ids(reference_dir: Path | None) -> Set[str] | None:
    if reference_dir is None:
        return None
    if not reference_dir.is_dir():
        raise FileNotFoundError(f"--reference-dir not found: {reference_dir}")
    ids = {p.stem for p in reference_dir.iterdir() if p.is_file()}
    if not ids:
        raise RuntimeError(f"No files under {reference_dir}")
    return ids


def _list_audio_files(audio_root: Path) -> List[Path]:
    return sorted(p for p in audio_root.rglob("*") if p.suffix.lower() in AUDIO_EXTS)


def _read_wav(path: Path, sr: int = 16000) -> np.ndarray:
    audio, file_sr = sf.read(str(path))
    if file_sr != sr:
        raise ValueError(f"{path}: expected {sr} Hz, got {file_sr}")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


@torch.no_grad()
def _extract_batch(model, fe, audios, device) -> List[np.ndarray]:
    """Return list of (768, T_wavlm) numpy arrays — saved layout for CARE."""
    inputs = fe(
        audios, sampling_rate=16000, padding=True,
        return_tensors="pt", return_attention_mask=True,
    )
    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
    out = model(**inputs)
    hidden = out.last_hidden_state                                # (B, T_out, 768)

    input_lens = inputs["attention_mask"].sum(dim=-1)
    out_lens = model._get_feat_extract_output_lengths(input_lens).to(device)

    results: List[np.ndarray] = []
    for i in range(hidden.size(0)):
        T = int(out_lens[i].item())
        feats = hidden[i, :T].transpose(0, 1)                     # (768, T)
        results.append(feats.detach().cpu().to(torch.float32).numpy())
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-root", required=True, type=str)
    parser.add_argument("--out-dir", required=True, type=str)
    parser.add_argument("--reference-dir", default=None, type=str)
    parser.add_argument("--model", default="microsoft/wavlm-base", type=str)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max-seconds", default=30.0, type=float)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print(f"Loading {args.model} ...")
    fe = AutoFeatureExtractor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    audio_root = Path(args.audio_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_ids = _load_reference_ids(Path(args.reference_dir)) if args.reference_dir else None

    all_files = _list_audio_files(audio_root)
    if ref_ids is not None:
        all_files = [p for p in all_files if p.stem in ref_ids]
    already_done = {p.stem for p in out_dir.glob("*.npy")}
    todo = [p for p in all_files if p.stem not in already_done]

    print(f"Total: {len(all_files)} | done: {len(already_done)} | to process: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    todo.sort(key=lambda p: p.stat().st_size)
    max_samples = int(args.max_seconds * 16000)

    n_ok, n_fail, n_skip = 0, 0, 0
    t0 = time.time()
    with torch.no_grad():
        for i in tqdm(range(0, len(todo), args.batch_size), desc="extract"):
            batch_paths = todo[i : i + args.batch_size]
            try:
                audios, kept = [], []
                for p in batch_paths:
                    a = _read_wav(p)
                    if a.shape[0] > max_samples or a.shape[0] < 1600:
                        n_skip += 1
                        continue
                    audios.append(a)
                    kept.append(p)
                if not audios:
                    continue
                feats_list = _extract_batch(model, fe, audios, device)
                for p, feat in zip(kept, feats_list):
                    np.save(out_dir / f"{p.stem}.npy", feat)
                    n_ok += 1
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"\n[warn] OOM at {batch_paths[0].name}; falling back to per-file")
                for p in batch_paths:
                    try:
                        a = _read_wav(p)
                        if a.shape[0] > max_samples or a.shape[0] < 1600:
                            n_skip += 1
                            continue
                        feats_list = _extract_batch(model, fe, [a], device)
                        np.save(out_dir / f"{p.stem}.npy", feats_list[0])
                        n_ok += 1
                    except Exception as e:
                        print(f"  [warn] {p.name}: {e}")
                        n_fail += 1
            except Exception as e:
                print(f"\n[warn] batch {batch_paths[0].name}: {e}")
                n_fail += len(batch_paths)

    elapsed = time.time() - t0
    print(f"\nDONE in {elapsed/60:.1f} min")
    print(f"  ok={n_ok} | failed={n_fail} | skipped={n_skip}")
    if n_ok:
        sample = next(out_dir.glob("*.npy"))
        arr = np.load(sample)
        print(f"  sample {sample.name}: shape={arr.shape}, dtype={arr.dtype}, "
              f"range [{arr.min():.2f}, {arr.max():.2f}]")


if __name__ == "__main__":
    main()
