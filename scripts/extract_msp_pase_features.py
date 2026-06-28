"""Extract PASE+ frame features for MSP-PODCAST 149K utterances.

For each .wav, runs PASE+ frontend → (256, T_100hz) → average-pool by 2 along
time → (T_50hz, 256) → save as .npy. Matches CARE paper Sec III-B:
    "These features are down-sampled by a factor of 2, producing target
     descriptors at a frequency of 50 Hz."

Output layout (one file per utterance, ready for CARE training):
    {out_dir}/{utt_id}.npy   shape (T_50hz, 256), float32

Usage:
    # On the cluster, in `care` conda env:
    cd /home/ouo/care_training/pase
    python extract_msp_pase_features.py \
        --pase-repo /home/ouo/care_training/pase \
        --pase-ckpt /home/ouo/care_training/pase/FE_e199.ckpt \
        --pase-cfg  /home/ouo/care_training/pase/cfg/frontend/PASE+.cfg \
        --audio-root /home/ouo/dataset/MSP_Podcast/Audios \
        --reference-dir /home/ouo/dataset/MSP_Podcast/Transcripts \
        --out-dir /home/ouo/care_training/data/pase_features \
        --batch-size 8 \
        --device cuda
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Set, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm

AUDIO_EXTS = (".wav", ".flac")


def _load_reference_ids(reference_dir: Path | None) -> Set[str] | None:
    """Filter to the labelled subset (matches CARE paper's 149K). Reference dir
    can be MSP_Podcast/Transcripts/ (any folder whose stems are the utt_ids)."""
    if reference_dir is None:
        return None
    if not reference_dir.is_dir():
        raise FileNotFoundError(f"--reference-dir not found: {reference_dir}")
    ids = {p.stem for p in reference_dir.iterdir() if p.is_file()}
    if not ids:
        raise RuntimeError(f"No files under {reference_dir}; cannot build id set")
    return ids


def _list_audio_files(audio_root: Path) -> List[Path]:
    return sorted(p for p in audio_root.rglob("*") if p.suffix.lower() in AUDIO_EXTS)


def _read_wav(path: Path, target_sr: int = 16000) -> np.ndarray:
    audio, sr = sf.read(str(path))
    if sr != target_sr:
        raise ValueError(f"{path}: expected {target_sr} Hz, got {sr} Hz")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


@torch.no_grad()
def _extract_batch(
    fe,
    audios: List[np.ndarray],
    device: torch.device,
) -> List[np.ndarray]:
    """PASE+ forward + downsample by 2 → list of (T_50hz, 256) numpy arrays."""
    # Pad to max length in batch
    lens = [a.shape[0] for a in audios]
    max_len = max(lens)
    padded = np.zeros((len(audios), 1, max_len), dtype=np.float32)
    for i, a in enumerate(audios):
        padded[i, 0, : len(a)] = a
    x = torch.from_numpy(padded).to(device)

    feats = fe(x)   # (B, 256, T_100hz)
    # Downsample by 2 (avg pool) along time → (B, 256, T_50hz)
    feats = F.avg_pool1d(feats, kernel_size=2, stride=2)
    feats = feats.transpose(1, 2).contiguous()   # (B, T_50hz, 256)

    out: List[np.ndarray] = []
    for i, raw_len in enumerate(lens):
        # PASE+ output frame rate ≈ raw_len / 160 (16kHz audio @ 100Hz frames)
        # After downsample by 2: raw_len / 320
        valid_T = max(1, raw_len // 320)
        out.append(feats[i, :valid_T].detach().cpu().to(torch.float32).numpy())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pase-repo", required=True, type=str,
                        help="Path to cloned santi-pdp/pase repo")
    parser.add_argument("--pase-ckpt", required=True, type=str,
                        help="Path to FE_e199.ckpt")
    parser.add_argument("--pase-cfg", required=True, type=str,
                        help="Path to cfg/frontend/PASE+.cfg")
    parser.add_argument("--audio-root", required=True, type=str)
    parser.add_argument("--out-dir", required=True, type=str)
    parser.add_argument("--reference-dir", default=None, type=str,
                        help="If set, only extract files whose stem is in this dir "
                             "(use MSP_Podcast/Transcripts/ for the labelled 149K subset)")
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max-seconds", default=30.0, type=float,
                        help="Skip utterances longer than this (avoid OOM on extreme files).")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.pase_repo).resolve()))
    from pase.models.frontend import wf_builder  # type: ignore

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print("Building PASE+ frontend ...")
    fe = wf_builder(args.pase_cfg).eval()
    fe.load_pretrained(args.pase_ckpt, load_last=True, verbose=False)
    fe = fe.to(device)

    audio_root = Path(args.audio_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_ids = _load_reference_ids(Path(args.reference_dir)) if args.reference_dir else None

    all_files = _list_audio_files(audio_root)
    if not all_files:
        raise FileNotFoundError(f"No audio under {audio_root}")

    # Filter by reference set
    if ref_ids is not None:
        all_files = [p for p in all_files if p.stem in ref_ids]
    # Skip already-extracted (resume-friendly)
    already_done = {p.stem for p in out_dir.glob("*.npy")}
    todo = [p for p in all_files if p.stem not in already_done]

    print(f"Total candidates: {len(all_files)} | already done: {len(already_done)} | "
          f"to process: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    # Sort by file size for more uniform batches (less padding waste)
    todo.sort(key=lambda p: p.stat().st_size)

    max_samples = int(args.max_seconds * 16000)
    n_ok, n_fail, n_skip = 0, 0, 0
    t0 = time.time()

    with torch.no_grad():
        for i in tqdm(range(0, len(todo), args.batch_size), desc="extract"):
            batch_paths = todo[i : i + args.batch_size]
            try:
                audios: List[np.ndarray] = []
                kept_paths: List[Path] = []
                for p in batch_paths:
                    a = _read_wav(p)
                    if a.shape[0] > max_samples:
                        n_skip += 1
                        continue
                    if a.shape[0] < 1600:    # < 0.1 sec — likely garbage
                        n_skip += 1
                        continue
                    audios.append(a)
                    kept_paths.append(p)
                if not audios:
                    continue
                feats_list = _extract_batch(fe, audios, device)
                for p, f in zip(kept_paths, feats_list):
                    np.save(out_dir / f"{p.stem}.npy", f)
                    n_ok += 1
            except torch.cuda.OutOfMemoryError:
                print(f"\n[warn] OOM on batch starting {batch_paths[0].name}; "
                      f"halving batch and retrying ...")
                torch.cuda.empty_cache()
                # Fall back: process this batch one-by-one
                for p in batch_paths:
                    try:
                        a = _read_wav(p)
                        if a.shape[0] > max_samples or a.shape[0] < 1600:
                            n_skip += 1
                            continue
                        feats_list = _extract_batch(fe, [a], device)
                        np.save(out_dir / f"{p.stem}.npy", feats_list[0])
                        n_ok += 1
                    except Exception as e:
                        print(f"  [warn] {p.name}: {e}")
                        n_fail += 1
            except Exception as e:
                print(f"\n[warn] batch starting {batch_paths[0].name}: {e}")
                n_fail += len(batch_paths)

    elapsed = time.time() - t0
    print(f"\nDONE in {elapsed/60:.1f} min")
    print(f"  ok={n_ok} | failed={n_fail} | skipped (too long/short)={n_skip}")
    print(f"  output dir: {out_dir.resolve()}")
    if n_ok:
        # Spot check
        sample = next(out_dir.glob("*.npy"))
        arr = np.load(sample)
        print(f"  sample {sample.name}: shape={arr.shape}, dtype={arr.dtype}, "
              f"range [{arr.min():.2f}, {arr.max():.2f}]")


if __name__ == "__main__":
    main()
