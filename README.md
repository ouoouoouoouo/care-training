# CARE training — MERITS-L audio backbone

Scripts to train **CARE** (Content and Acoustic Representations of Emotions)
from scratch on MSP-PODCAST, as the audio encoder for the
[MERITS-L paper](https://arxiv.org/abs/2407.07198) reproduction in
[merits-l-text](https://github.com/ouoouoouoouo/merits-l-text).

CARE paper: Dutta & Ganapathy, *"Leveraging Content and Acoustic Representations
for Speech Emotion Recognition"* ([arXiv 2409.05566](https://arxiv.org/abs/2409.05566)).

Official CARE repo: [iiscleap/CARE](https://github.com/iiscleap/CARE).
We use it as-is for training (with patches), and provide the feature-extraction
pipeline + integration scripts here.

---

## Why this repo

The official CARE repo (`iiscleap/CARE`):
- Doesn't release pretrained weights → must train from scratch
- Was last touched 2024, dependencies are old
- Uses **OpenSMILE** for acoustic supervision in code, but the **paper** uses **PASE+**

This repo:
- Patches the broken dependencies (`pynvrtc`, `cupy`, `torchqrnn` chain)
- Pre-computes everything CARE training needs (PASE+, WavLM-base, RoBERTa, transcripts)
- Provides scripts to use the trained CARE checkpoint downstream in our existing
  audio Stage I / II / III pipeline

---

## Roadmap (7 phases)

| Phase | Task | Time | Status |
|-------|------|------|--------|
| 1 | Set up `care` conda env + clone CARE + PASE+ + patch torchqrnn | 1 hr | ✅ done |
| 2 | Extract PASE+ features for 149K MSP-PODCAST | 4-8 hr GPU | script ready |
| 3 | (Optional) Extract WavLM-base frame features | 2-4 hr GPU | script ready, skip-able |
| 4 | Extract RoBERTa-base mean-pool features from Whisper transcripts | 30 min | script ready |
| 5 | Convert transcripts CSV → JSON, build train/val pickle splits | 10 min | script ready |
| 6 | Modify CARE config.py + dataset_pase.py, run pretraining (200K-800K steps) | 2-3 days | todo |
| 7 | Wrap trained CARE encoder → use in downstream audio pipeline | 1 hr | todo |

> **Why Phase 3 is optional**: CARE's `dataset_pase.py` loads `wavlm_tokens` but
> never actually uses them in `__getitem__` (dead code path). We can provide
> a placeholder text file via `prepare_care_inputs.py --include-wavlm-tokens-stub`
> and patch the dataset class in Phase 6 to skip the load entirely.

See [docs/SETUP_zh.md](docs/SETUP_zh.md) for step-by-step instructions (中文).

---

## Quickstart

### Prerequisites

- Conda env named `care` with Python 3.10
- CUDA 12.x driver, RTX 4090 (or similar; **NOT** Blackwell — PyTorch 2.6 yet)
- MSP-PODCAST audio at `/home/ouo/dataset/MSP_Podcast/Audios/`
- MSP-PODCAST transcripts at `/home/ouo/dataset/MSP_Podcast/Transcripts/`

### Setup env + repos

```bash
mkdir -p /home/ouo/care_training && cd /home/ouo/care_training

# Clone this repo
git clone https://github.com/<your-handle>/care-training.git

# Clone CARE (official)
git clone https://github.com/iiscleap/CARE.git

# Clone PASE+ and download checkpoint
git clone https://github.com/santi-pdp/pase.git
cd pase
pip install gdown
gdown 1xwlZMGnEt9bGKCVcqDeNrruLFQW5zUEW -O FE_e199.ckpt
cd ..

# Install env
conda activate care
pip install -r care-training/requirements.txt

# Apply patches (cupy/pynvrtc workaround for torchqrnn)
bash care-training/patches/apply_torchqrnn_patches.sh
```

### Phase 2 — Extract PASE+ features

```bash
export CUDA_VISIBLE_DEVICES=0   # pick a non-Blackwell GPU

python care-training/scripts/extract_msp_pase_features.py \
    --pase-repo /home/ouo/care_training/pase \
    --pase-ckpt /home/ouo/care_training/pase/FE_e199.ckpt \
    --pase-cfg  /home/ouo/care_training/pase/cfg/frontend/PASE+.cfg \
    --audio-root /home/ouo/dataset/MSP_Podcast/Audios \
    --reference-dir /home/ouo/dataset/MSP_Podcast/Transcripts \
    --out-dir /home/ouo/care_training/data/pase_features \
    --batch-size 8 \
    --device cuda
```

Output: `~/care_training/data/pase_features/{utt_id}.npy` each `(T_50hz, 256)`.

### Phase 4 — Extract RoBERTa-base mean-pool features (semantic supervision target)

Reuses the Whisper transcripts CSV from your `merits-l-text` reproduction.

```bash
python care-training/scripts/extract_msp_roberta_mean.py \
    --transcripts-csv /home/ouo/merits-l-text/data/manifests/msp_podcast/transcripts.csv \
    --out-dir /home/ouo/care_training/data/roberta_features \
    --model roberta-base \
    --batch-size 64
```

Output: `~/care_training/data/roberta_features/{utt_id}_text.npy` each `(768,)`.

### Phase 5 — Prepare CARE inputs (JSON + train/val pickle + wavlm_tokens stub)

```bash
python care-training/scripts/prepare_care_inputs.py \
    --transcripts-csv /home/ouo/merits-l-text/data/manifests/msp_podcast/transcripts.csv \
    --out-dir /home/ouo/care_training/data \
    --val-fraction 0.20 --seed 42 \
    --include-wavlm-tokens-stub
```

Output:
- `~/care_training/data/whisper_transcripts.json`
- `~/care_training/data/trainlist.pkl` / `vallist.pkl`
- `~/care_training/data/wavlm_tokens.txt` (placeholder; ignored after Phase 6 patches)

### Phase 3 — (Optional) Extract WavLM-base frame features

Only run if you decide to use real WavLM tokens (e.g. cluster them yourself).
Otherwise the Phase 5 stub is enough.

```bash
python care-training/scripts/extract_msp_wavlm_frame.py \
    --audio-root /home/ouo/dataset/MSP_Podcast/Audios \
    --reference-dir /home/ouo/dataset/MSP_Podcast/Transcripts \
    --out-dir /home/ouo/care_training/data/wavlm_features \
    --model microsoft/wavlm-base \
    --batch-size 8
```

### Phase 6 — Apply patches + launch training

#### 6.1 Generate text_labels.json from your MERITS-L pseudo-labels

```bash
python care-training/scripts/prepare_care_text_labels.py \
    --pseudo-labels-csv /home/ouo/merits-l-text/data/manifests/msp_podcast/pseudo_labels.csv \
    --out-json /home/ouo/care_training/data/text_labels.json
```

Maps GPT-3.5 labels (negative/neutral/positive → 0/1/2).

#### 6.2 Apply patches to CARE source

```bash
bash care-training/patches/apply_care_patches.sh /home/ouo/care_training/CARE
```

What it does:
- Replaces `config.py` with paths to your `/home/ouo/care_training/data/`.
- Patches `dataset_pase.py`: `pickle5`→`pickle`, removes `<100` token filter,
  tolerates missing `roberta_logits`.
- Patches `train_pase.py`: enables cuDNN benchmark + disables anomaly detection
  (~2× speedup), adds optional WandB logging, saves `state_dict` not full model,
  saves `best.pth` whenever val loss improves.

All originals backed up to `*.bak` files — revert with `mv config.py.bak config.py` etc.

#### 6.3 Smoke test (50-100 steps, single GPU)

```bash
export CUDA_VISIBLE_DEVICES=0   # RTX 4090
export WANDB_API_KEY=...         # optional
conda activate care

cd /home/ouo/care_training/CARE/pretraining

# Quick sanity (training will checkpoint at 10000 steps; Ctrl+C earlier to stop)
python train_pase.py /home/ouo/care_training/ckpts_smoketest \
    --batch_size 16
```

Watch for:
- ✅ Dataset loads without crash (no KeyError on text_labels / wavlm_tokens)
- ✅ Forward pass works, both losses produce numbers (not NaN)
- ✅ Loss values **decrease** over first few hundred steps
- ✅ GPU memory usage reasonable (4090 has 24GB; batch=16 should fit)

#### 6.4 Full training (200K-800K steps)

```bash
tmux new -s care_train
export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY=$(python -c "import netrc; print(netrc.netrc().hosts['api.wandb.ai'][2])")
conda activate care

cd /home/ouo/care_training/CARE/pretraining
python train_pase.py /home/ouo/care_training/ckpts \
    --batch_size 32 \
    --num_layers 6 \
    --pool_fn avg
```

Detach with `Ctrl+B D`. Expected runtime on single RTX 4090:
- 200K steps (paper-reported): ~1.5-2 days
- 800K steps (release code default): ~6-8 days

Stop with `Ctrl+C` once val loss plateaus. Final model: `ckpts/best.pth`.

### Phase 7 — Integrate trained CARE into audio pipeline (TBD after Phase 6)

---

## Storage estimate

| Item | Size |
|------|------|
| PASE+ features (149K × 250 frames × 256) | ~40 GB |
| WavLM-base frame features (149K × 250 frames × 768 × 13 layers) | ~1.5 TB ⚠️ |
| RoBERTa mean-pool features (149K × 768) | ~460 MB |
| CARE checkpoints | ~2 GB |

⚠️ The WavLM frame features are HUGE. Phase 3 may need to store only what CARE's
`dataset_pase.py` actually reads (likely just conv extractor + 6 common layers,
not all 13). TBD when we get to Phase 3.

---

## Critical decisions (and our choices)

| Decision | Paper says | Code does | We choose |
|----------|------------|-----------|-----------|
| Acoustic target | PASE+ (256-dim) | OpenSMILE eGeMAPS | **PASE+** (faithful to paper) |
| Pre-training steps | 200K | 800K | **200K with early-stop** |
| ASR for transcripts | Whisper-large-v3 | (same) | **Whisper-large-v3** (reuse our merits-l-text transcripts) |
| Audio crop | 5 sec | 5 sec | 5 sec |
| Optimizer | AdamW, lr=1e-5, batch=128 | (same) | (same) |

---

## Known issues / patches

### 1. `torchqrnn` needs `cupy` + `pynvrtc` (yanked from PyPI)

PASE+ frontend uses QRNN as its final layer. QRNN's `forget_mult.py` imports
`cupy.cuda.function` and `pynvrtc.compiler.Program` at module top, both of
which are unavailable on modern setups.

**Fix**: `patches/apply_torchqrnn_patches.sh` makes those imports optional and
forces `ForgetMult.forward` to use the pure-PyTorch `CPUForgetMult` path (which
works on GPU tensors too, just slower than the fused CUDA kernel).

### 2. PASE+ `requirements.txt` is ancient

Pinning `numpy==1.16.4`, `torchaudio==0.4.0`, `cupy-cuda101`, etc. — completely
incompatible with PyTorch 2.6 + Python 3.10.

**Fix**: Don't `pip install -r pase/requirements.txt`. Just install minimum
dependencies (`gammatone`, `torchvision`, `cupy-cuda12x`) and patch torchqrnn.

### 3. Blackwell GPUs (sm_120) not fully supported by PyTorch 2.6

Warning: `NVIDIA RTX PRO 5000 Blackwell ... is not compatible with current
PyTorch`. Code runs but numerical diff between CPU/GPU is ~0.01.

**Fix**: Use a non-Blackwell GPU (RTX 4090 = sm_89) via `CUDA_VISIBLE_DEVICES`,
or upgrade to PyTorch 2.7+ / nightly with cu128.

---

## License

MIT.
