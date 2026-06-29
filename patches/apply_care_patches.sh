#!/usr/bin/env bash
# Apply all patches to CARE/pretraining/ for our reproduction setup.
#
# What it does:
#   1. Copy our care_config.py over CARE's config.py (using our paths)
#   2. Patch dataset_pase.py:
#        - replace pickle5 with stdlib pickle
#        - remove the `len(wavlm_tokens[name]) < 100` filter (our placeholder is short)
#        - make roberta_logits load tolerate missing files
#   3. Patch train_pase.py:
#        - enable cuDNN + benchmark (released code disabled them; slow)
#        - disable autograd anomaly detection (debug-only, ~2× slowdown)
#        - add WandB logging (optional, opt-in via WANDB_API_KEY env)
#        - rename "model-{step}.pth" save to use state_dict (smaller, safer)
#
# Idempotent: safe to re-run.
#
# Usage:
#   bash apply_care_patches.sh /home/ouo/care_training/CARE

set -euo pipefail

CARE_ROOT="${1:-/home/ouo/care_training/CARE}"
PT_DIR="${CARE_ROOT}/pretraining"
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[[ -d "$PT_DIR" ]] || { echo "❌ Not found: $PT_DIR"; exit 1; }

echo "Patching CARE at: $CARE_ROOT"
echo ""

# ----------------------------------------------------------------------------
# Step 1: replace config.py
# ----------------------------------------------------------------------------
echo "Step 1: replacing config.py ..."
[[ -f "${PT_DIR}/config.py.bak" ]] || cp "${PT_DIR}/config.py" "${PT_DIR}/config.py.bak"
cp "${THIS_DIR}/care_config.py" "${PT_DIR}/config.py"
echo "  ✅ config.py replaced (original at config.py.bak)"
echo ""

# ----------------------------------------------------------------------------
# Step 2: patch dataset_pase.py
# ----------------------------------------------------------------------------
echo "Step 2: patching dataset_pase.py ..."
export DATASET="${PT_DIR}/dataset_pase.py"
[[ -f "${DATASET}.bak" ]] || cp "$DATASET" "${DATASET}.bak"

python << 'PY'
import os
path = os.environ['DATASET']
with open(path) as f:
    src = f.read()

# 2a) pickle5 -> pickle (with fallback for old envs)
old1 = "import pickle5 as pickle"
new1 = "try:\n    import pickle5 as pickle\nexcept ImportError:\n    import pickle"
if old1 in src and new1 not in src:
    src = src.replace(old1, new1)
    print("  ✅ 2a: pickle5 fallback added")
elif new1 in src:
    print("  ℹ️ 2a: already applied")
else:
    print("  ⚠️ 2a: pickle5 import not found")

# 2b) Remove the wavlm_tokens length filter (our placeholder is 1 token long)
old2 = """        names = list(self.wavlm_tokens.keys())
        for name in names:
            if name not in self.transcripts:
                continue
            if len(self.wavlm_tokens[name]) < 100:
                del self.transcripts[name]
                del self.wavlm_tokens[name]
                del self.wavlm_tokens_6[name]"""
new2 = """        # PATCHED: skip length filter (placeholder wavlm_tokens are not used downstream)
        names = list(self.wavlm_tokens.keys())
        # for name in names:
        #     if name not in self.transcripts:
        #         continue
        #     if len(self.wavlm_tokens[name]) < 100:
        #         del self.transcripts[name]
        #         del self.wavlm_tokens[name]
        #         del self.wavlm_tokens_6[name]"""
if old2 in src:
    src = src.replace(old2, new2)
    print("  ✅ 2b: wavlm_tokens length filter removed")
elif "PATCHED: skip length filter" in src:
    print("  ℹ️ 2b: already applied")
else:
    print("  ⚠️ 2b: filter block not found")

# 2c) Make roberta_logits load tolerate missing files
old3 = "        roberta_logits = np.load(os.path.join(self.roberta_logits_folder, wav_name.replace(\".wav\", \"_text.npy\")))"
new3 = """        # PATCHED: tolerate missing roberta_logits (we don't generate them; not used in loss)
        _logits_path = os.path.join(self.roberta_logits_folder, wav_name.replace(".wav", "_text.npy"))
        if os.path.exists(_logits_path):
            roberta_logits = np.load(_logits_path)
        else:
            roberta_logits = np.zeros((3,), dtype=np.float32)"""
if old3 in src:
    src = src.replace(old3, new3)
    print("  ✅ 2c: roberta_logits load made optional")
elif "PATCHED: tolerate missing roberta_logits" in src:
    print("  ℹ️ 2c: already applied")
else:
    print("  ⚠️ 2c: roberta_logits load line not found")

with open(path, 'w') as f:
    f.write(src)
PY
echo ""

# ----------------------------------------------------------------------------
# Step 3: patch train_pase.py
# ----------------------------------------------------------------------------
echo "Step 3: patching train_pase.py ..."
export TRAIN="${PT_DIR}/train_pase.py"
[[ -f "${TRAIN}.bak" ]] || cp "$TRAIN" "${TRAIN}.bak"

python << 'PY'
import os
path = os.environ['TRAIN']
with open(path) as f:
    src = f.read()

# 3a) Add WandB import at top (optional - only used if WANDB_API_KEY set)
old_imp = "import argparse\n"
new_imp = """import argparse
try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False
"""
if "_HAS_WANDB" not in src and "import wandb" not in src:
    src = src.replace(old_imp, new_imp)
    print("  ✅ 3a: wandb import added")
else:
    print("  ℹ️ 3a: already applied")

# 3b) Enable cuDNN + benchmark for performance (released code disabled them).
#     Also disable anomaly detection (debug-only, ~2× slowdown).
old_perf = """torch.autograd.set_detect_anomaly(True)
#CUDA devices enabled
device = torch.device(\"cuda:0\" if torch.cuda.is_available() else \"cpu\")
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True"""
new_perf = """# PATCHED: turn off anomaly mode + enable cuDNN benchmark for ~2× speedup
# torch.autograd.set_detect_anomaly(True)
#CUDA devices enabled
device = torch.device(\"cuda:0\" if torch.cuda.is_available() else \"cpu\")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False"""
if old_perf in src:
    src = src.replace(old_perf, new_perf)
    print("  ✅ 3b: cuDNN flags enabled, anomaly disabled")
elif "PATCHED: turn off anomaly mode" in src:
    print("  ℹ️ 3b: already applied")
else:
    print("  ⚠️ 3b: perf-flag block not found (may be safe to ignore)")

# 3c) Init WandB after args parsing (opt-in via WANDB_API_KEY)
old_train_def = "def train(args):\n    os.makedirs(args.checkpoint_dir, exist_ok=True)"
new_train_def = """def train(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # PATCHED: optional WandB init (set WANDB_API_KEY to enable)
    use_wandb = _HAS_WANDB and bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "care-training"),
            name=os.environ.get("WANDB_RUN_NAME", f"care_msp_bs{args.batch_size}"),
            config={
                "batch_size": args.batch_size,
                "lr": LEARNING_RATE,
                "betas": BETAS,
                "steps": STEPS,
                "num_layers": args.num_layers,
                "alpha": args.alpha,
                "common_model": args.common_model,
                "use_conv": args.use_conv,
                "use_pretrained": args.use_pretrained,
                "pool_fn": args.pool_fn,
                "supervised": args.supervised,
                "checkpoint_dir": args.checkpoint_dir,
            },
        )
        print(f"WandB run: {wandb.run.url}")"""
if old_train_def in src:
    src = src.replace(old_train_def, new_train_def)
    print("  ✅ 3c: WandB init added")
elif "use_wandb = _HAS_WANDB" in src:
    print("  ℹ️ 3c: already applied")
else:
    print("  ⚠️ 3c: train() entry point not matched")

# 3d) Add WandB log calls after the log-interval prints
old_log = """            if num_steps % 10000 == 0:
                torch.save(emo_model, os.path.join(args.checkpoint_dir, \"model-\"+str(num_steps)+\".pth\"))
                train_speechtext_loss_n = train_speechtext_loss/num_steps
                train_opensmile_loss_n = train_opensmile_loss/num_steps
                logger.info(\"*\"*40)
                logger.info(f\"Step: {num_steps}\")
                # logger.info(f\"Audio Masked Loss: {train_audio_masked_loss_n}\")
                # logger.info(f\"Audio Unmasked Loss: {train_audio_unmasked_loss_n}\")
                logger.info(f\"Speech Text Distillation Loss: {train_speechtext_loss_n}\")
                logger.info(f\"Speech Opensmile Loss: {train_opensmile_loss_n}\")
                logger.info(\"*\"*40)"""
new_log = """            if num_steps % 10000 == 0:
                torch.save(emo_model.state_dict(), os.path.join(args.checkpoint_dir, \"model-\"+str(num_steps)+\".pth\"))
                train_speechtext_loss_n = train_speechtext_loss/num_steps
                train_opensmile_loss_n = train_opensmile_loss/num_steps
                logger.info(\"*\"*40)
                logger.info(f\"Step: {num_steps}\")
                logger.info(f\"Speech Text Distillation Loss: {train_speechtext_loss_n}\")
                logger.info(f\"Speech Opensmile Loss: {train_opensmile_loss_n}\")
                logger.info(\"*\"*40)
                if use_wandb:
                    wandb.log({\"train/loss_sem\": train_speechtext_loss_n,
                               \"train/loss_acoust\": train_opensmile_loss_n,
                               \"train/loss_total\": train_speechtext_loss_n + train_opensmile_loss_n},
                              step=num_steps)"""
if old_log in src:
    src = src.replace(old_log, new_log)
    print("  ✅ 3d: WandB train log + state_dict save")
elif "use_wandb:" in src and "wandb.log({\"train/loss_sem" in src:
    print("  ℹ️ 3d: already applied")
else:
    print("  ⚠️ 3d: log block not matched (may need manual review)")

# 3e) Add WandB log after validation
old_val = """                    if total_valid_loss < best_valid_loss:
                        torch.save(emo_model, os.path.join(args.checkpoint_dir, \"model-\"+str(num_steps)+\".pth\"))
                        # valid_audio_loss_n = valid_audio_loss/valid_steps
                        valid_speechtext_loss_n = valid_speechtext_loss/valid_steps
                        valid_opensmile_loss_n = valid_opensmile_loss/valid_steps
                        logger.info(\"*\"*40)
                        logger.info(f\"Step: {num_steps}\")
                        # logger.info(f\"Val Audio Loss: {valid_audio_loss_n}\")
                        logger.info(f\"Val Distillation Loss: {valid_speechtext_loss_n}\")
                        logger.info(f\"Val Opensmile Loss: {valid_opensmile_loss_n}\")
                        logger.info(\"*\"*40)
                        best_valid_loss = total_valid_loss"""
new_val = """                    valid_speechtext_loss_n = valid_speechtext_loss/valid_steps
                    valid_opensmile_loss_n = valid_opensmile_loss/valid_steps
                    if use_wandb:
                        wandb.log({\"val/loss_sem\": valid_speechtext_loss_n,
                                   \"val/loss_acoust\": valid_opensmile_loss_n,
                                   \"val/loss_total\": total_valid_loss}, step=num_steps)
                    if total_valid_loss < best_valid_loss:
                        torch.save(emo_model.state_dict(), os.path.join(args.checkpoint_dir, \"best.pth\"))
                        logger.info(\"*\"*40)
                        logger.info(f\"Step: {num_steps}   (new best)\")
                        logger.info(f\"Val Distillation Loss: {valid_speechtext_loss_n}\")
                        logger.info(f\"Val Opensmile Loss: {valid_opensmile_loss_n}\")
                        logger.info(\"*\"*40)
                        best_valid_loss = total_valid_loss"""
if old_val in src:
    src = src.replace(old_val, new_val)
    print("  ✅ 3e: WandB val log + best.pth save")
elif "wandb.log({\"val/loss_sem" in src:
    print("  ℹ️ 3e: already applied")
else:
    print("  ⚠️ 3e: val block not matched (may need manual review)")

with open(path, 'w') as f:
    f.write(src)
PY
echo ""

# ----------------------------------------------------------------------------
echo "==============================================="
echo "All patches applied. To revert any file:"
echo "  mv ${PT_DIR}/<file>.bak ${PT_DIR}/<file>"
echo ""
echo "Next steps:"
echo "  1. python care-training/scripts/prepare_care_text_labels.py ..."
echo "  2. Test: python ${PT_DIR}/dataset_pase.py  (or dry-run via train_pase.py)"
echo "  3. Launch training (see care-training/README.md Phase 6)"
echo "==============================================="
