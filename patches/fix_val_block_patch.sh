#!/usr/bin/env bash
# Follow-up to apply_care_patches.sh — fixes the 3e patch (val block in
# train_pase.py) that didn't match in the main script due to whitespace
# differences. Idempotent.

set -euo pipefail

CARE_ROOT="${1:-/home/ouo/care_training/CARE}"
export TRAIN="${CARE_ROOT}/pretraining/train_pase.py"
[[ -f "$TRAIN" ]] || { echo "❌ Not found: $TRAIN"; exit 1; }

python << 'PY'
import os, re
path = os.environ['TRAIN']
with open(path) as f:
    src = f.read()

changed = False

# Fix 1: change `torch.save(emo_model, ...)` in val block to state_dict + "best.pth"
# Anchor on the surrounding context to be sure we patch the val one, not the train one.
old_save = '                    if total_valid_loss < best_valid_loss:\n                        torch.save(emo_model, os.path.join(args.checkpoint_dir, "model-"+str(num_steps)+".pth"))'
new_save = '''                    # PATCHED: log val metrics to WandB if available
                    valid_speechtext_loss_n = valid_speechtext_loss/valid_steps
                    valid_opensmile_loss_n = valid_opensmile_loss/valid_steps
                    if use_wandb:
                        wandb.log({"val/loss_sem": valid_speechtext_loss_n,
                                   "val/loss_acoust": valid_opensmile_loss_n,
                                   "val/loss_total": total_valid_loss}, step=num_steps)
                    if total_valid_loss < best_valid_loss:
                        torch.save(emo_model.state_dict(), os.path.join(args.checkpoint_dir, "best.pth"))'''

if old_save in src:
    src = src.replace(old_save, new_save)
    changed = True
    print("  ✅ 3e.1: val block — WandB log + state_dict save to best.pth")
elif "PATCHED: log val metrics to WandB" in src:
    print("  ℹ️  3e.1: already applied")
else:
    # Try without exact whitespace — search for the pattern using regex
    pat = re.compile(
        r'(\s+)if total_valid_loss < best_valid_loss:\s*\n'
        r'\s+torch\.save\(emo_model, os\.path\.join\(args\.checkpoint_dir, "model-"\+str\(num_steps\)\+"\.pth"\)\)',
        re.MULTILINE
    )
    m = pat.search(src)
    if m:
        indent = m.group(1).rstrip('\n')
        replacement = (
            f'{indent}# PATCHED: log val metrics to WandB if available\n'
            f'{indent}valid_speechtext_loss_n = valid_speechtext_loss/valid_steps\n'
            f'{indent}valid_opensmile_loss_n = valid_opensmile_loss/valid_steps\n'
            f'{indent}if use_wandb:\n'
            f'{indent}    wandb.log({{"val/loss_sem": valid_speechtext_loss_n,\n'
            f'{indent}               "val/loss_acoust": valid_opensmile_loss_n,\n'
            f'{indent}               "val/loss_total": total_valid_loss}}, step=num_steps)\n'
            f'{indent}if total_valid_loss < best_valid_loss:\n'
            f'{indent}    torch.save(emo_model.state_dict(), os.path.join(args.checkpoint_dir, "best.pth"))'
        )
        src = pat.sub(replacement, src, count=1)
        changed = True
        print("  ✅ 3e.1: val block patched (regex match)")
    else:
        print("  ⚠️  3e.1: val block pattern not found — skipping")

# Fix 2: remove the inner duplicate `valid_speechtext_loss_n = ...` lines that
# are now redundant since we compute them BEFORE the if-check.
# Find and remove the redundant lines inside the if block.
dup_block_pat = re.compile(
    r'                    if total_valid_loss < best_valid_loss:\n'
    r'                        torch\.save\(emo_model\.state_dict\(\), os\.path\.join\(args\.checkpoint_dir, "best\.pth"\)\)\n'
    r'                        # valid_audio_loss_n = valid_audio_loss/valid_steps\n'
    r'                        valid_speechtext_loss_n = valid_speechtext_loss/valid_steps\n'
    r'                        valid_opensmile_loss_n = valid_opensmile_loss/valid_steps\n',
    re.MULTILINE
)
m2 = dup_block_pat.search(src)
if m2:
    cleaned = (
        '                    if total_valid_loss < best_valid_loss:\n'
        '                        torch.save(emo_model.state_dict(), os.path.join(args.checkpoint_dir, "best.pth"))\n'
    )
    src = dup_block_pat.sub(cleaned, src, count=1)
    changed = True
    print("  ✅ 3e.2: removed duplicate valid_*_loss_n lines inside if-block")

if changed:
    with open(path, 'w') as f:
        f.write(src)
    print("  Saved.")
else:
    print("  No changes needed.")
PY

echo "Done."
