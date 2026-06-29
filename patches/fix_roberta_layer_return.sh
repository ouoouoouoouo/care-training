#!/usr/bin/env bash
# Patch model_pase.py to handle new transformers RobertaLayer return format.
#
# Old transformers (~<4.40): RobertaLayer.forward returns (hidden_states, attentions) → tuple
# New transformers (>=4.40): may return just hidden_states (tensor) directly
#
# CARE code does `x = layer(x)[0]` which:
#   - tuple case: gets hidden_states (3D)   ✓
#   - tensor case: indexes first batch element (drops batch dim → 2D)  ✗
#
# Fix: check isinstance, fall back to tensor if not tuple/model-output.

set -euo pipefail

CARE_ROOT="${1:-/home/ouo/care_training/CARE}"
export MODEL="${CARE_ROOT}/pretraining/model_pase.py"
[[ -f "$MODEL" ]] || { echo "❌ Not found: $MODEL"; exit 1; }

[[ -f "${MODEL}.bak" ]] || cp "$MODEL" "${MODEL}.bak"

python << 'PY'
import os, re
path = os.environ['MODEL']
with open(path) as f:
    src = f.read()

# Define a helper at the top after imports
helper = '''

# PATCHED: helper to handle both old (tuple) and new (tensor / ModelOutput)
# return formats of HuggingFace transformer layers.
def _unwrap_layer_out(out):
    """Returns the hidden_states tensor from a transformer layer's output."""
    if isinstance(out, tuple):
        return out[0]
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    return out

'''

if "_unwrap_layer_out" not in src:
    # Insert helper after the last import line. Find the last "import" / "from" line.
    lines = src.split('\n')
    last_import = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            last_import = i
    lines.insert(last_import + 1, helper)
    src = '\n'.join(lines)
    print("  ✅ helper _unwrap_layer_out inserted")
else:
    print("  ℹ️ helper already present")

# Replace `layer(x)[0]` patterns with `_unwrap_layer_out(layer(x))`
# Look for these specific patterns (avoid grabbing unrelated [0] indexing).
patterns = [
    ("x = layer(x)[0]", "x = _unwrap_layer_out(layer(x))"),
    ("audio_features = self.feature_projection_audio(audio_features)[0]",
     "audio_features = _unwrap_layer_out(self.feature_projection_audio(audio_features))"),
]

for old, new in patterns:
    if old in src and new not in src:
        src = src.replace(old, new)
        print(f"  ✅ replaced: {old.strip()[:60]}")
    elif new in src:
        print(f"  ℹ️ already replaced: {new.strip()[:60]}")
    else:
        print(f"  ⚠️ pattern not found: {old.strip()[:60]}")

with open(path, 'w') as f:
    f.write(src)
PY

echo ""
echo "Done. Test with:"
echo "  cd ${CARE_ROOT}/pretraining"
echo "  python train_pase.py /home/ouo/care_training/ckpts_smoketest --batch_size 4"
