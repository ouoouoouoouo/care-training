#!/usr/bin/env bash
# Patch torchqrnn (Salesforce QRNN) to work without cupy/pynvrtc.
#
# pynvrtc was YANKED from PyPI (no package available), and cupy is heavy.
# We make both imports optional and force ForgetMult.forward to use
# CPUForgetMult (pure PyTorch) which works on GPU tensors as well — just
# slower than the fused CUDA kernel.
#
# Idempotent: safe to run multiple times.

set -euo pipefail

TORCHQRNN_DIR=$(python -c "import torchqrnn, os; print(os.path.dirname(torchqrnn.__file__))" 2>/dev/null) \
    || { echo "❌ torchqrnn not installed. Run: pip install --no-deps git+https://github.com/salesforce/pytorch-qrnn.git"; exit 1; }

FILE="${TORCHQRNN_DIR}/forget_mult.py"
[[ -f "$FILE" ]] || { echo "❌ Not found: $FILE"; exit 1; }
echo "Patching: $FILE"

# Backup once
[[ -f "${FILE}.bak" ]] || cp "$FILE" "${FILE}.bak"

python << PYEOF
path = "$FILE"
with open(path) as f:
    src = f.read()

# Patch 1: make cupy/pynvrtc imports optional
old_import = "from cupy.cuda import function\nfrom pynvrtc.compiler import Program"
new_import = """try:
    from cupy.cuda import function
    from pynvrtc.compiler import Program
    _HAS_CUDA_KERNELS = True
except ImportError:
    function = None
    Program = None
    _HAS_CUDA_KERNELS = False"""

if old_import in src:
    src = src.replace(old_import, new_import)
    print("  ✅ Patch 1 applied: imports wrapped in try/except")
elif "_HAS_CUDA_KERNELS" in src:
    print("  ℹ️  Patch 1 already applied")
else:
    print("  ⚠️  Patch 1 pattern not found — manual check needed")

# Patch 2: force ForgetMult.forward to use CPUForgetMult (pure PyTorch)
old_forward = """        # Use CUDA by default unless it's available
        use_cuda = use_cuda and torch.cuda.is_available()
        # Ensure the user is aware when ForgetMult is not GPU version as it's far faster
        if use_cuda: assert f.is_cuda and x.is_cuda, 'GPU ForgetMult with fast element-wise CUDA kernel requested but tensors not on GPU'"""

new_forward = """        # PATCHED: pynvrtc/cupy CUDA kernel path unavailable; force pure-PyTorch CPUForgetMult
        # (CPUForgetMult is pure PyTorch and works on GPU tensors as well, just slower)
        use_cuda = False"""

if old_forward in src:
    src = src.replace(old_forward, new_forward)
    print("  ✅ Patch 2 applied: ForgetMult.forward forced to CPUForgetMult")
elif "PATCHED: pynvrtc/cupy" in src:
    print("  ℹ️  Patch 2 already applied")
else:
    print("  ⚠️  Patch 2 pattern not found — manual check needed")

with open(path, 'w') as f:
    f.write(src)
PYEOF

echo ""
echo "Verifying ..."
python -c "from torchqrnn import QRNN; print('  ✅ torchqrnn imports OK')"
echo ""
echo "DONE. To revert: mv ${FILE}.bak ${FILE}"
