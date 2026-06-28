# CARE 訓練環境 setup 完整教學 (中文)

從零到能跑 PASE+ + CARE 的完整步驟，包含所有踩過的雷。

> 目標：訓出我們自己的 CARE checkpoint，取代現在用 raw WavLM-base 的 audio Stage I。
> 預期 audio Stage I weighted F1 從 0.6396 提升到 ~0.66~0.69 (對齊 paper 0.6693)。

---

## Phase 1 — 環境 setup

### 1.1 建 conda env

```bash
conda create -n care python=3.10 -y
conda activate care
pip install --upgrade pip
```

### 1.2 裝 PyTorch (cu124, torch >= 2.6)

```bash
pip install "torch>=2.6" "torchaudio>=2.6" torchvision \
    --index-url https://download.pytorch.org/whl/cu124

# 驗證
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

> ⚠️ 如果你 GPU 是 RTX PRO 5000 Blackwell (sm_120)，PyTorch 2.6 不支援，
> 會顯示 warning。改用 RTX 4090 或升 PyTorch 2.7+ cu128。

### 1.3 Workspace 結構

```bash
mkdir -p /home/ouo/care_training
cd /home/ouo/care_training

# Clone CARE (官方訓練程式碼)
git clone https://github.com/iiscleap/CARE.git

# Clone PASE+ (acoustic supervision)
git clone https://github.com/santi-pdp/pase.git

# 下載 PASE+ 預訓練 checkpoint (~32 MB)
pip install gdown
cd pase
gdown 1xwlZMGnEt9bGKCVcqDeNrruLFQW5zUEW -O FE_e199.ckpt
ls -la FE_e199.ckpt
cd ..

# Clone 這個 helper repo
git clone https://github.com/<your-handle>/care-training.git
```

### 1.4 裝 helper repo 的 deps

```bash
cd care-training
pip install -r requirements.txt
```

這會裝：transformers / soundfile / librosa / gammatone / torchqrnn / cupy /
opensmile / gdown 等。

### 1.5 Patch torchqrnn (繞過 yanked pynvrtc)

```bash
bash patches/apply_torchqrnn_patches.sh
```

這個 patch 做兩件事：
- 把 `from cupy.cuda import function` 跟 `from pynvrtc.compiler import Program` 包進 try/except
- 強制 `ForgetMult.forward` 用 `CPUForgetMult`（pure PyTorch，能在 GPU 上跑，只是不用 fused kernel）

### 1.6 驗證 PASE+ 可用

```bash
cd /home/ouo/care_training/pase
python << 'EOF'
import sys, torch
sys.path.insert(0, '/home/ouo/care_training/pase')

from pase.models.frontend import wf_builder
fe = wf_builder('cfg/frontend/PASE+.cfg').eval()
fe.load_pretrained('FE_e199.ckpt', load_last=True, verbose=False)

dummy = torch.randn(1, 1, 16000)
with torch.no_grad():
    feats = fe(dummy)
print(f"✅ CPU shape: {feats.shape}")  # expect (1, 256, 100)

fe = fe.cuda()
dummy = dummy.cuda()
with torch.no_grad():
    feats = fe(dummy)
print(f"✅ GPU shape: {feats.shape}")
EOF
```

通過 `(1, 256, 100)` 兩次 → **PASE+ 完全可用，Phase 1 完成**。

---

## 踩過的雷整理

| 問題 | 修法 |
|------|------|
| `pip install -r pase/requirements.txt` 整包炸（古老依賴） | **不要跑**，照 requirements.txt 個別安裝 |
| `ahoproc_tools` 用 deprecated `sklearn` 套件 | `SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True pip install ...` |
| `gammatone` git+repo 安裝慢 | 用 PyPI 版 `pip install gammatone` |
| `No module named 'torchvision'` | 跟 torch 同 cu124 index 裝 `torchvision` |
| PASE+ frontend 用 QRNN, 但 `pip install -r` 沒裝到 | `pip install --no-deps git+https://github.com/salesforce/pytorch-qrnn.git` |
| QRNN 需要 cupy | `pip install cupy-cuda12x` (~133 MB) |
| QRNN 需要 pynvrtc 但 PyPI 被 yank | **Patch torchqrnn** 讓兩個 import 變 optional |
| QRNN.forward 預設 `use_cuda=True`, assert tensor on GPU | **Patch** ForgetMult.forward 強制 `use_cuda = False` (走 CPUForgetMult, 仍能在 GPU 跑) |
| Blackwell (sm_120) 不被 PyTorch 2.6 支援 | 用 4090 / `CUDA_VISIBLE_DEVICES=<4090 idx>` |

---

## Phase 2 — PASE+ feature 抽取（4~8 小時 GPU）

```bash
export CUDA_VISIBLE_DEVICES=0    # 確認是 4090 不是 Blackwell

tmux new -s pase_extract

mkdir -p /home/ouo/care_training/data/pase_features

python /home/ouo/care_training/care-training/scripts/extract_msp_pase_features.py \
    --pase-repo /home/ouo/care_training/pase \
    --pase-ckpt /home/ouo/care_training/pase/FE_e199.ckpt \
    --pase-cfg  /home/ouo/care_training/pase/cfg/frontend/PASE+.cfg \
    --audio-root /home/ouo/dataset/MSP_Podcast/Audios \
    --reference-dir /home/ouo/dataset/MSP_Podcast/Transcripts \
    --out-dir /home/ouo/care_training/data/pase_features \
    --batch-size 8 \
    --device cuda
```

`Ctrl+B D` 離開 tmux。

監看進度：
```bash
watch -n 60 'ls /home/ouo/care_training/data/pase_features | wc -l'
```

預期最後 ~148,950 個 .npy 檔（149,307 - 失敗的 ~200 - 過短/過長 skip 的 ~150）。

---

## Phase 3-7

（之後階段腳本陸續加入）

---

## 故障排除

### 問題：PASE+ forward 跑出 NaN

可能 patch 改太多。回退測試：
```bash
mv /home/ouo/miniconda3/envs/care/lib/python3.10/site-packages/torchqrnn/forget_mult.py.bak \
   /home/ouo/miniconda3/envs/care/lib/python3.10/site-packages/torchqrnn/forget_mult.py
# 然後重新跑 patches/apply_torchqrnn_patches.sh
```

### 問題：抽 PASE+ 跑到一半 OOM

把 `--batch-size 8` 降到 4 或 2。

### 問題：CARE 訓練 step 速度比預期慢很多

可能 dataloader 是 bottleneck。`htop` 看 CPU 是否吃滿，或調 `num_workers`。
