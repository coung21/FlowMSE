# FlowMSE TorchScript Inference Guide

This guide shows how to use the exported TorchScript model for speech enhancement in your own Python code, and via the provided CLI script.

The exported module wraps the full inference pipeline (STFT, chunking/overlap-add, Euler solver, iSTFT). You only need to provide a mono waveform tensor and the number of solver steps.

## What the TorchScript module expects

Contract
- Input 1: noisy waveform — 1D float32 Torch tensor of shape [T], values in roughly [-1, 1]
- Input 2: solver_steps — scalar Torch tensor (integer), on the same device as the waveform
- Output: enhanced waveform — 1D float32 Torch tensor of shape [T] at the model's sample rate

Notes
- Sample rate: the model was exported for 16 kHz by default. Resample inputs to 16 kHz before calling, or use the provided CLI which can do this for you.
- Device: run on CUDA when available; both inputs must be on the same device as the model.
- Batching: the exported wrapper takes a single waveform (1D). If you need batching, call it in a loop or wrap externally.

## Prerequisite: export a TorchScript model

From the project root, export a TorchScript file once (you likely already did this):

```bash
python inference_wrapper.py \
  --config src/config/config.yaml \
  --checkpoint runs/ckpts/flowmse_20251027_141914.pth \
  --output runs/torchscript/flowmse_ts.pt
```

This produces `runs/torchscript/flowmse_ts.pt`.

## Option A: Use the built-in CLI runner

We ship a convenience runner at `scripts/run_torchscript.py`.

Single file:
```bash
python scripts/run_torchscript.py \
  --model runs/torchscript/flowmse_ts.pt \
  --input path/to/noisy.wav \
  --output path/to/enhanced.wav \
  --steps 20 \
  --device cuda \
  --model_sr 16000
```

Folder of WAVs:
```bash
python scripts/run_torchscript.py \
  --model runs/torchscript/flowmse_ts.pt \
  --input data/dataset/test/noisy \
  --output outputs/enhanced \
  --steps 20 \
  --device cuda \
  --model_sr 16000
```

Flags
- `--model`: path to the exported `.pt` file
- `--input`: WAV file path or a directory of `.wav` files
- `--output`: output WAV path or directory
- `--steps`: number of Euler solver steps (20 is a good starting point)
- `--device`: `cuda` or `cpu`
- `--model_sr`: model’s sample rate; inputs will be resampled if needed (requires `librosa`)

## Option B: Minimal Python code snippet

Embed TorchScript inference in your own code:

```python
import torch
import soundfile as sf
import numpy as np

try:
    import librosa
except Exception:
    librosa = None

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ts_model = torch.jit.load('runs/torchscript/flowmse_ts.pt', map_location=device)
ts_model.eval()

# Load audio as mono float32
wav, sr = sf.read('path/to/noisy.wav')
if wav.ndim == 2:
    wav = wav.mean(axis=1)
wav = wav.astype('float32')

# Resample to 16 kHz if needed
target_sr = 16000
if sr != target_sr:
    if librosa is None:
        raise RuntimeError('librosa required to resample from %d to %d' % (sr, target_sr))
    wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    sr = target_sr

# To tensor on device
noisy_t = torch.from_numpy(wav).to(device)
steps_t = torch.tensor(20, device=device)  # number of solver steps

with torch.no_grad():
    enhanced_t = ts_model(noisy_t, steps_t)

enhanced = enhanced_t.detach().cpu().numpy()
sf.write('path/to/enhanced.wav', enhanced, sr)
```

## Integration tips
- Keep `solver_steps` as a Torch tensor. If you pass an `int`, ensure your wrapper accepts it; the included wrapper expects a tensor for traceability.
- If you run on CPU, load with `map_location='cpu'` and put inputs on CPU.
- For very long files, the wrapper already handles chunking and overlap-add internally.

## Dependencies
- Required at runtime: `torch`, `soundfile`
- Optional for resampling: `librosa`

All are listed in `requirements.txt`.

## Troubleshooting
- "Expected Tensor for argument": pass `steps` as a Torch tensor on the same device as the waveform (`torch.tensor(20, device=device)`).
- CUDA out of memory: try `--device cpu` (slower), or reduce other GPU loads. The wrapper already chunks processing to control memory.
- Mismatched sample rate or noisy output pitch: ensure inputs are resampled to the model SR used at export (default 16 kHz).
- Torch version mismatch: TorchScript files are usually forward-compatible, but loading on much older PyTorch may fail. Try a similar or newer PyTorch version.

## Where things live
- Export script: `inference_wrapper.py`
- TorchScript artifact: `runs/torchscript/flowmse_ts.pt`
- CLI runner: `scripts/run_torchscript.py`

Happy enhancing!
