import argparse
import os
from typing import Optional

import torch
import soundfile as sf

try:
    import librosa
except Exception:
    librosa = None


def load_audio(path: str, target_sr: Optional[int] = None) -> tuple[torch.Tensor, int]:
    """
    Load an audio file as mono float32 tensor in range [-1, 1].
    Optionally resample to target_sr if provided and different.
    Returns (waveform[T], sample_rate).
    """
    wav, sr = sf.read(path, always_2d=False)
    # Ensure float32
    if wav.dtype != 'float32':
        wav = wav.astype('float32')
    # To mono if multi-channel
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    # Resample if needed
    if target_sr is not None and sr != target_sr:
        if librosa is None:
            raise RuntimeError(
                f"Resampling required from {sr} to {target_sr}, but librosa is not available. "
                "Install librosa or provide audio at target sample rate."
            )
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    # Convert to torch tensor [T]
    wav_t = torch.from_numpy(wav)
    return wav_t, sr


def save_audio(path: str, wav_t: torch.Tensor, sr: int):
    wav = wav_t.detach().cpu().numpy()
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    sf.write(path, wav, sr)


def process_file(ts_model: torch.jit.ScriptModule,
                 input_path: str,
                 output_path: str,
                 solver_steps: int,
                 device: torch.device,
                 model_sr: Optional[int] = None):
    # Load input audio
    wav_t, in_sr = load_audio(input_path, target_sr=model_sr)
    wav_t = wav_t.to(device)

    # Prepare steps tensor on the same device
    steps_t = torch.tensor(int(solver_steps), device=device)

    with torch.no_grad():
        enhanced = ts_model(wav_t, steps_t)

    # Save using original input sample rate if model_sr is None, else model_sr
    out_sr = model_sr if model_sr is not None else in_sr
    save_audio(output_path, enhanced, out_sr)


def main():
    parser = argparse.ArgumentParser(description="Run exported TorchScript denoiser on wavs.")
    parser.add_argument('--model', type=str, required=True,
                        help='Path to TorchScript .pt file (e.g., runs/torchscript/flowmse_ts.pt)')
    parser.add_argument('--input', type=str, required=True,
                        help='Input wav file or directory of wav files')
    parser.add_argument('--output', type=str, required=True,
                        help='Output wav file or directory to store enhanced files')
    parser.add_argument('--steps', type=int, default=20,
                        help='Number of solver steps to run (default: 20)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Device to run on: 'cuda' or 'cpu'")
    parser.add_argument('--model_sr', type=int, default=16000,
                        help='Sample rate the model expects (export-time SR). Will resample inputs if needed (default: 16000).')

    args = parser.parse_args()

    device = torch.device(args.device)
    map_location = device if device.type == 'cpu' else None
    ts_model = torch.jit.load(args.model, map_location=map_location)
    # Ensure the module lives on the target device
    try:
        ts_model.to(device)
    except Exception:
        pass
    ts_model.eval()

    # Decide whether input/output are files or folders
    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        for name in os.listdir(args.input):
            if not name.lower().endswith('.wav'):
                continue
            in_path = os.path.join(args.input, name)
            out_path = os.path.join(args.output, name)
            print(f"Processing: {in_path} -> {out_path}")
            process_file(ts_model, in_path, out_path, args.steps, device, args.model_sr)
    else:
        # Single file
        out_path = args.output
        if os.path.isdir(out_path):
            os.makedirs(out_path, exist_ok=True)
            base = os.path.basename(args.input)
            out_path = os.path.join(out_path, base)
        print(f"Processing: {args.input} -> {out_path}")
        process_file(ts_model, args.input, out_path, args.steps, device, args.model_sr)


if __name__ == '__main__':
    main()
