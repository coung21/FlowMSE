import torch
import os
import torch.nn as nn
import torchaudio
import yaml
from src.model import ConvVAE
from src.data import STFTTransform
import argparse 


def _complex_to_2channels(spec: torch.Tensor) -> torch.Tensor:
    """Convert complex STFT [F, T] or [B, F, T] to 2-channel real/imag [1, 2, F, T]."""
    if spec.dim() == 2:
        F, T = spec.shape
        real, imag = spec.real, spec.imag
        return torch.stack([real, imag], dim=0).unsqueeze(0)  # [1, 2, F, T]
    elif spec.dim() == 3:
        # [B, F, T] -> [B, 2, F, T]
        real, imag = spec.real, spec.imag
        return torch.stack([real, imag], dim=1)
    else:
        raise ValueError(f"Unexpected spec shape {spec.shape}")


def _floor_to_multiple(x: int, m: int) -> int:
    return x - (x % m)

@torch.no_grad()
def inference(model, stft_transform, noisy_waveform, config): 
    model.eval()

    device = next(model.parameters()).device

    noisy_waveform = noisy_waveform.to(device) # [T]

    cfg_data = config['data']
    cfg_stft = config['stft']

    target_samples = int(cfg_data['sample_rate'] * cfg_data['duration_sec'])
    # Derive chunk length in STFT frames the same way as training by running a dummy STFT
    dummy = torch.zeros(target_samples, device=device)
    dummy_spec = stft_transform(dummy)
    chunk_len_frames = dummy_spec.shape[-1]
    # Ensure time dimension divisible by 16 for the VAE
    chunk_len_frames = _floor_to_multiple(chunk_len_frames, 16)
    overlap_frames = chunk_len_frames // 2
    step_frames = chunk_len_frames - overlap_frames

    orig_length = noisy_waveform.shape[0]

    noisy_spec = stft_transform(noisy_waveform) # [F, T]
    # Convert to [1, 2, F, T]
    noisy_spec = _complex_to_2channels(noisy_spec)  # [1, 2, F, T]

    B, C, F, T = noisy_spec.shape

    # Frequency bins expected by model: floor to multiple of 16 (drop highest bin if needed)
    target_F = _floor_to_multiple(F, 16)

    pad_long_file = (step_frames - (T - overlap_frames) % step_frames) % step_frames

    pad_short_file = 0
    if T < chunk_len_frames:
        pad_short_file = chunk_len_frames - T

    pad_frames = max(pad_long_file, pad_short_file)

    noisy_spec_padded = torch.nn.functional.pad(noisy_spec, (0, pad_frames)) # [1, 2, F, T']
    T_padded = noisy_spec_padded.shape[-1]

    out_spec = torch.zeros_like(noisy_spec_padded) # [1, 2, F, T']

    window_sum = torch.zeros_like(noisy_spec_padded) # [1, 2, F, T']

    fade_window = torch.hann_window(chunk_len_frames, periodic=False).to(device)
    fade_window = fade_window.view(1, 1, 1, -1) # Shape: [1, 1, 1, chunk_len_frames]

    for start_frame in range(0, T_padded - overlap_frames, step_frames):
        end_frame = start_frame + chunk_len_frames
        chunk_in = noisy_spec_padded[:, :, :target_F, start_frame:end_frame] # [1, 2, target_F, chunk_len]

        # Forward pass through VAE
        chunk_out, _, _ = model(chunk_in)

        out_spec[:, :, :target_F, start_frame:end_frame] += chunk_out * fade_window
        window_sum[:, :, :target_F, start_frame:end_frame] += fade_window

    window_sum = torch.where(window_sum == 0, 1.0, window_sum)
    final_out_spec = out_spec / window_sum
    final_out_spec = final_out_spec[:, :, :, :T]  # [1, 2, F, T]

    final_out_spec_complex = torch.complex(final_out_spec[:,0,:,:], final_out_spec[:,1,:,:]).squeeze(0) # [F, T]

    enhanced_waveform = torch.istft(
        final_out_spec_complex,
        n_fft=cfg_stft['n_fft'],
        hop_length=cfg_stft['hop_length'],
        win_length=cfg_stft['win_length'],
        window=torch.hann_window(cfg_stft['win_length']).to(device),
        length=orig_length
    )  # [T]

    return enhanced_waveform.cpu()




def main(args):

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)['test']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    SR = config['data']['sample_rate']

    # Build STFT first to determine shapes
    stft_transform = STFTTransform(
        n_fft=config['stft']['n_fft'],
        hop_length=config['stft']['hop_length'],
        win_length=config['stft']['win_length'],
    )

    # Determine model input sizes (F,T) from config
    target_samples = int(config['data']['sample_rate'] * config['data']['duration_sec'])
    dummy = torch.zeros(target_samples)
    dummy_spec = stft_transform(dummy)
    F_orig, T_dummy = dummy_spec.shape
    target_F = _floor_to_multiple(F_orig, 16)
    target_T = _floor_to_multiple(T_dummy, 16)

    model = ConvVAE(
        in_channels=2,
        out_channels=2,
        n_features=64,
        z_dim=128,
        input_f=target_F,
        input_t=target_T,
    ).to(device)

    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    print(f'Model loaded from {args.checkpoint}')

    noisy_waveform, sr = torchaudio.load(args.input)  # [1, T]

    if sr != SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SR)
        noisy_waveform = resampler(noisy_waveform)

    if noisy_waveform.shape[0] > 1:
        noisy_waveform = torch.mean(noisy_waveform, dim=0)
    else:
        noisy_waveform = noisy_waveform.squeeze(0) # [T]

    print(f'Noisy waveform loaded from {args.input}, length: {noisy_waveform.shape[0]} samples')

    # Resolve output path: accept directory or file path. If a directory (or no extension),
    # auto-generate a filename based on the input with .wav extension.
    input_stem = os.path.splitext(os.path.basename(args.input))[0]
    default_filename = f"{input_stem}_enhanced.wav"

    # Determine if user passed a directory-like path or a file path
    output_arg = args.output
    output_ext = os.path.splitext(output_arg)[1].lower()
    is_dir_like = (
        output_arg.endswith(os.sep)
        or output_ext == ""
        or os.path.isdir(output_arg)
    )

    # Determine destination filename to reuse on fallback
    dest_filename = default_filename if is_dir_like else os.path.basename(output_arg)

    output_path = (
        os.path.join(output_arg, dest_filename) if is_dir_like else output_arg
    )

    # Create parent directory if any
    output_dir = os.path.dirname(output_path)
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
        except PermissionError:
            # Fall back to a local ./outputs directory
            fallback_dir = os.path.join(os.getcwd(), "outputs")
            os.makedirs(fallback_dir, exist_ok=True)
            print(f"[WARN] No permission to write to '{output_dir}'. Falling back to '{fallback_dir}'.")
            output_path = os.path.join(fallback_dir, dest_filename)

    enhenced_waveform = inference(
        model=model,
        stft_transform=stft_transform,
        noisy_waveform=noisy_waveform,
        config=config
    )

    torchaudio.save(output_path, enhenced_waveform.unsqueeze(0), SR)

    print(f'Enhanced waveform saved to {output_path}')



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Denoise audio using trained model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the config YAML file.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint file.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input noisy audio file.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the enhanced audio file.')

    args = parser.parse_args()

    main(args)