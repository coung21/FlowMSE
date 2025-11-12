import torch
import os
import torchaudio
import yaml
import argparse

from src.model import GeneratorConv            # ← dùng Generator của GAN
from src.data import STFTTransform


def _floor_to_multiple(x: int, m: int) -> int:
    return x - (x % m)


def _rms(x: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(x.float() ** 2)) + 1e-12)


def _complex_to_ri(x: torch.Tensor) -> torch.Tensor:
    """
    [B, F, T] (complex) -> [B, 2, F, T] (real, imag)
    """
    assert torch.is_complex(x), "Expected complex tensor."
    return torch.stack([x.real, x.imag], dim=1)


def _ri_to_complex(x_ri: torch.Tensor) -> torch.Tensor:
    """
    [B, 2, F, T] (real, imag) -> [B, F, T] (complex)
    """
    assert x_ri.dim() == 4 and x_ri.size(1) == 2, "Expected [B, 2, F, T]."
    return torch.complex(x_ri[:, 0], x_ri[:, 1])


@torch.no_grad()
def inference(model, stft_transform, noisy_waveform, config):
    """
    model: GeneratorConv (nhận [B, 2, F, T] thực; trả [B, 2, F, T])
    stft_transform: STFTTransform
    noisy_waveform: tensor [T]
    config: config['test']
    """
    model.eval()
    device = next(model.parameters()).device

    noisy_waveform = noisy_waveform.to(device)  # [T]

    cfg_data = config['data']
    cfg_stft = config['stft']

    # Suy ra độ dài chunk theo cấu hình train
    target_samples = int(cfg_data['sample_rate'] * cfg_data['duration_sec'])
    dummy = torch.zeros(target_samples, device=device)
    dummy_spec = stft_transform(dummy)  # complex [F, T_dummy]
    chunk_len_frames = _floor_to_multiple(dummy_spec.shape[-1], 16)  # bội 16 vì 4 lần /2
    overlap_frames = chunk_len_frames // 2
    step_frames = chunk_len_frames - overlap_frames

    orig_length = noisy_waveform.shape[0]

    # STFT phức đầu vào
    noisy_spec = stft_transform(noisy_waveform).unsqueeze(0)  # [1, F, T] complex
    _, F, T = noisy_spec.shape

    # Bên tần số: cắt về bội số 16
    target_F = _floor_to_multiple(F, 16)

    # Tính padding thời gian để quét hết file
    pad_long_file = (step_frames - (T - overlap_frames) % step_frames) % step_frames
    pad_short_file = max(0, chunk_len_frames - T)
    pad_frames = max(pad_long_file, pad_short_file)

    noisy_spec_padded = torch.nn.functional.pad(noisy_spec, (0, pad_frames))  # [1, F, T']
    T_padded = noisy_spec_padded.shape[-1]

    # Bộ đệm kết quả & trọng số (complex)
    out_spec = torch.zeros_like(noisy_spec_padded, dtype=noisy_spec_padded.dtype)
    window_sum = torch.zeros_like(noisy_spec_padded, dtype=noisy_spec_padded.dtype)

    # Cửa sổ overlap-add
    fade_window = torch.hann_window(chunk_len_frames, periodic=False).to(device).view(1, 1, -1)

    chunk_means = []
    for start in range(0, T_padded - overlap_frames, step_frames):
        end = start + chunk_len_frames

        # Lấy đoạn vào (complex), cắt F
        chunk_in_c = noisy_spec_padded[:, :target_F, start:end]  # [1, target_F, L]
        # → RI cho Generator
        chunk_in_ri = _complex_to_ri(chunk_in_c)                 # [1, 2, target_F, L]

        # Forward qua GeneratorConv (residual_out=True → trả phổ đã khử nhiễu)
        chunk_out_ri = model(chunk_in_ri)                        # [1, 2, target_F, L]
        chunk_out_c = _ri_to_complex(chunk_out_ri)               # [1, target_F, L] complex

        # Debug thống kê
        chunk_means.append(float(chunk_out_c.abs().mean().item()))

        # Overlap-add theo thời gian
        out_spec[:, :target_F, start:end] += chunk_out_c * fade_window
        window_sum[:, :target_F, start:end] += fade_window

    # Tránh chia 0
    window_sum_real = window_sum.real
    window_sum_safe = torch.where(window_sum_real == 0,
                                  torch.tensor(1.0, device=device),
                                  window_sum_real)
    final_out_spec = out_spec / window_sum_safe
    final_out_spec = final_out_spec[:, :, :T]  # cắt về T gốc

    # ISTFT
    final_c = final_out_spec.squeeze(0)  # [F, T] complex
    enhanced_waveform = torch.istft(
        final_c,
        n_fft=cfg_stft['n_fft'],
        hop_length=cfg_stft['hop_length'],
        win_length=cfg_stft['win_length'],
        window=torch.hann_window(cfg_stft['win_length']).to(device),
        length=orig_length
    )  # [T]

    # Safety gain
    noisy_rms = _rms(noisy_waveform)
    enh_rms = _rms(enhanced_waveform)
    if enh_rms < 1e-5 and noisy_rms > 0:
        gain = min(10.0, noisy_rms / max(enh_rms, 1e-8))
        enhanced_waveform = enhanced_waveform * gain

    # Debug
    try:
        avg_mag = sum(chunk_means) / max(1, len(chunk_means))
        print(f"[inference] G-out|mean_abs ~ {avg_mag:.6f}, noisy_rms={noisy_rms:.6f}, enh_rms={_rms(enhanced_waveform):.6f}")
    except Exception:
        pass

    return enhanced_waveform.cpu()


def main(args):
    with open(args.config, 'r') as f:
        config_all = yaml.safe_load(f)
    config = config_all['test']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    SR = config['data']['sample_rate']

    # STFT (khớp train)
    stft_transform = STFTTransform(
        n_fft=config['stft']['n_fft'],
        hop_length=config['stft']['hop_length'],
        win_length=config['stft']['win_length'],
    )

    # Khởi tạo GeneratorConv (residual_out=True để sinh phổ sạch theo kiểu y = x + r)
    model = GeneratorConv(residual_out=True).to(device)

    # Load checkpoint (strict=False để chấp nhận khác biệt nhỏ giữa phiên bản)
    if args.checkpoint:
        raw_state = torch.load(args.checkpoint, map_location=device)
        load_result = model.load_state_dict(raw_state, strict=False)
        if load_result.missing_keys:
            print(f"[INFO] Missing keys when loading: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"[INFO] Unexpected keys ignored: {load_result.unexpected_keys}")
        print(f"Model loaded from {args.checkpoint}")
    else:
        print("[INFO] No checkpoint provided. Using randomly initialized Generator for inference test.")

    model.eval()

    # Load noisy wav
    noisy_waveform, sr = torchaudio.load(args.input)  # [C, T]
    if sr != SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SR)
        noisy_waveform = resampler(noisy_waveform)

    # Mono
    if noisy_waveform.shape[0] > 1:
        noisy_waveform = torch.mean(noisy_waveform, dim=0)
    else:
        noisy_waveform = noisy_waveform.squeeze(0)  # [T]

    print(f'Noisy waveform loaded from {args.input}, length: {noisy_waveform.shape[0]} samples')

    # Xác định đường lưu output
    input_stem = os.path.splitext(os.path.basename(args.input))[0]
    default_filename = f"{input_stem}_enhanced.wav"

    output_arg = args.output
    output_ext = os.path.splitext(output_arg)[1].lower()
    is_dir_like = (output_arg.endswith(os.sep) or output_ext == "" or os.path.isdir(output_arg))
    dest_filename = default_filename if is_dir_like else os.path.basename(output_arg)
    output_path = (os.path.join(output_arg, dest_filename) if is_dir_like else output_arg)

    # Tạo thư mục
    output_dir = os.path.dirname(output_path)
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
        except PermissionError:
            fallback_dir = os.path.join(os.getcwd(), "outputs")
            os.makedirs(fallback_dir, exist_ok=True)
            print(f"[WARN] No permission to write to '{output_dir}'. Falling back to '{fallback_dir}'.")
            output_path = os.path.join(fallback_dir, dest_filename)

    # Chạy inference
    enhanced_waveform = inference(
        model=model,
        stft_transform=stft_transform,
        noisy_waveform=noisy_waveform,
        config=config
    )

    # Lưu file
    torchaudio.save(output_path, enhanced_waveform.unsqueeze(0), SR)
    print(f'Enhanced waveform saved to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Denoise audio using GAN GeneratorConv (complex STFT).')
    parser.add_argument('--config', type=str, required=True, help='Path to the config YAML file.')
    parser.add_argument('--checkpoint', type=str, required=False, default=None, help='Path to the Generator checkpoint file (optional).')
    parser.add_argument('--input', type=str, required=True, help='Path to the input noisy audio file.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the enhanced audio file.')

    args = parser.parse_args()
    main(args)
