import torch
import os
import torch.nn as nn
import torchaudio
import yaml
from src.model import SpeechEnhancementVAE
from src.data import STFTTransform
import argparse


def _floor_to_multiple(x: int, m: int) -> int:
    return x - (x % m)


def _rms(x: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(x.float() ** 2)) + 1e-12)


@torch.no_grad()
def inference(model, stft_transform, noisy_waveform, config):
    """
    model: SpeechEnhancementVAE (nhận complex STFT [B, F, T])
    stft_transform: STFTTransform
    noisy_waveform: tensor [T]
    config: config['test']
    """
    model.eval()
    device = next(model.parameters()).device

    noisy_waveform = noisy_waveform.to(device)  # [T]

    cfg_data = config['data']
    cfg_stft = config['stft']

    # Xác định độ dài khung theo cấu hình training
    target_samples = int(cfg_data['sample_rate'] * cfg_data['duration_sec'])
    dummy = torch.zeros(target_samples, device=device)
    dummy_spec = stft_transform(dummy)  # complex [F, T_dummy]
    chunk_len_frames = dummy_spec.shape[-1]
    chunk_len_frames = _floor_to_multiple(chunk_len_frames, 16)  # bội số 16 cho down/up 4 lần
    overlap_frames = chunk_len_frames // 2
    step_frames = chunk_len_frames - overlap_frames

    orig_length = noisy_waveform.shape[0]

    # STFT phức đầu vào
    noisy_spec = stft_transform(noisy_waveform).unsqueeze(0)  # [1, F, T]
    B, F, T = noisy_spec.shape

    # Tần số: cắt về bội số 16 (do kiến trúc stride=2 bốn lần)
    target_F = _floor_to_multiple(F, 16)

    # Tính padding theo chiều thời gian để quét hết file
    pad_long_file = (step_frames - (T - overlap_frames) % step_frames) % step_frames
    pad_short_file = 0
    if T < chunk_len_frames:
        pad_short_file = chunk_len_frames - T
    pad_frames = max(pad_long_file, pad_short_file)

    # Pad theo thời gian
    noisy_spec_padded = torch.nn.functional.pad(noisy_spec, (0, pad_frames))  # [1, F, T']
    T_padded = noisy_spec_padded.shape[-1]

    # Bộ đệm kết quả & trọng số cửa sổ (complex)
    out_spec = torch.zeros_like(noisy_spec_padded, dtype=noisy_spec_padded.dtype)       # [1, F, T']
    window_sum = torch.zeros_like(noisy_spec_padded, dtype=noisy_spec_padded.dtype)     # [1, F, T']

    # Cửa sổ chồng lấn theo thời gian
    fade_window = torch.hann_window(chunk_len_frames, periodic=False).to(device)  # [chunk_len]
    fade_window = fade_window.view(1, 1, -1)  # [1, 1, chunk_len]

    chunk_means = []
    for start_frame in range(0, T_padded - overlap_frames, step_frames):
        end_frame = start_frame + chunk_len_frames

        # Lấy đoạn vào, cắt theo F
        chunk_in = noisy_spec_padded[:, :target_F, start_frame:end_frame]  # [1, target_F, chunk_len]

        # Forward qua VAE (làm việc trên complex)
        chunk_out, _, _ = model(chunk_in)  # [1, target_F, chunk_len], complex

        # Thống kê debug
        chunk_means.append(float(chunk_out.abs().mean().item()))

        # Áp cửa sổ overlap-add (broadcast theo F)
        out_spec[:, :target_F, start_frame:end_frame] += chunk_out * fade_window
        window_sum[:, :target_F, start_frame:end_frame] += fade_window

    # Tránh chia 0
    window_sum_real = window_sum.real
    window_sum_safe = torch.where(window_sum_real == 0, torch.tensor(1.0, device=device), window_sum_real)
    # Chia theo phần thực của trọng số (cửa sổ là thực)
    final_out_spec = out_spec / window_sum_safe
    final_out_spec = final_out_spec[:, :, :T]  # cắt về chiều T gốc, [1, F, T]

    # ISTFT
    final_out_spec_complex = final_out_spec.squeeze(0)  # [F, T]
    enhanced_waveform = torch.istft(
        final_out_spec_complex,
        n_fft=cfg_stft['n_fft'],
        hop_length=cfg_stft['hop_length'],
        win_length=cfg_stft['win_length'],
        window=torch.hann_window(cfg_stft['win_length']).to(device),
        length=orig_length
    )  # [T]

    # Safety gain: tránh mức âm lượng quá nhỏ
    noisy_rms = _rms(noisy_waveform)
    enh_rms = _rms(enhanced_waveform)
    if enh_rms < 1e-5 and noisy_rms > 0:
        gain = min(10.0, noisy_rms / max(enh_rms, 1e-8))
        enhanced_waveform = enhanced_waveform * gain

    # Debug
    try:
        avg_mag = sum(chunk_means) / max(1, len(chunk_means))
        print(f"[inference] chunk_out|mean_abs ~ {avg_mag:.6f}, noisy_rms={noisy_rms:.6f}, enh_rms={_rms(enhanced_waveform):.6f}")
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

    # STFT
    stft_transform = STFTTransform(
        n_fft=config['stft']['n_fft'],
        hop_length=config['stft']['hop_length'],
        win_length=config['stft']['win_length'],
    )

    # Suy ra F, T từ cấu hình chunk (giống train) — chỉ để kiểm soát chunk_len & target_F
    target_samples = int(config['data']['sample_rate'] * config['data']['duration_sec'])
    dummy = torch.zeros(target_samples)
    dummy_spec = stft_transform(dummy)  # [F, T_dummy], complex
    F_orig, T_dummy = dummy_spec.shape
    target_F = _floor_to_multiple(F_orig, 16)
    target_T = _floor_to_multiple(T_dummy, 16)  # dùng cho chunk_len_frames trong inference

    # Tạo model
    model = SpeechEnhancementVAE(z_dim=128).to(device)

    # Load checkpoint
    if args.checkpoint:
        raw_state = torch.load(args.checkpoint, map_location=device)
        # Attempt automatic key remapping for legacy checkpoints
        remapped = {}
        legacy_map = {
            'vae.mu.weight': 'fc_mu.weight',
            'vae.mu.bias': 'fc_mu.bias',
            'vae.logvar.weight': 'fc_log_var.weight',
            'vae.logvar.bias': 'fc_log_var.bias',
            'vae.fc_dec.weight': 'fc_z.weight',
            'vae.fc_dec.bias': 'fc_z.bias',
        }
        missing_legacy = []
        for k,v in raw_state.items():
            if k in legacy_map:
                remapped[legacy_map[k]] = v
            else:
                remapped[k] = v
        # Load with strict=False to ignore any unmatched old keys
        load_result = model.load_state_dict(remapped, strict=False)
        print(f"Model loaded from {args.checkpoint}. Missing keys: {load_result.missing_keys}. Unexpected keys ignored.")
        # Warn if legacy keys existed but were not all remapped
        legacy_present = [k for k in raw_state if k in legacy_map]
        for lk in legacy_present:
            if legacy_map[lk] not in remapped:
                missing_legacy.append(lk)
        if missing_legacy:
            print(f"[WARN] Unmapped legacy keys: {missing_legacy}")
    else:
        print('[INFO] No checkpoint provided. Using randomly initialized model for inference test.')

    model.eval()

    # Load noisy wav
    noisy_waveform, sr = torchaudio.load(args.input)  # [1, T] or [C, T]
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
    parser = argparse.ArgumentParser(description='Denoise audio using SpeechEnhancementVAE (complex STFT).')
    parser.add_argument('--config', type=str, required=True, help='Path to the config YAML file.')
    parser.add_argument('--checkpoint', type=str, required=False, default=None, help='Path to the model checkpoint file (optional).')
    parser.add_argument('--input', type=str, required=True, help='Path to the input noisy audio file.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the enhanced audio file.')

    args = parser.parse_args()
    main(args)
