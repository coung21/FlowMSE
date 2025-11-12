from src.data import get_dataloader
import argparse
import torch
from src.model import SpeechEnhancementVAE
from src.data import STFTTransform
from sample import inference
import yaml
from tqdm import tqdm
import os
from datetime import datetime
import wandb

from torchmetrics.audio import SignalDistortionRatio
from torchmetrics.audio import SignalNoiseRatio
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore


# Metrics (đều nhận waveform 1D [T])
pesq = PerceptualEvaluationSpeechQuality(16000, 'wb')
stoi = ShortTimeObjectiveIntelligibility(16000, False)
sdr = SignalDistortionRatio()
snr = SignalNoiseRatio()
si_snr = ScaleInvariantSignalNoiseRatio()
si_sdr = ScaleInvariantSignalDistortionRatio()
dnsmos = DeepNoiseSuppressionMeanOpinionScore(16000, False)  # trả về [p808_mos, mos_sig, mos_bak, mos_ovr, ...]


def _floor_to_multiple(x: int, m: int) -> int:
    return x - (x % m)


def evaluate(args):
    with open(args.config, 'r') as f:
        full_config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Init W&B
    wb_cfg = (full_config.get('wandb') if isinstance(full_config, dict) else None) or {}
    project = wb_cfg.get('project_name', 'SpeechEnhancementVAE')
    default_name = f"eval-{os.path.splitext(os.path.basename(args.checkpoint))[0]}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_name = args.wandb_run_name or wb_cfg.get('run_name') or default_name
    wandb.init(project=project, name=run_name, config=full_config)

    # Build STFT (dùng cho inference)
    config = full_config['test']
    stft_transform = STFTTransform(
        n_fft=config['stft']['n_fft'],
        hop_length=config['stft']['hop_length'],
        win_length=config['stft']['win_length'],
    )

    # Tạo model SpeechEnhancementVAE và load checkpoint
    z_dim = full_config.get('train', {}).get('z_dim', 128)  # dùng cùng z_dim như train nếu có
    model = SpeechEnhancementVAE(z_dim=z_dim).to(device)

    # Load checkpoint with legacy key remap support
    raw_state = torch.load(args.checkpoint, map_location=device)
    remapped = {}
    legacy_map = {
        'vae.mu.weight': 'fc_mu.weight',
        'vae.mu.bias': 'fc_mu.bias',
        'vae.logvar.weight': 'fc_log_var.weight',
        'vae.logvar.bias': 'fc_log_var.bias',
        'vae.fc_dec.weight': 'fc_z.weight',
        'vae.fc_dec.bias': 'fc_z.bias',
    }
    for k, v in raw_state.items():
        remapped[legacy_map.get(k, k)] = v
    load_result = model.load_state_dict(remapped, strict=False)
    if load_result.missing_keys:
        print(f"[INFO] Missing keys when loading: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"[INFO] Unexpected keys ignored: {load_result.unexpected_keys}")
    model.eval()
    print(f'Model loaded from {args.checkpoint}')

    # DataLoader eval
    eval_loader = get_dataloader(full_config, mode='test')

    # Tổng hợp metric
    total_pesq = 0.0
    total_stoi = 0.0
    total_sdr = 0.0
    total_snr = 0.0
    total_si_snr = 0.0
    total_si_sdr = 0.0
    total_dnsmos = 0.0
    num_samples = 0

    pbar = tqdm(eval_loader, desc='Evaluating')

    for batch_idx, batch in enumerate(pbar, start=1):
        # Batch của EvalDataset: (clean_waveform, noisy_waveform), mỗi cái là [B, T]
        clean_batch, noisy_batch = batch
        B = clean_batch.shape[0]
        for i in range(B):
            clean_waveform = clean_batch[i].to(device)
            noisy_waveform = noisy_batch[i].to(device)

            # Inference (trả về waveform CPU)
            enhanced_waveform = inference(
                model=model,
                stft_transform=stft_transform,
                noisy_waveform=noisy_waveform,
                config=config
            )  # CPU [T]

            # Cắt về cùng độ dài
            min_len = min(clean_waveform.shape[0], enhanced_waveform.shape[0])
            clean_wav = clean_waveform[:min_len].cpu()
            enh_wav = enhanced_waveform[:min_len].cpu()

            # Tính metric từng mẫu
            pesq_score = pesq(enh_wav, clean_wav).item()
            stoi_score = stoi(enh_wav, clean_wav).item()
            sdr_score = sdr(enh_wav, clean_wav).item()
            snr_score = snr(enh_wav, clean_wav).item()
            si_snr_score = si_snr(enh_wav, clean_wav).item()
            si_sdr_score = si_sdr(enh_wav, clean_wav).item()
            # dnsmos: lấy chỉ số MOS tổng quan (mos_ovr)
            try:
                dnsmos_scores = dnsmos(enh_wav)[3].item()
            except Exception:
                # dự phòng: nếu API thay đổi, dùng giá trị đầu ra đầu tiên
                dnsmos_scores = float(dnsmos(enh_wav)[0].item())

            # Cộng dồn
            total_pesq += pesq_score
            total_stoi += stoi_score
            total_sdr += sdr_score
            total_snr += snr_score
            total_si_snr += si_snr_score
            total_si_sdr += si_sdr_score
            total_dnsmos += dnsmos_scores
            num_samples += 1

            # Log per-sample
            wandb.log({
                'eval/pesq': pesq_score,
                'eval/stoi': stoi_score,
                'eval/sdr': sdr_score,
                'eval/snr': snr_score,
                'eval/si_snr': si_snr_score,
                'eval/si_sdr': si_sdr_score,
                'eval/dnsmos': dnsmos_scores,
                'eval/sample_idx_global': num_samples,
            })

        # Cập nhật thanh tiến trình theo trung bình tạm thời
        denom = max(1, num_samples)
        pbar.set_postfix({
            'PESQ': total_pesq / denom,
            'STOI': total_stoi / denom,
            'SDR': total_sdr / denom,
            'SNR': total_snr / denom,
            'SI-SNR': total_si_snr / denom,
            'SI-SDR': total_si_sdr / denom,
            'DNSMOS': total_dnsmos / denom,
        })

    # Trung bình cuối cùng
    avg_pesq = total_pesq / num_samples
    avg_stoi = total_stoi / num_samples
    avg_sdr = total_sdr / num_samples
    avg_snr = total_snr / num_samples
    avg_si_snr = total_si_snr / num_samples
    avg_si_sdr = total_si_sdr / num_samples
    avg_dnsmos = total_dnsmos / num_samples

    print('Final Evaluation Results:')
    print(f'PESQ: {avg_pesq:.4f}')
    print(f'STOI: {avg_stoi:.4f}')
    print(f'SDR: {avg_sdr:.4f}')
    print(f'SNR: {avg_snr:.4f}')
    print(f'SI-SNR: {avg_si_snr:.4f}')
    print(f'SI-SDR: {avg_si_sdr:.4f}')
    print(f'DNSMOS: {avg_dnsmos:.4f}')

    # Log tổng kết
    wandb.log({
        'eval/avg_pesq': avg_pesq,
        'eval/avg_stoi': avg_stoi,
        'eval/avg_sdr': avg_sdr,
        'eval/avg_snr': avg_snr,
        'eval/avg_si_snr': avg_si_snr,
        'eval/avg_si_sdr': avg_si_sdr,
        'eval/avg_dnsmos': avg_dnsmos,
        'eval/num_samples': num_samples,
    })
    wandb.summary['eval/final_pesq'] = avg_pesq
    wandb.summary['eval/final_stoi'] = avg_stoi
    wandb.summary['eval/final_sdr'] = avg_sdr
    wandb.summary['eval/final_snr'] = avg_snr
    wandb.summary['eval/final_si_snr'] = avg_si_snr
    wandb.summary['eval/final_si_sdr'] = avg_si_sdr
    wandb.summary['eval/final_dnsmos'] = avg_dnsmos
    wandb.summary['device'] = str(device)
    wandb.summary['checkpoint'] = args.checkpoint
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint.')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Optional W&B run name for evaluation.')
    args = parser.parse_args()

    evaluate(args)
