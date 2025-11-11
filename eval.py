from src.data import get_dataloader
import argparse
import torch
from src.model import ConvVAE
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


# Metrics list:
# - PESQ - wb
# - STOI
# - SDR
# - SNR
# - SI-SNR
# - SI-SDR
# - DNSMOS


pesq = PerceptualEvaluationSpeechQuality(16000, 'wb') # 1d waveform [T]
stoi = ShortTimeObjectiveIntelligibility(16000, False) # 1d waveform [T]
sdr = SignalDistortionRatio() # 1d waveform [T]
snr = SignalNoiseRatio() # 1d waveform [T]
si_snr = ScaleInvariantSignalNoiseRatio() # 1d waveform [T]
si_sdr = ScaleInvariantSignalDistortionRatio() # 1d waveform [T]
dnsmos = DeepNoiseSuppressionMeanOpinionScore(16000, False) # output: [5]: [p808_mos, mos_sig, mos_bak, mos_ovr]

def _floor_to_multiple(x: int, m: int) -> int:
    return x - (x % m)


def evaluate(args):
    with open(args.config, 'r') as f:
        full_config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Initialize Weights & Biases for evaluation tracking
    wb_cfg = (full_config.get('wandb') if isinstance(full_config, dict) else None) or {}
    project = wb_cfg.get('project_name', 'FlowMSE')
    default_name = f"eval-{os.path.splitext(os.path.basename(args.checkpoint))[0]}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_name = args.wandb_run_name or wb_cfg.get('run_name') or default_name
    wandb.init(project=project, name=run_name, config=full_config)


    # Build STFT first to determine model input shapes
    config = full_config['test']
    stft_transform = STFTTransform(
        n_fft=config['stft']['n_fft'],
        hop_length=config['stft']['hop_length'],
        win_length=config['stft']['win_length'],
    )

    # Infer input_f, input_t from configured duration and STFT params
    target_samples = int(config['data']['sample_rate'] * config['data']['duration_sec'])
    dummy = torch.zeros(target_samples)
    dummy_spec = stft_transform(dummy)
    F_orig, T_dummy = dummy_spec.shape
    target_F = _floor_to_multiple(F_orig, 16)
    target_T = _floor_to_multiple(T_dummy, 16)

    # Create ConvVAE and load checkpoint
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


    eval_loader = get_dataloader(full_config, mode='test')



    total_pesq = 0.0
    total_stoi = 0.0
    total_sdr = 0.0
    total_snr = 0.0
    total_si_snr = 0.0
    total_si_sdr = 0.0
    total_dnsmos = 0.0

    pbar = tqdm(eval_loader, desc='Evaluating')

    for step_idx, batch in enumerate(pbar, start=1):

        clean_waveform, noisy_waveform = batch
        clean_waveform = clean_waveform[0].to(device)
        noisy_waveform = noisy_waveform[0].to(device)

        enhanced_waveform = inference(
            model=model,
            stft_transform=stft_transform,
            noisy_waveform=noisy_waveform,
            config=config
        ).to(device)

        min_len = min(clean_waveform.shape[0], enhanced_waveform.shape[0])
        clean_waveform = clean_waveform[:min_len].cpu() #[T]
        enhanced_waveform = enhanced_waveform[:min_len].cpu() # [T]
        # Compute metrics here

        pesq_score = pesq(enhanced_waveform, clean_waveform).item()
        stoi_score = stoi(enhanced_waveform, clean_waveform).item()
        sdr_score = sdr(enhanced_waveform, clean_waveform).item()
        snr_score = snr(enhanced_waveform, clean_waveform).item()
        si_snr_score = si_snr(enhanced_waveform, clean_waveform).item()
        si_sdr_score = si_sdr(enhanced_waveform, clean_waveform).item()
        dnsmos_scores = dnsmos(enhanced_waveform)[3].item()  # Get mos_ovr

        total_pesq += pesq_score
        total_stoi += stoi_score
        total_sdr += sdr_score
        total_snr += snr_score  
        total_si_snr += si_snr_score
        total_si_sdr += si_sdr_score
        total_dnsmos += dnsmos_scores

        # Log per-sample metrics to W&B
        wandb.log({
            'eval/pesq': pesq_score,
            'eval/stoi': stoi_score,
            'eval/sdr': sdr_score,
            'eval/snr': snr_score,
            'eval/si_snr': si_snr_score,
            'eval/si_sdr': si_sdr_score,
            'eval/dnsmos': dnsmos_scores,
            'eval/step': step_idx,
        })

        pbar.set_postfix({'PESQ': total_pesq / (pbar.n + 1),
                          'STOI': total_stoi / (pbar.n + 1),
                          'SDR': total_sdr / (pbar.n + 1),
                          'SNR': total_snr / (pbar.n + 1),
                          'SI-SNR': total_si_snr / (pbar.n + 1),
                          'SI-SDR': total_si_sdr / (pbar.n + 1),
                          'DNSMOS': total_dnsmos / (pbar.n + 1),
                          })
        # break  # REMOVE THIS BREAK AFTER DEBUGGING

    num_samples = len(eval_loader)
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

    # Log final averages and metadata
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