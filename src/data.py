import torch
import os
from torch.utils.data import Dataset, DataLoader
import torchaudio


class STFTTransform:
    def __init__(self, n_fft=512, hop_length=128, win_length=512):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.hann_window(win_length)

    def __call__(self, waveform):
        
        window = self.window.to(waveform.device)

        stft_result = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            center=True,
            pad_mode='reflect'
        )

        return stft_result
        

class SEDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, transform, sample_rate=16000, target_length=None):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.transform = transform
        self.sample_rate = sample_rate
        self.target_length = target_length

        self.file_list = [f for f in os.listdir(self.clean_dir) if f.endswith('.wav')]

        if not self.file_list:
            raise ValueError("No .wav files found in the specified clean directory.")
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_name = self.file_list[idx]

        clean_path = os.path.join(self.clean_dir, file_name)
        noisy_path = os.path.join(self.noisy_dir, file_name)

        try:
            clean_waveform, sr_clean = torchaudio.load(clean_path)
            noisy_waveform, sr_noisy = torchaudio.load(noisy_path)
        except Exception as e:
            raise RuntimeError(f"Error loading audio files: {e}")
        

        current_len = clean_waveform.shape[1]

        if current_len > self.target_length:
            start_idx = torch.randint(0, current_len - self.target_length + 1, (1,)).item()
            clean_waveform = clean_waveform[:, start_idx:start_idx + self.target_length]
            noisy_waveform = noisy_waveform[:, start_idx:start_idx + self.target_length]
        elif current_len < self.target_length:
            pad_len = self.target_length - current_len
            clean_waveform = torch.nn.functional.pad(clean_waveform, (0, pad_len))
            noisy_waveform = torch.nn.functional.pad(noisy_waveform, (0, pad_len))

        

        clean_waveform = clean_waveform.squeeze(0)
        noisy_waveform = noisy_waveform.squeeze(0)

        target_spec = self.transform(clean_waveform)
        source_spec = self.transform(noisy_waveform)

        return source_spec, target_spec # (freq_bins, time_frames)
    

def get_dataloader(config, mode='train'):
    if mode == 'train':
        cfg = config['train']
        data_cfg = cfg['data']
        stft_cfg = cfg['stft']
    else:
        cfg = config['test']
        data_cfg = cfg['data']
        stft_cfg = cfg['stft']


    try:
        duration_sec = data_cfg['duration_sec']
        sample_rate = data_cfg.get('sample_rate', 16000)
        target_length = int(duration_sec * sample_rate)
    except KeyError as e:
        raise KeyError(f"Missing required data configuration key: {e}")

    transform = STFTTransform(
        n_fft=stft_cfg['n_fft'],
        hop_length=stft_cfg['hop_length'],
        win_length=stft_cfg['win_length']
    )

    ds = SEDataset(
        clean_dir=data_cfg['clean_dir'],
        noisy_dir=data_cfg['noisy_dir'],
        transform=transform,
        sample_rate=data_cfg.get('sample_rate', 16000),
        target_length=target_length
    )

    dataloader = DataLoader(
        dataset=ds,
        batch_size=data_cfg['batch_size'],
        shuffle=True if mode == 'train' else False,
        num_workers=data_cfg.get('num_workers', 4),
        pin_memory=True
    )

    print(f"{mode.capitalize()} DataLoader created with {len(ds)} samples.")
    return dataloader
