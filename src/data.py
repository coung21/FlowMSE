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
        

class TrainDataset(Dataset):
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
        
        # Resample if needed
        if sr_clean != self.sample_rate:
            clean_waveform = torchaudio.transforms.Resample(orig_freq=sr_clean, new_freq=self.sample_rate)(clean_waveform)
        if sr_noisy != self.sample_rate:
            noisy_waveform = torchaudio.transforms.Resample(orig_freq=sr_noisy, new_freq=self.sample_rate)(noisy_waveform)

        # Convert to mono if multi-channel
        if clean_waveform.shape[0] > 1:
            clean_waveform = torch.mean(clean_waveform, dim=0, keepdim=True)
        if noisy_waveform.shape[0] > 1:
            noisy_waveform = torch.mean(noisy_waveform, dim=0, keepdim=True)

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

        return source_spec, target_spec  # complex tensors with shape (freq_bins, time_frames)
    

class EvalDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, sample_rate=16000):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.sample_rate = sample_rate

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

        if sr_clean != self.sample_rate:
            clean_waveform = torchaudio.transforms.Resample(orig_freq=sr_clean, new_freq=self.sample_rate)(clean_waveform)
        if sr_noisy != self.sample_rate:
            noisy_waveform = torchaudio.transforms.Resample(orig_freq=sr_noisy, new_freq=self.sample_rate)(noisy_waveform)

        if clean_waveform.shape[0] > 1:
            clean_waveform = torch.mean(clean_waveform, dim=0)
        if noisy_waveform.shape[0] > 1:
            noisy_waveform = torch.mean(noisy_waveform, dim=0)

        clean_waveform = clean_waveform.squeeze(0)
        noisy_waveform = noisy_waveform.squeeze(0) 

        return clean_waveform, noisy_waveform # (T,), (T,)


def get_dataloader(config, mode='train'):
    if mode == 'train':
        cfg = config['train']
        data_cfg = cfg['data']
        stft_cfg = cfg['stft']
    else:
        cfg = config['test']
        data_cfg = cfg['data']
        stft_cfg = cfg['stft']


    duration_sec = data_cfg['duration_sec']
    sample_rate = data_cfg.get('sample_rate', 16000)
    target_length = int(duration_sec * sample_rate)

    transform = STFTTransform(
        n_fft=stft_cfg['n_fft'],
        hop_length=stft_cfg['hop_length'],
        win_length=stft_cfg['win_length']
    )

    if mode == 'train':
        ds = TrainDataset(
            clean_dir=data_cfg['clean_dir'],
            noisy_dir=data_cfg['noisy_dir'],
            transform=transform,
            sample_rate=data_cfg.get('sample_rate', 16000),
        target_length=target_length
        )
    else:
        ds = EvalDataset(
            clean_dir=data_cfg['clean_dir'],
            noisy_dir=data_cfg['noisy_dir'],
            sample_rate=data_cfg.get('sample_rate', 16000)
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
