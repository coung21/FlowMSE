import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import yaml
import os
import argparse
from src.model import SinusoidalTimeEmbedding, ConvBlock, DownBlock, UpBlock, UNet





class FlowMatcherInferenceWrapper(nn.Module):
    def __init__(self, 
                 model: UNet, 
                 n_fft: int, 
                 hop_length: int, 
                 win_length: int, 
                 chunk_len_frames: int, 
                 overlap_frames: int):
        super().__init__()
        
        # 1. Gán mô hình con
        self.model = model
        
        # 2. Lưu các tham số
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.chunk_len_frames = chunk_len_frames
        self.overlap_frames = overlap_frames
        self.step_frames = chunk_len_frames - overlap_frames

        # 3. Tạo và đăng ký các buffer (tương tự STFTTransform)
        self.register_buffer('stft_window', torch.hann_window(win_length))
        self.register_buffer('fade_window', torch.hann_window(chunk_len_frames, periodic=False).view(1, 1, 1, -1))

    @torch.jit.export
    def _euler_solver(self, x: torch.Tensor, num_steps: int) -> torch.Tensor:
        """
        Thực hiện Euler solver bên trong module.
        """
        t_start: float = 0.0
        t_end: float = 1.0
        
        time_steps = torch.linspace(t_start, t_end, num_steps + 1, device=x.device)
        dt: float = (t_end - t_start) / float(num_steps)
        
        x_t = x.clone()

        for i in range(num_steps):
            t_current = time_steps[i]
            t_tensor = torch.full((x.shape[0],), t_current, device=x.device)
            v_t = self.model(x_t, t_tensor)
            x_t = x_t + v_t * dt

        return x_t

    def forward(self, noisy_waveform: torch.Tensor, solver_steps: torch.Tensor) -> torch.Tensor:
        """
        Toàn bộ logic từ `inference` được chuyển vào đây.
        """
        # Lưu độ dài gốc để istft
        orig_length: int = noisy_waveform.shape[0]
        device = noisy_waveform.device
        # Cho phép trace bằng cách nhận tensor và ép kiểu sang int
        if isinstance(solver_steps, torch.Tensor):
            num_steps: int = int(solver_steps.item())
        else:
            num_steps: int = int(solver_steps)
        
        # --- 1. STFT (Từ STFTTransform) ---
        noisy_spec_complex = torch.stft(
            noisy_waveform, 
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.stft_window.to(device),
            return_complex=True,
            center=True,
            pad_mode='reflect'
        ) # [F, T]
        
        noisy_spec = torch.stack([noisy_spec_complex.real, noisy_spec_complex.imag], dim=0).unsqueeze(0) # [1, 2, F, T]

        # --- 2. Padding (Từ inference) ---
        B, C, n_freq, T = noisy_spec.shape

        pad_long_file: int = (self.step_frames - (T - self.overlap_frames) % self.step_frames) % self.step_frames

        pad_short_file: int = 0
        if T < self.chunk_len_frames:
            pad_short_file = self.chunk_len_frames - T

        pad_frames: int = max(pad_long_file, pad_short_file)

        noisy_spec_padded = F.pad(noisy_spec, (0, pad_frames))
        T_padded: int = noisy_spec_padded.shape[-1]

        # --- 3. Chunking & Solving Loop (Từ inference) ---
        out_spec = torch.zeros_like(noisy_spec_padded)
        window_sum = torch.zeros_like(noisy_spec_padded)
        
        fade_window_device = self.fade_window.to(device)

        for start_frame in range(0, T_padded - self.overlap_frames, self.step_frames):
            end_frame = start_frame + self.chunk_len_frames
            chunk_in = noisy_spec_padded[:, :, :, start_frame:end_frame]
            
            # Gọi solver
            chunk_out = self._euler_solver(chunk_in, num_steps)

            out_spec[:, :, :, start_frame:end_frame] += chunk_out * fade_window_device
            window_sum[:, :, :, start_frame:end_frame] += fade_window_device

        # --- 4. Normalization & Un-padding (Từ inference) ---
        window_sum = torch.where(window_sum == 0.0, torch.ones_like(window_sum), window_sum)
        final_out_spec = out_spec / window_sum
        final_out_spec = final_out_spec[:, :, :, :T]  # [1, 2, F, T]

        # --- 5. iSTFT (Từ inference) ---
        final_out_spec_complex = torch.complex(final_out_spec[:,0,:,:], final_out_spec[:,1,:,:]).squeeze(0) # [F, T]

        enhanced_waveform = torch.istft(
            final_out_spec_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.stft_window.to(device),
            length=orig_length
        )  # [T]

        return enhanced_waveform
    


def export_main(args):
    
    with open(args.config, 'r') as f:
        # Load toàn bộ config, không chỉ 'test'
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device} for export')
    config = config['test']
    print(config)
    # 1. Tải Config
    cfg_data = config['data']
    cfg_stft = config['stft']

    # 2. Khởi tạo mô hình UNet
    model = UNet(
        in_channels=2,
        out_channels=2
    ).to(device)

    # 3. Load trọng số
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f'Model loaded from {args.checkpoint}')

    # 4. Tính toán các tham số cho Pipeline
    target_samples = int(cfg_data['sample_rate'] * cfg_data['duration_sec'])
    hop_length = cfg_stft['hop_length']
    chunk_len_frames = (target_samples // hop_length) + 1
    overlap_frames = chunk_len_frames // 2

    # 5. Khởi tạo Pipeline "cha"
    pipeline = FlowMatcherInferenceWrapper(
        model=model,
        n_fft=cfg_stft['n_fft'],
        hop_length=cfg_stft['hop_length'],
        win_length=cfg_stft['win_length'],
        chunk_len_frames=chunk_len_frames,
        overlap_frames=overlap_frames
    ).to(device)
    pipeline.eval()

    # 6. Biên dịch (Scripting)
    print("Scripting the pipeline... (This may take a moment)")
    try:
        scripted_pipeline = torch.jit.script(pipeline)
        
        # 7. Lưu file TorchScript
        scripted_pipeline.save(args.output)
        print(f"Successfully scripted and saved to {args.output}")

    except Exception as e:
        print(f"Failed to script the model. Error: {e}")
        print("Attempting to trace instead... (May be less robust)")
        
        # Nếu script thất bại (ví dụ: do logic quá phức tạp)
        # thử dùng trace. Cần dummy input.
        try:
            # Tạo input mẫu
            dummy_waveform = torch.randn(int(cfg_data['sample_rate'] * 5.0)).to(device) # 5 giây
            dummy_steps = torch.tensor(20).to(device) # Lấy từ config hoặc mặc định 20
            
            traced_pipeline = torch.jit.trace(pipeline, (dummy_waveform, dummy_steps))
            traced_pipeline.save(args.output)
            print(f"Successfully traced and saved to {args.output}")
        except Exception as e_trace:
            print(f"Tracing also failed. Error: {e_trace}")
            print("Could not export the model.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export Denoise model to TorchScript.')
    parser.add_argument('--config', type=str, required=True, help='Path to the config YAML file (phải chứa đầy đủ các key data, stft, model, test).')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint file (.pth).')
    parser.add_argument('--output', type=str, required=True, help='Path to save the exported TorchScript file (.pt).')

    args = parser.parse_args()

    # Tạo thư mục output nếu chưa có
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    export_main(args)