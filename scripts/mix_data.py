import os, glob, random, math
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
import pandas as pd

## CONFIGURATION
SR = 16000  # Sample rate
TRAIN_SNRs = [-5, 0, 5]  # SNR levels for training data
TEST_SNRs = [-2.5, 2.5]  # SNR levels for testing data

RANDOM_SEED = 42

VIVOS_ROOT = 'data/vivos'
DEMAND_ROOT = 'data/demand'
OUT_ROOT = 'data/dataset'

TEST_NOISES = None

## UTILITY FUNCTIONS
def list_wavs(directory):
    return sorted(glob.glob(os.path.join(directory, '**', '*.wav'), recursive=True))

def load_mono(path, sr=SR):
    x, r = sf.read(path)
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    if r != sr:
        x = librosa.resample(x, orig_sr=r, target_sr=sr)
    return x.astype(np.float32)

def normalize_rms(x, target_rms=0.1):
    rms = np.sqrt(np.mean(x**2)) + 1e-12
    return x * (target_rms / rms)

def mix_at_snr(clean, noise, snr_db):
    L = len(clean)
    noise = noise[:L]

    clean_power = np.mean(clean ** 2) + 1e-12
    noise_power = np.mean(noise ** 2) + 1e-12

    snr_lin = 10 ** (snr_db / 10)
    scale = np.sqrt(clean_power / (snr_lin * noise_power))
    noise_scaled = noise * scale

    mixed = clean + noise_scaled

    peak = np.max(np.abs(mixed))

    if peak > 0.999:
        mixed = mixed / (peak + 1e-12) * 0.999

    return mixed, noise_scaled

def ensure_dir(directory):
    os.makedirs(directory, exist_ok=True)

def vivos_split(vivos_root):
    train_files = list_wavs(os.path.join(vivos_root, "train", "waves"))
    test_files  = list_wavs(os.path.join(vivos_root, "test",  "waves"))
    if not train_files:
        raise RuntimeError("Không tìm thấy file train trong vivos/train/waves")
    if not test_files:
        raise RuntimeError("Không tìm thấy file test trong vivos/test/waves")
    return train_files, test_files

def demand_files(demand_root):
    """Trả về dict: env -> list các file chxx.wav"""
    envs = {}
    for env_name in sorted(next(os.walk(demand_root))[1]):
        env_dir = os.path.join(demand_root, env_name)
        wavs = list_wavs(env_dir)
        if wavs:
            envs[env_name] = wavs
    if not envs:
        raise RuntimeError("Không tìm thấy noise trong demand/*/chxx.wav")
    return envs

def pick_noise_segment(noise_file, target_len):
    n = load_mono(noise_file, SR)
    if len(n) >= target_len:
        start = random.randint(0, len(n) - target_len)
        return n[start:start+target_len]
    else:
        reps = int(math.ceil(target_len / len(n)))
        n = np.tile(n, reps)
        return n[:target_len]
    

## MAIN
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

train_clean_files, test_clean_files = vivos_split(VIVOS_ROOT)
env2files = demand_files(DEMAND_ROOT)
all_envs = sorted(env2files.keys())

if TEST_NOISES is None:
    if len(all_envs) < 6:
        raise RuntimeError("Cần ít nhất 6 môi trường noise trong demand để tách train/test")
    TEST_NOISE_ENVS = random.sample(all_envs, 5)

TRAIN_NOISE_ENVS = [e for e in all_envs if e not in TEST_NOISE_ENVS]

print(f"[NOISE] Tổng env: {len(all_envs)}")
print(f"[NOISE] Train envs ({len(TRAIN_NOISE_ENVS)}): {TRAIN_NOISE_ENVS}")
print(f"[NOISE] Test  envs ({len(TEST_NOISE_ENVS)}): {TEST_NOISE_ENVS}")


for split in ["train", "test"]:
    ensure_dir(os.path.join(OUT_ROOT, split, "clean"))
    ensure_dir(os.path.join(OUT_ROOT, split, "noisy"))

train_log, test_log = [], []

print("==MIX TRAIN==")
for clean_path in tqdm(train_clean_files):
    clean = load_mono(clean_path, SR)

    env = random.choice(TRAIN_NOISE_ENVS)
    noise_file = random.choice(env2files[env])
    snr_db = random.choice(TRAIN_SNRs)

    noise_segment = pick_noise_segment(noise_file, len(clean))
    noisy, _ = mix_at_snr(clean, noise_segment, snr_db)

    fname = os.path.basename(clean_path)
    out_clean = os.path.join(OUT_ROOT, "train", "clean", fname)
    out_noisy = os.path.join(OUT_ROOT, "train", "noisy", fname)
    
    sf.write(out_clean, clean, SR)
    sf.write(out_noisy, noisy, SR)

    train_log.append({
        "split": "train",
        "file": fname,
        "clean_path": out_clean,
        "noisy_path": out_noisy,
        "sr": SR,
        "noise_env": env,
        "noise_file": noise_file,
        "snr_db": snr_db
    })


print("==MIX TEST==")
for clean_path in tqdm(test_clean_files):
    clean = load_mono(clean_path, SR)

    env = random.choice(TEST_NOISE_ENVS)
    noise_file = random.choice(env2files[env])
    snr_db = random.choice(TEST_SNRs)

    noise_segment = pick_noise_segment(noise_file, len(clean))

    clean = normalize_rms(clean)
    noise_segment = normalize_rms(noise_segment)

    noisy, _ = mix_at_snr(clean, noise_segment, snr_db)

    fname = os.path.basename(clean_path)
    out_clean = os.path.join(OUT_ROOT, "test", "clean", fname)
    out_noisy = os.path.join(OUT_ROOT, "test", "noisy", fname)
    
    sf.write(out_clean, clean, SR)
    sf.write(out_noisy, noisy, SR)

    test_log.append({
        "split": "test",
        "file": fname,
        "clean_path": out_clean,
        "noisy_path": out_noisy,
        "sr": SR,
        "noise_env": env,
        "noise_file": noise_file,
        "snr_db": snr_db
    })

pd.DataFrame(train_log).to_csv(os.path.join(OUT_ROOT, "train_log.csv"), index=False)
pd.DataFrame(test_log).to_csv(os.path.join(OUT_ROOT, "test_log.csv"), index=False)

print("==DONE==")