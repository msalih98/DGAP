# Shared audio and spectrogram utilities.

import os
import random

import numpy as np
import librosa
import torch
from PIL import Image
from sklearn.metrics import roc_curve

try:
    import soundfile as sf
except ModuleNotFoundError:
    sf = None

SR         = 16000
N_FFT      = 512
HOP_LENGTH = 128

DB_MIN    = -120.0
DB_MAX    = 25.0
DB_OFFSET = 20.0

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))


def device_from_arg(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_wav(path: str) -> np.ndarray:
    if sf is not None:
        wav, sr = sf.read(path)
    else:
        wav, sr = librosa.load(path, sr=SR, mono=True)
    if wav.ndim > 1:
        wav = wav[:, 0]
    if sr != SR:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=SR)
    return wav.astype(np.float32)


def write_flac(path: str, wav: np.ndarray, sr: int = SR) -> None:
    if sf is None:
        raise RuntimeError("python-soundfile is required for FLAC output")
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    sf.write(path, np.clip(np.asarray(wav, dtype=np.float32), -1, 1), sr,
             format="FLAC", subtype="PCM_16")


def amp_to_db(x):
    return 20.0 * np.log10(np.maximum(1e-5, x))


def db_to_amp(x):
    return np.power(10.0, x * 0.05)


def normalize(S):
    return (S - DB_MIN) / (DB_MAX - DB_MIN)


def denormalize(S):
    return S * (DB_MAX - DB_MIN) + DB_MIN


def wav2spec(wav):
    D    = librosa.stft(wav, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_db = amp_to_db(np.abs(D)) - DB_OFFSET
    S    = normalize(S_db)
    P    = np.angle(D)
    return S, P


def spec2wav(S, P, length=None):
    S_db = denormalize(S) + DB_OFFSET
    mag  = db_to_amp(S_db)
    # Pad the dropped Nyquist bin with zeros: librosa would otherwise infer
    # n_fft = 2*(rows-1), which mismatches the forward STFT's n_fft=512.
    expected = N_FFT // 2 + 1
    if mag.shape[0] < expected:
        pad_rows = expected - mag.shape[0]
        mag = np.concatenate([mag, np.zeros((pad_rows, mag.shape[1]), dtype=mag.dtype)], axis=0)
        P   = np.concatenate([P,   np.zeros((pad_rows, P.shape[1]),   dtype=P.dtype)],   axis=0)
    return librosa.istft(mag * np.exp(1j * P), n_fft=N_FFT, hop_length=HOP_LENGTH, length=length)


def spec_to_windows(S, win_size=256):
    S = S[:256, :]
    T = S.shape[1]
    nwindow = T // win_size
    windows = []
    if nwindow == 0:
        pad = np.zeros((256, win_size - T))
        windows.append(np.concatenate([S, pad], axis=1))
    else:
        for w in range(nwindow):
            windows.append(S[:, w * win_size:(w + 1) * win_size])
        if T > nwindow * win_size:
            windows.append(S[:, -win_size:])
    return windows


def windows_to_spec(windows, original_T):
    nwindow = len(windows)
    if nwindow == 1:
        return windows[0][:, :original_T]

    S_recon = np.concatenate(windows[:-1], axis=1)
    remaining = original_T - S_recon.shape[1]
    if remaining > 0:
        S_recon = np.concatenate([S_recon, windows[-1][:, -remaining:]], axis=1)
    return S_recon[:, :original_T]


def spec_to_img(spec):
    arr = (spec * 255).astype(np.uint8)
    return Image.fromarray(np.stack([arr, arr, arr], axis=2))


def compute_eer(bona, spoof):
    y = np.r_[np.ones(len(bona)), np.zeros(len(spoof))]
    s = np.r_[bona, spoof]
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fnr[i] + fpr[i]) / 2.0), float(thr[i])
