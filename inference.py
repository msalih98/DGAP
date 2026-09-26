# These are the detector wrappers used for scoring and attacks.

import importlib.util
import json
import os

import torch
import yaml

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

BONAFIDE_LABEL = 1


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pad_wave(x, max_len):
    L = x.shape[-1]
    if L >= max_len:
        return x[..., :max_len]
    num_repeats = (max_len // L) + 1
    repeats = [1] * (x.dim() - 1) + [num_repeats]
    return x.repeat(*repeats)[..., :max_len]


# --------------------------- AASIST ---------------------------
AASIST_ROOT    = os.path.join(REPO_ROOT, "aasist")
AASIST_CONFIG  = os.path.join(AASIST_ROOT, "config", "AASIST.conf")
AASIST_WEIGHTS = os.path.join(AASIST_ROOT, "models", "weights", "AASIST.pth")
AASIST_MAX_LEN = 64600


def load_aasist(device, config_path=AASIST_CONFIG, weights=AASIST_WEIGHTS):
    with open(config_path) as f:
        config = json.load(f)
    mod = _load_module(os.path.join(AASIST_ROOT, "models", "AASIST.py"), "aasist_model")
    model = mod.Model(config["model_config"]).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    return model


def pad_aasist_torch(x, max_len=AASIST_MAX_LEN):
    return _pad_wave(x, max_len)


def aasist_forward(model, wav_tensor):
    return model(pad_aasist_torch(wav_tensor, AASIST_MAX_LEN))


# -------------------------- RawGAT-ST --------------------------
RAWGAT_ROOT    = os.path.join(REPO_ROOT, "rawgat_st")
RAWGAT_CONFIG  = os.path.join(RAWGAT_ROOT, "model_config_RawGAT_ST.yaml")
RAWGAT_WEIGHTS = os.path.join(RAWGAT_ROOT, "Pre_trained_models", "RawGAT_ST_mul", "Best_epoch.pth")
RAWGAT_MAX_LEN = 64600


def load_rawgat(device, config_path=RAWGAT_CONFIG, weights=RAWGAT_WEIGHTS):
    mod = _load_module(os.path.join(RAWGAT_ROOT, "model.py"), "rawgat_model")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model = mod.RawGAT_ST(cfg["model"], device).to(device)
    sd = torch.load(weights, map_location=device)
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return model


def pad_rawgat_torch(x, max_len=RAWGAT_MAX_LEN):
    return _pad_wave(x, max_len)


# ------------------------- Res-TSSDNet -------------------------
# input length is fixed at 96000 samples (6 s @ 16 kHz) by the model's
# hardcoded max-pool; class convention is inverted: bonafide = index 0
TSSDNET_ROOT    = os.path.join(REPO_ROOT, "tssdnet")
TSSDNET_WEIGHTS = os.path.join(
    TSSDNET_ROOT, "pretrained",
    "Res_TSSDNet_time_frame_61_ASVspoof2019_LA_Loss_0.0017_dEER_0.74%_eEER_1.64%.pth")
TSSDNET_MAX_LEN = 96000
TSSDNET_BONAFIDE_IDX = 0


def load_tssdnet(device, weights=TSSDNET_WEIGHTS):
    mod = _load_module(os.path.join(TSSDNET_ROOT, "models.py"), "tssdnet_model")
    model = mod.SSDNet1D()
    ckpt = torch.load(weights, map_location=device)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


def pad_tssdnet_torch(x, max_len=TSSDNET_MAX_LEN):
    if x.dim() == 3:
        x = x.squeeze(1)
    B, T = x.shape
    if T >= max_len:
        x = x[:, :max_len]
    else:
        reps = max_len // T + 1
        x = x.repeat(1, reps)[:, :max_len]
    return x.unsqueeze(1)


# ---------------------- unified accessors ----------------------
def get_detector_logits(name, device):
    if name == "aasist":
        model = load_aasist(device)
        def logits_fn(x):
            _, logits = aasist_forward(model, x)
            return logits
        return model, logits_fn, AASIST_MAX_LEN, 1, 0
    elif name == "rawgat":
        model = load_rawgat(device)
        def logits_fn(x):
            if x.dim() == 1:
                x = x.unsqueeze(0)
            return model(pad_rawgat_torch(x, RAWGAT_MAX_LEN), Freq_aug=False)
        return model, logits_fn, RAWGAT_MAX_LEN, 1, 0
    elif name == "res_tssdnet":
        model = load_tssdnet(device)
        def logits_fn(x):
            return model(pad_tssdnet_torch(x, TSSDNET_MAX_LEN))
        return model, logits_fn, TSSDNET_MAX_LEN, 0, 1
    raise ValueError(f"Unknown detector: {name}")


def get_detector(name, device):
    model, logits_fn, max_len, bona_idx, _ = get_detector_logits(name, device)
    return model, lambda x: logits_fn(x)[:, bona_idx], max_len
