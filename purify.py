# This is the main runner of the purification defense.

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import yaml
from tqdm import tqdm

from utils import (read_wav, write_flac, device_from_arg,
                   wav2spec, spec2wav, spec_to_windows, windows_to_spec)
from diffusion_model import (RevGuidedDiffusion, make_args,
                             window_to_tensor, tensor_to_window)
from inference import get_detector


def _pad_numpy(w, max_len):
    w = w.astype(np.float32)
    L = len(w)
    if L >= max_len:
        return w[:max_len]
    reps = (max_len // L) + 1
    return np.tile(w, reps)[:max_len]


@torch.no_grad()
def purify_waveform(wav, diff, t):
    diff.args.t = int(t)

    S_full, P = wav2spec(wav)
    S_main = S_full[:256, :].copy()
    S_ny   = S_full[256:257, :].copy()
    L          = len(wav)
    original_T = S_main.shape[1]
    windows = spec_to_windows(S_main, win_size=256)

    purified_windows = []
    for window in windows:
        img = window_to_tensor(window).to(diff.device)
        out = diff.image_editing_sample(img)
        purified_windows.append(tensor_to_window(out))

    S_pur = windows_to_spec(purified_windows, original_T=original_T)
    n = min(S_pur.shape[1], P.shape[1])
    S_recon = np.concatenate([S_pur[:, :n], S_ny[:, :n]], axis=0)
    w = spec2wav(S_recon, P[:, :n], length=L)
    peak = float(np.max(np.abs(w)))
    if peak > 1.0:
        w = w / peak
    return w.astype(np.float32)


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)

    mode = cfg.get("mode", "dgap")
    if mode not in ("dgap", "uniform"):
        raise ValueError(f"{path}: unknown mode '{mode}'")
    if "checkpoint" not in cfg or cfg["checkpoint"] is None:
        raise ValueError(f"{path}: checkpoint missing")
    if "defense" not in cfg or cfg["defense"].get("t_score") is None:
        raise ValueError(f"{path}: defense.t_score missing")

    out = {
        "attack":      cfg.get("attack", "unknown"),
        "mode":        mode,
        "checkpoint":  cfg["checkpoint"],
        "t_score":     int(cfg["defense"]["t_score"]),
    }

    if mode == "dgap":
        gate = cfg["gate"]
        if gate.get("t") is None or gate.get("tau") is None:
            raise ValueError(f"{path}: gate.t or gate.tau missing")
        out["gate_t"] = int(gate["t"])
        out["tau"]    = float(gate["tau"])
    return out


def _save_flac_if_dir(save_dir, key, wav):
    if save_dir:
        write_flac(os.path.join(save_dir, f"{key}.flac"), wav.astype(np.float32))


def _score_one(score_fn, device, wav, max_len):
    x = torch.from_numpy(_pad_numpy(wav, max_len)[None, :]).to(device)
    with torch.no_grad():
        return float(score_fn(x).detach().cpu().item())


def _resume_load(save_dir, key, resume):
    if resume and save_dir:
        path = os.path.join(save_dir, f"{key}.flac")
        if os.path.exists(path):
            return read_wav(path)
    return None


# Probe at the gate level, then purify strongly only if d > tau.
def defend_batch_dgap(wavs, keys, cfg, diff, score_fn, max_len, device, save_dir=None, resume=False):
    n = len(wavs)
    s_low_arr        = np.zeros(n, dtype=np.float64)
    d_arr            = np.zeros(n, dtype=np.float64)
    gate_fires_arr   = np.zeros(n, dtype=bool)
    s_high_strong_arr = np.full(n, np.nan, dtype=np.float64)
    score_arr        = np.zeros(n, dtype=np.float64)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for i in tqdm(range(n), desc="dgap per-sample", file=sys.stderr, mininterval=2.0):
        w, k = wavs[i], keys[i]
        s_low = _score_one(score_fn, device, w, max_len)
        pg = purify_waveform(w, diff, cfg["gate_t"])
        s_high_gate = _score_one(score_fn, device, pg, max_len)
        d = s_low - s_high_gate
        fires = bool(d > cfg["tau"])

        s_low_arr[i] = s_low
        d_arr[i] = d
        gate_fires_arr[i] = fires

        if fires:
            ps = _resume_load(save_dir, k, resume)
            if ps is None:
                ps = purify_waveform(w, diff, cfg["t_score"])
                _save_flac_if_dir(save_dir, k, ps)
            s_high_strong = _score_one(score_fn, device, ps, max_len)
            s_high_strong_arr[i] = s_high_strong
            score_arr[i] = s_high_strong
        else:
            score_arr[i] = s_low
            _save_flac_if_dir(save_dir, k, w)

    return {
        "s_low":         s_low_arr,
        "d":             d_arr,
        "gate_fires":    gate_fires_arr,
        "s_high_strong": s_high_strong_arr,
        "score":         score_arr,
    }


def defend_batch_uniform(wavs, keys, cfg, diff, score_fn, max_len, device, save_dir=None, resume=False):
    n = len(wavs)
    s_low_arr  = np.zeros(n, dtype=np.float64)
    s_high_arr = np.zeros(n, dtype=np.float64)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for i in tqdm(range(n), desc="uniform per-sample", file=sys.stderr, mininterval=2.0):
        w, k = wavs[i], keys[i]
        s_low_arr[i]  = _score_one(score_fn, device, w, max_len)
        p = _resume_load(save_dir, k, resume)
        if p is None:
            p = purify_waveform(w, diff, cfg["t_score"])
            _save_flac_if_dir(save_dir, k, p)
        s_high_arr[i] = _score_one(score_fn, device, p, max_len)

    return {
        "s_low":    s_low_arr,
        "s_high":   s_high_arr,
        "score":    s_high_arr,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",            required=True, help="YAML config path")
    p.add_argument("--input",             required=True, help="audio file or directory")
    p.add_argument("--out_json",          required=True, help="per-sample output JSON")
    p.add_argument("--detector",          default="aasist", choices=["aasist", "rawgat", "res_tssdnet"],
                   help="Detector to score with (default: aasist)")
    p.add_argument("--save_purified_dir", default=None,
                   help="optional: save defended .flac per sample (purified if gate fires, original copy otherwise)")
    p.add_argument("--resume", action="store_true",
                   help="reuse already-saved .flac in save_purified_dir (skip re-purify); for resuming interrupted runs")
    p.add_argument("--device",            default="auto")
    return p.parse_args()


def main():
    a = parse_args()
    device = device_from_arg(a.device)
    cfg = load_config(a.config)
    cfg["detector"] = a.detector  # traceability in output JSON

    print(f"[purify]  attack={cfg['attack']}  mode={cfg['mode']}  detector={cfg['detector']}", flush=True)
    print(f"  checkpoint      = {cfg['checkpoint']}", flush=True)
    if cfg["mode"] == "dgap":
        print(f"  gate.t          = {cfg['gate_t']}", flush=True)
        print(f"  gate.tau        = {cfg['tau']:+.4f}", flush=True)
    print(f"  defense.t_score = {cfg['t_score']}", flush=True)
    print(f"  device          = {device}", flush=True)

    if os.path.isdir(a.input):
        files = sorted(glob.glob(os.path.join(a.input, "*.flac")) +
                       glob.glob(os.path.join(a.input, "*.wav")))
    elif os.path.isfile(a.input):
        files = [a.input]
    else:
        raise FileNotFoundError(a.input)
    print(f"  input           = {a.input}   ({len(files)} files)", flush=True)

    diff_args = make_args(t=cfg["t_score"], checkpoint=cfg["checkpoint"])
    diff = RevGuidedDiffusion(diff_args, device=device)
    _, score_fn, max_len = get_detector(a.detector, device)

    if a.save_purified_dir:
        os.makedirs(a.save_purified_dir, exist_ok=True)

    t0 = time.time()
    keys = [os.path.splitext(os.path.basename(f))[0] for f in files]
    wavs = [read_wav(f) for f in files]

    if cfg["mode"] == "dgap":
        out = defend_batch_dgap(wavs, keys, cfg, diff, score_fn, max_len, device,
                                  save_dir=a.save_purified_dir, resume=a.resume)
    else:
        out = defend_batch_uniform(wavs, keys, cfg, diff, score_fn, max_len, device,
                                     save_dir=a.save_purified_dir, resume=a.resume)

    records = []
    for i, k in enumerate(keys):
        if cfg["mode"] == "dgap":
            rec = {
                "key":           k,
                "score":         float(out["score"][i]),
                "s_low":         float(out["s_low"][i]),
                "d":             float(out["d"][i]),
                "gate_fires":    bool(out["gate_fires"][i]),
                "s_high_strong": float(out["s_high_strong"][i]) if out["gate_fires"][i] else None,
            }
        else:
            rec = {
                "key":    k,
                "score":  float(out["score"][i]),
                "s_low":  float(out["s_low"][i]),
                "s_high": float(out["s_high"][i]),
            }
        records.append(rec)
        # Note: .flac files are saved incrementally inside defend_batch_*

    summary = {
        "n":           len(records),
        "score_mean":  float(np.mean(out["score"])),
        "s_low_mean":  float(np.mean(out["s_low"])),
    }
    if cfg["mode"] == "dgap":
        summary["n_gate_fires"] = int(out["gate_fires"].sum())
        summary["d_mean"]       = float(np.mean(out["d"]))
    else:
        summary["s_high_mean"]  = float(np.mean(out["s_high"]))

    os.makedirs(os.path.dirname(a.out_json) or ".", exist_ok=True)
    with open(a.out_json, "w") as f:
        json.dump({"config": cfg, "summary": summary, "records": records}, f, indent=2)

    print(f"\nelapsed: {time.time() - t0:.0f}s", flush=True)
    print(f"  {summary}", flush=True)
    print(f"saved -> {a.out_json}", flush=True)


if __name__ == "__main__":
    main()
