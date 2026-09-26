# This extracts the development-split features used for calibration.

import argparse
import glob
import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from utils import read_wav, device_from_arg
from diffusion_model import RevGuidedDiffusion, make_args
from inference import get_detector
from purify import purify_waveform, _pad_numpy

DEFAULT_CKPT = "models/model300000.pt"


@torch.no_grad()
def bona_logits(score_fn, device, wavs, max_len, batch_size=16):
    scores = []
    for i in range(0, len(wavs), batch_size):
        batch = wavs[i:i + batch_size]
        x = torch.from_numpy(np.stack([_pad_numpy(w, max_len) for w in batch])).to(device)
        scores.extend(score_fn(x).detach().cpu().tolist())
    return scores


def extract_sweep_features(wavs, score_fn, max_len, diff, device, t_high_list, batch_size=16):
    s_low = bona_logits(score_fn, device, wavs, max_len, batch_size=batch_size)
    print(f"    s_low done  ({len(wavs)} samples, mean={np.mean(s_low):+.3f})", flush=True)

    s_high_map, d_map = {}, {}
    for t in t_high_list:
        t0 = time.time()
        purified = [purify_waveform(w, diff, t=t)
                    for w in tqdm(wavs, desc=f"  t={t:>3}", unit="wav", ncols=80, leave=False)]
        s_high_t = bona_logits(score_fn, device, purified, max_len, batch_size=batch_size)
        s_high_map[t] = s_high_t
        d_map[t]      = [lo - hi for lo, hi in zip(s_low, s_high_t)]
        d_arr = np.asarray(d_map[t])
        print(f"    t={t:>3}  d_mean={d_arr.mean():+5.2f}±{d_arr.std():.2f}  ({time.time()-t0:.0f}s)", flush=True)
    return {"s_low": s_low, "s_high": s_high_map, "d": d_map}


def _summarize(s_low, s_high_map, d_map, t_high_list):
    s_low_arr = np.asarray(s_low, dtype=np.float64)
    summary = {
        "n":            int(s_low_arr.size),
        "s_low_mean":   float(s_low_arr.mean()),
        "s_low_std":    float(s_low_arr.std()),
        "s_low_median": float(np.median(s_low_arr)),
    }
    for t in t_high_list:
        s_high_arr = np.asarray(s_high_map[t], dtype=np.float64)
        d_arr      = np.asarray(d_map[t],      dtype=np.float64)
        summary[f"s_high_{t}_mean"] = float(s_high_arr.mean())
        summary[f"s_high_{t}_std"]  = float(s_high_arr.std())
        summary[f"d_{t}_mean"]      = float(d_arr.mean())
        summary[f"d_{t}_std"]       = float(d_arr.std())
        summary[f"d_{t}_median"]    = float(np.median(d_arr))
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--detector",    default="aasist", choices=["aasist", "rawgat", "res_tssdnet"])
    p.add_argument("--dev_root",    required=True)
    p.add_argument("--bona_sub",    default=None)
    p.add_argument("--clean_sub",   default=None)
    p.add_argument("--adv_sub",     default=None)
    p.add_argument("--classes",     default="bona,clean_spoof,adv",
                   help="comma-separated subset of: bona, clean_spoof, adv")
    p.add_argument("--attack_name", default="unknown")
    p.add_argument("--t_high_list", default="120,180,240")
    p.add_argument("--N",           type=int, default=300)
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--checkpoint",  default=DEFAULT_CKPT)
    p.add_argument("--device",      default="auto")
    p.add_argument("--out_json",    required=True)
    return p.parse_args()


def main():
    a = parse_args()
    t_high_list = [int(x) for x in a.t_high_list.split(",")]
    device = device_from_arg(a.device)

    print(f"[extract]  detector={a.detector}  attack={a.attack_name}  "
          f"t_high_list={t_high_list}  N={a.N}  device={device}", flush=True)
    print(f"  dev_root = {a.dev_root}", flush=True)
    print(f"  out_json = {a.out_json}", flush=True)

    diff_args = make_args(t=t_high_list[0], checkpoint=a.checkpoint)
    diff = RevGuidedDiffusion(diff_args, device=device)
    _, score_fn, max_len = get_detector(a.detector, device)

    classes_to_extract = [c.strip() for c in a.classes.split(",") if c.strip()]
    all_subs = {"bona": a.bona_sub, "clean_spoof": a.clean_sub, "adv": a.adv_sub}
    for lab in classes_to_extract:
        if lab not in all_subs:
            raise ValueError(f"--classes: unknown class '{lab}' (valid: bona, clean_spoof, adv)")
        if all_subs[lab] is None:
            raise ValueError(f"a *_sub directory is required for class '{lab}'")
    subs = {lab: all_subs[lab] for lab in classes_to_extract}

    files_per_class = {}
    for lab, sub in subs.items():
        fs = sorted(glob.glob(os.path.join(a.dev_root, sub, "*.flac")))
        if a.N > 0:
            fs = fs[:a.N]
        files_per_class[lab] = fs
        print(f"  {lab:<12} {len(fs):>4} files  ({sub})", flush=True)

    out_samples, out_summary = {}, {}
    grand_t0 = time.time()
    cfg = {"detector": a.detector, "attack": a.attack_name, "t_high_list": t_high_list,
           "N": a.N, "dev_root": a.dev_root, "subs": subs, "checkpoint": a.checkpoint}
    os.makedirs(os.path.dirname(a.out_json) or ".", exist_ok=True)

    for lab, fs in files_per_class.items():
        t0   = time.time()
        wavs = [read_wav(f) for f in fs]
        keys = [os.path.splitext(os.path.basename(f))[0] for f in fs]

        feats = extract_sweep_features(wavs, score_fn, max_len, diff, device,
                                       t_high_list=t_high_list, batch_size=a.batch_size)

        records = []
        for i, k in enumerate(keys):
            rec = {"key": k, "s_low": feats["s_low"][i]}
            for t in t_high_list:
                rec[f"s_high_{t}"] = feats["s_high"][t][i]
                rec[f"d_{t}"]      = feats["d"][t][i]
            records.append(rec)
        out_samples[lab] = records
        out_summary[lab] = _summarize(feats["s_low"], feats["s_high"], feats["d"], t_high_list)

        s = out_summary[lab]
        line = f"  {lab:<12}  s_low={s['s_low_mean']:+5.2f}±{s['s_low_std']:.2f}"
        for t in t_high_list:
            line += f"   d@{t}={s[f'd_{t}_mean']:+5.2f}±{s[f'd_{t}_std']:.2f}"
        print(line + f"   ({time.time() - t0:.0f}s)", flush=True)

        with open(a.out_json, "w") as f:
            json.dump({"config": cfg, "summary": out_summary, "samples": out_samples}, f, indent=2)

    print(f"\nelapsed total: {time.time() - grand_t0:.0f}s", flush=True)
    print(f"saved -> {a.out_json}", flush=True)


if __name__ == "__main__":
    main()
