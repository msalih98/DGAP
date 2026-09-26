# This is the C&W attack on a deepfake detector.

import argparse
import json
import os
import sys

import random

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import read_wav, write_flac, set_seed, device_from_arg, REPO_ROOT
from inference import get_detector_logits

SPLITS = {
    "dev":  ("ASVspoof2019.LA.cm.dev.trl.txt",  "ASVspoof2019_LA_dev/flac"),
    "eval": ("ASVspoof2019.LA.cm.eval.trl.txt", "ASVspoof2019_LA_eval/flac"),
}


def split_paths(split):
    proto, audio = SPLITS[split]
    la_root = os.path.join(REPO_ROOT, "data", "LA")
    return (os.path.join(la_root, "ASVspoof2019_LA_cm_protocols", proto),
            os.path.join(la_root, audio))


def load_keys(protocol_path, n_spoof, n_bona, seed):
    rng = random.Random(seed)
    spoof, bona = [], []
    with open(protocol_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            if parts[4] == "bonafide":
                bona.append(parts[1])
            elif parts[4] == "spoof":
                spoof.append(parts[1])
    return (rng.sample(spoof, min(n_spoof, len(spoof))),
            rng.sample(bona,  min(n_bona,  len(bona))))


def read_keys_file(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


@torch.no_grad()
def predict(logits_fn, wav, device, bona_idx, spoof_idx):
    x = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0).to(device)
    logits = logits_fn(x).detach().cpu()[0]
    return {"spoof_logit": float(logits[spoof_idx]),
            "bona_logit":  float(logits[bona_idx]),
            "pred":        int(torch.argmax(logits).item())}



def cw_attack(logits_fn, wav, device, bona_idx, spoof_idx,
              c=1.0, kappa=10.0, lr=0.01, steps=200, early_stop=True):
    x0 = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0).to(device)
    x0 = x0.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    w = torch.atanh(x0).detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([w], lr=lr)
    best_adv  = x0.detach()
    best_dist = float("inf")
    success_step = -1

    for step in range(steps):
        x_adv = torch.tanh(w)
        logits = logits_fn(x_adv)
        f = torch.clamp(logits[0, spoof_idx] - logits[0, bona_idx] + kappa, min=0.0)
        d = ((x_adv - x0) ** 2).sum()
        loss = c * f + d
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            cur = torch.tanh(w)
            cur_logits = logits_fn(cur)
            cur_pred = int(torch.argmax(cur_logits, dim=1).item())
            cur_margin = float((cur_logits[0, bona_idx] - cur_logits[0, spoof_idx]).item())
            cur_d = float(((cur - x0) ** 2).sum().item())

        if cur_pred == bona_idx and cur_margin >= kappa:
            if success_step < 0:
                success_step = step
            if cur_d < best_dist:
                best_dist = cur_d
                best_adv  = cur.detach()
            if early_stop and step >= 50:
                break

    adv_np = best_adv.cpu().numpy()[0].astype(np.float32)
    info = {
        "success":      success_step >= 0,
        "success_step": success_step,
        "final_step":   step,
        "l2_norm":      float(np.sqrt(((adv_np - wav) ** 2).sum())),
        "linf_norm":    float(np.abs(adv_np - wav).max()),
    }
    return adv_np, info


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--detector", default="aasist", choices=["aasist", "rawgat", "res_tssdnet"])
    p.add_argument("--split",    choices=["dev", "eval"], required=True)
    p.add_argument("--c",        type=float, default=1.0)
    p.add_argument("--kappa",    type=float, default=10.0)
    p.add_argument("--lr",       type=float, default=0.01)
    p.add_argument("--steps",    type=int, default=200)
    p.add_argument("--n_spoof",  type=int, default=300)
    p.add_argument("--n_bona",   type=int, default=300)
    p.add_argument("--spoof_keys_file", default=None,
                   help="use an exact utterance-id list instead of random sampling")
    p.add_argument("--bona_keys_file",  default=None)
    p.add_argument("--no_early_stop", action="store_true")
    p.add_argument("--out_root", required=True)
    p.add_argument("--seed",     type=int, default=123)
    p.add_argument("--device",   default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = device_from_arg(args.device)
    _, logits_fn, _, bona_idx, spoof_idx = get_detector_logits(args.detector, device)

    print(f"[cw]  detector={args.detector}  split={args.split}  c={args.c}  "
          f"kappa={args.kappa}  steps={args.steps}  device={device}", flush=True)

    protocol_path, audio_dir = split_paths(args.split)
    if args.spoof_keys_file or args.bona_keys_file:
        assert args.spoof_keys_file and args.bona_keys_file, "provide BOTH key files"
        spoof_keys = read_keys_file(args.spoof_keys_file)
        bona_keys  = read_keys_file(args.bona_keys_file)
    else:
        spoof_keys, bona_keys = load_keys(protocol_path, args.n_spoof, args.n_bona, args.seed)

    bona_dir  = os.path.join(args.out_root, "bona")
    clean_dir = os.path.join(args.out_root, "clean_spoof")
    adv_dir   = os.path.join(args.out_root, "adv")

    for key in tqdm(bona_keys, desc="bonafide"):
        src = os.path.join(audio_dir, key + ".flac")
        if os.path.exists(src):
            write_flac(os.path.join(bona_dir, key + ".flac"), read_wav(src))

    rows = []
    for key in tqdm(spoof_keys, desc="cw"):
        src = os.path.join(audio_dir, key + ".flac")
        if not os.path.exists(src):
            continue
        wav = read_wav(src)
        clean = predict(logits_fn, wav, device, bona_idx, spoof_idx)
        write_flac(os.path.join(clean_dir, key + ".flac"), wav)
        if clean["pred"] == bona_idx:
            rows.append({"key": key, "skipped": "already_bonafide", "clean": clean})
            continue
        adv, info = cw_attack(logits_fn, wav, device, bona_idx, spoof_idx,
                              c=args.c, kappa=args.kappa, lr=args.lr, steps=args.steps,
                              early_stop=not args.no_early_stop)
        adv_path = os.path.join(adv_dir, key + ".flac")
        write_flac(adv_path, adv)
        rows.append({"key": key, "clean": clean, "cw": info,
                     "adv": predict(logits_fn, read_wav(adv_path), device, bona_idx, spoof_idx)})

    attacked = [r for r in rows if "adv" in r]
    success  = [r for r in attacked if r["adv"]["pred"] == bona_idx]
    summary = {"attack": "cw", "detector": args.detector, "split": args.split,
               "c": args.c, "kappa": args.kappa, "lr": args.lr, "steps": args.steps,
               "n_attacked": len(attacked), "n_success": len(success),
               "success_rate": len(success) / max(len(attacked), 1),
               "out_root": args.out_root, "rows": rows}
    os.makedirs(args.out_root, exist_ok=True)
    with open(os.path.join(args.out_root, "attack_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
