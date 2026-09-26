# This is the PGD-linf attack on a deepfake detector.

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
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


def pgd_linf_attack(model, logits_fn, wav, device, eps, steps, bona_idx):
    x0 = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0).to(device)
    delta = torch.empty_like(x0).uniform_(-1.0, 1.0) * eps
    x = (x0 + delta).clamp(-1, 1).detach()

    target = torch.tensor([bona_idx], device=device)
    alpha = 2.5 * eps / max(steps, 1)
    ce = nn.CrossEntropyLoss()
    for _ in range(steps):
        x.requires_grad_(True)
        loss = ce(logits_fn(x), target)
        model.zero_grad(set_to_none=True)
        loss.backward()
        g = x.grad.detach()
        x = x.detach() - alpha * g.sign()
        delta = (x - x0).detach().clamp(-eps, eps)
        x = (x0 + delta).clamp(-1, 1)
    return x.detach().cpu().numpy()[0].astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--detector", default="aasist", choices=["aasist", "rawgat", "res_tssdnet"])
    p.add_argument("--split",    choices=["dev", "eval"], required=True)
    p.add_argument("--eps",      type=float, default=0.005)
    p.add_argument("--steps",    type=int, default=20)
    p.add_argument("--n_spoof",  type=int, default=300)
    p.add_argument("--n_bona",   type=int, default=300)
    p.add_argument("--spoof_keys_file", default=None,
                   help="use an exact utterance-id list instead of random sampling")
    p.add_argument("--bona_keys_file",  default=None)
    p.add_argument("--out_root", required=True)
    p.add_argument("--seed",     type=int, default=123)
    p.add_argument("--device",   default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = device_from_arg(args.device)
    model, logits_fn, _, bona_idx, spoof_idx = get_detector_logits(args.detector, device)

    print(f"[pgd-linf]  detector={args.detector}  split={args.split}  "
          f"eps={args.eps}  steps={args.steps}  device={device}", flush=True)

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
    for key in tqdm(spoof_keys, desc="pgd_linf"):
        src = os.path.join(audio_dir, key + ".flac")
        if not os.path.exists(src):
            continue
        wav = read_wav(src)
        clean = predict(logits_fn, wav, device, bona_idx, spoof_idx)
        write_flac(os.path.join(clean_dir, key + ".flac"), wav)
        if clean["pred"] == bona_idx:
            rows.append({"key": key, "skipped": "already_bonafide", "clean": clean})
            continue
        adv = pgd_linf_attack(model, logits_fn, wav, device, args.eps, args.steps, bona_idx)
        adv_path = os.path.join(adv_dir, key + ".flac")
        write_flac(adv_path, adv)
        rows.append({"key": key, "clean": clean,
                     "adv": predict(logits_fn, read_wav(adv_path), device, bona_idx, spoof_idx)})

    attacked = [r for r in rows if "adv" in r]
    success  = [r for r in attacked if r["adv"]["pred"] == bona_idx]
    summary = {"attack": "pgd_linf", "detector": args.detector, "split": args.split,
               "eps": args.eps, "steps": args.steps,
               "n_attacked": len(attacked), "n_success": len(success),
               "success_rate": len(success) / max(len(attacked), 1),
               "out_root": args.out_root, "rows": rows}
    os.makedirs(args.out_root, exist_ok=True)
    with open(os.path.join(args.out_root, "attack_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
