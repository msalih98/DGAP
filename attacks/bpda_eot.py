# This is the BPDA+EOT attack against the defense itself.

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import read_wav, write_flac, set_seed, device_from_arg, REPO_ROOT
from inference import get_detector_logits
from diffusion_model import RevGuidedDiffusion, make_args
from purify import purify_waveform, load_config

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
def score_np(logits_fn, wav, device, bona_idx):
    x = torch.from_numpy(wav.astype(np.float32))[None, :].to(device)
    return float(logits_fn(x)[0, bona_idx].cpu())


def defense_forward(wav, diff, logits_fn, device, cfg, bona_idx):
    if cfg["mode"] == "uniform":
        return purify_waveform(wav, diff, cfg["t_score"]), True
    s_low = score_np(logits_fn, wav, device, bona_idx)
    probe = purify_waveform(wav, diff, cfg["gate_t"])
    d = s_low - score_np(logits_fn, probe, device, bona_idx)
    if d > cfg["tau"]:
        return purify_waveform(wav, diff, cfg["t_score"]), True
    return wav.astype(np.float32), False


def bpda_eot_pgd(wav, forward_fn, logits_fn, device, eps, steps, eot, bona_idx):
    x0 = torch.from_numpy(wav.astype(np.float32)).to(device)
    target = torch.full((eot,), bona_idx, device=device)
    alpha = 2.5 * eps / max(steps, 1)
    x_adv = (x0 + torch.empty_like(x0).uniform_(-eps, eps)).clamp(-1, 1).detach()

    for _ in range(steps):
        xa = x_adv.cpu().numpy()
        draws = [forward_fn(xa)[0] for _ in range(eot)]
        vt = torch.from_numpy(np.stack(draws)).to(device).requires_grad_(True)
        loss = F.cross_entropy(logits_fn(vt), target)
        grad = torch.autograd.grad(loss, vt)[0].mean(0)
        x_adv = x0 + (x_adv - alpha * grad.sign() - x0).clamp(-eps, eps)
        x_adv = x_adv.clamp(-1, 1).detach()
    return x_adv.cpu().numpy().astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--detector", default="aasist", choices=["aasist", "rawgat"])
    p.add_argument("--config",   required=True, help="defense config to attack")
    p.add_argument("--split",    choices=["dev", "eval"], required=True)
    p.add_argument("--eps",      type=float, default=0.005)
    p.add_argument("--steps",    type=int, default=20)
    p.add_argument("--eot",      type=int, default=8)
    p.add_argument("--n_spoof",  type=int, default=120)
    p.add_argument("--n_bona",   type=int, default=120)
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
    cfg = load_config(args.config)
    _, logits_fn, _, bona_idx, spoof_idx = get_detector_logits(args.detector, device)
    diff = RevGuidedDiffusion(make_args(t=cfg["t_score"], checkpoint=cfg["checkpoint"]),
                              device=device)

    print(f"[bpda+eot]  detector={args.detector}  mode={cfg['mode']}  "
          f"eps={args.eps}  steps={args.steps}  eot={args.eot}  device={device}", flush=True)

    forward_fn = lambda w: defense_forward(w, diff, logits_fn, device, cfg, bona_idx)

    protocol_path, audio_dir = split_paths(args.split)
    if args.spoof_keys_file or args.bona_keys_file:
        assert args.spoof_keys_file and args.bona_keys_file, "provide BOTH key files"
        spoof_keys = read_keys_file(args.spoof_keys_file)
        bona_keys  = read_keys_file(args.bona_keys_file)
    else:
        spoof_keys, bona_keys = load_keys(protocol_path, args.n_spoof, args.n_bona, args.seed)

    os.makedirs(args.out_root, exist_ok=True)
    bona_scores, bona_fired = [], []
    for key in tqdm(bona_keys, desc="bonafide", file=sys.stderr):
        src = os.path.join(audio_dir, key + ".flac")
        if not os.path.exists(src):
            continue
        defended, fired = forward_fn(read_wav(src))
        bona_scores.append(score_np(logits_fn, defended, device, bona_idx))
        bona_fired.append(fired)

    adv_dir = os.path.join(args.out_root, "adv")
    rows, adv_scores, adv_fired = [], [], []
    for key in tqdm(spoof_keys, desc="bpda_eot", file=sys.stderr):
        src = os.path.join(audio_dir, key + ".flac")
        if not os.path.exists(src):
            continue
        wav = read_wav(src)
        adv = bpda_eot_pgd(wav, forward_fn, logits_fn, device,
                           args.eps, args.steps, args.eot, bona_idx)
        write_flac(os.path.join(adv_dir, key + ".flac"), adv)
        defended, fired = forward_fn(adv)
        s = score_np(logits_fn, defended, device, bona_idx)
        adv_scores.append(s)
        adv_fired.append(fired)
        rows.append({"key": key, "score_defended": s, "gate_fired": bool(fired)})

    from utils import compute_eer
    eer_adv, _ = compute_eer(bona_scores, adv_scores)
    summary = {
        "attack": "bpda_eot", "detector": args.detector, "mode": cfg["mode"],
        "config": args.config, "split": args.split,
        "eps": args.eps, "steps": args.steps, "eot": args.eot,
        "n_bona": len(bona_scores), "n_adv": len(adv_scores),
        "eer_adv": eer_adv,
        "benign_flag_rate": float(np.mean(bona_fired)) if bona_fired else 0.0,
        "adv_flag_rate":    float(np.mean(adv_fired)) if adv_fired else 0.0,
        "rows": rows,
    }
    with open(os.path.join(args.out_root, "attack_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
