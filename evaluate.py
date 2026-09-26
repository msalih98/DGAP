# This evaluates a detector on the bona / clean_spoof / adv directories
# and reports EER_clean, EER_adv and the objective y.

import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from utils import read_wav, device_from_arg, compute_eer
from inference import get_detector
from purify import _pad_numpy


@torch.no_grad()
def score_dir(score_fn, device, max_len, flac_dir, out_json, batch_size=8):
    paths = sorted(glob.glob(os.path.join(flac_dir, "*.flac")))
    if not paths:
        raise SystemExit(f"No .flac files found in {flac_dir}")
    rows = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        wavs = [read_wav(p) for p in batch_paths]
        x = torch.from_numpy(np.stack([_pad_numpy(w, max_len) for w in wavs])).to(device)
        scores = score_fn(x).detach().cpu().tolist()
        for p, s in zip(batch_paths, scores):
            rows.append({"key": os.path.basename(p), "score": float(s)})
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({"flac_dir": flac_dir, "n": len(rows), "rows": rows}, f, indent=2)
    print(f"  scored {len(rows)} files -> {out_json}", flush=True)
    return [r["score"] for r in rows]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--detector",  default="aasist", choices=["aasist", "rawgat", "res_tssdnet"])
    p.add_argument("--bona_dir",  required=True)
    p.add_argument("--clean_dir", required=True)
    p.add_argument("--adv_dir",   required=True)
    p.add_argument("--out_dir",   required=True)
    p.add_argument("--lam",       type=float, default=1.0,
                   help="weight of the adversarial EER term in the objective y")
    p.add_argument("--device",    default="auto")
    p.add_argument("--batch_size", type=int, default=8)
    args = p.parse_args()

    device = device_from_arg(args.device)
    print(f"[eval]  detector={args.detector}  device={device}", flush=True)
    _, score_fn, max_len = get_detector(args.detector, device)

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    bona  = score_dir(score_fn, device, max_len, args.bona_dir,
                      os.path.join(args.out_dir, "scores_bona.json"), args.batch_size)
    clean = score_dir(score_fn, device, max_len, args.clean_dir,
                      os.path.join(args.out_dir, "scores_clean_spoof.json"), args.batch_size)
    adv   = score_dir(score_fn, device, max_len, args.adv_dir,
                      os.path.join(args.out_dir, "scores_adv.json"), args.batch_size)
    print(f"  scoring done in {time.time() - t0:.1f}s", flush=True)

    print(f"\n  bona:        {len(bona)}  (mean: {np.mean(bona):+.3f})")
    print(f"  clean_spoof: {len(clean)}  (mean: {np.mean(clean):+.3f})")
    print(f"  adv:         {len(adv)}  (mean: {np.mean(adv):+.3f})")

    eer_clean, thr_c = compute_eer(bona, clean)
    eer_adv,   thr_d = compute_eer(bona, adv)
    y = eer_clean + args.lam * eer_adv

    print(f"\n  EER_clean: {eer_clean*100:6.2f}%   (thr={thr_c:+.3f})")
    print(f"  EER_adv:   {eer_adv*100:6.2f}%   (thr={thr_d:+.3f})")
    print(f"\n  y = EER_clean + {args.lam}*EER_adv = {y*100:6.2f}%")

    summary = {
        "detector": args.detector, "lambda": args.lam,
        "bona_dir": args.bona_dir, "clean_dir": args.clean_dir, "adv_dir": args.adv_dir,
        "n_bona": len(bona), "n_clean": len(clean), "n_adv": len(adv),
        "mean_bona":  float(np.mean(bona)),
        "mean_clean": float(np.mean(clean)),
        "mean_adv":   float(np.mean(adv)),
        "eer_clean": eer_clean,
        "eer_adv":   eer_adv,
        "y":         y,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  saved -> {args.out_dir}/", flush=True)


if __name__ == "__main__":
    main()
