# This calibrates the gate and purification levels on the development split.

import argparse
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from utils import compute_eer

ATTACKS = ["pgd_l2", "cw", "linf"]
ALL_GATE_T = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300]
GAMMA_GRID = np.round(np.arange(0.5, 3.005, 0.01), 2).tolist()


def eer(pos, neg):
    return compute_eer(pos, neg)[0]


def auc_pos_neg(pos, neg):
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = np.r_[pos, neg]
    return float(roc_auc_score(y, s))


def class_arrays(data, t_list):
    out = {}
    for lab in ("bona", "clean_spoof", "adv"):
        recs = data["samples"][lab]
        out[lab] = {
            "s_low":  np.array([r["s_low"] for r in recs]),
            "s_high": {t: np.array([r[f"s_high_{t}"] for r in recs]) for t in t_list},
            "d":      {t: np.array([r[f"d_{t}"] for r in recs]) for t in t_list},
        }
    return out


def analyze_attack(data):
    cfg = data["config"]
    t_list = cfg["t_high_list"]
    gate_t_candidates = [t for t in ALL_GATE_T if t in t_list]

    arr = class_arrays(data, t_list)
    bo, cs, ad = arr["bona"], arr["clean_spoof"], arr["adv"]

    eer_c_nd = eer(bo["s_low"], cs["s_low"])
    eer_d_nd = eer(bo["s_low"], ad["s_low"])

    # gate statistics (mu, sigma) from benign inputs (bona + clean_spoof)
    gate_stats = {}
    for gate_t in gate_t_candidates:
        cd = np.r_[bo["d"][gate_t], cs["d"][gate_t]]
        gate_stats[gate_t] = {
            "mu":    float(cd.mean()),
            "sigma": float(cd.std()),
            "auc":   auc_pos_neg(ad["d"][gate_t], cd),
        }

    # per-t_score: uniform baseline + DGAP (gate_t x gamma) grid
    per_t = []
    for t_score in t_list:
        eer_c_uni = eer(bo["s_high"][t_score], cs["s_high"][t_score])
        eer_d_uni = eer(bo["s_high"][t_score], ad["s_high"][t_score])

        dgap_grid = []
        for gate_t in gate_t_candidates:
            mu_g = gate_stats[gate_t]["mu"]
            sg_g = gate_stats[gate_t]["sigma"]
            for gamma in GAMMA_GRID:
                tau = mu_g + gamma * sg_g
                sbo = np.where(bo["d"][gate_t] > tau, bo["s_high"][t_score], bo["s_low"])
                scs = np.where(cs["d"][gate_t] > tau, cs["s_high"][t_score], cs["s_low"])
                sad = np.where(ad["d"][gate_t] > tau, ad["s_high"][t_score], ad["s_low"])
                ec, ed = eer(sbo, scs), eer(sbo, sad)
                dgap_grid.append({
                    "gate_t": gate_t, "gamma": float(gamma), "tau": float(tau),
                    "eer_c": ec, "eer_adv": ed, "y": ec + ed,
                })

        per_t.append({
            "t_score":       t_score,
            "uniform_eer_c": eer_c_uni,
            "uniform_eer_adv": eer_d_uni,
            "uniform_y":     eer_c_uni + eer_d_uni,
            "dgap":          dgap_grid,
        })

    return {
        "config":            cfg,
        "gate_t_candidates": gate_t_candidates,
        "gate_stats":        gate_stats,
        "no_defense":        {"eer_c": eer_c_nd, "eer_adv": eer_d_nd, "y": eer_c_nd + eer_d_nd},
        "per_t":             per_t,
    }


def find_optimum(per_t):
    best = {"y": float("inf")}
    for row in per_t:
        for g in row["dgap"]:
            if g["y"] < best["y"]:
                best = {"y": g["y"], "eer_c": g["eer_c"], "eer_adv": g["eer_adv"],
                        "t_score": row["t_score"], "gate_t": g["gate_t"],
                        "gamma": g["gamma"], "tau": g["tau"]}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True,
                    help="directory holding features_dev_{attack}.json")
    args = ap.parse_args()
    results_dir = args.results_dir
    print(f"[calibrate] results_dir={results_dir}", flush=True)

    results = {}
    for atk in ATTACKS:
        path = os.path.join(results_dir, f"features_dev_{atk}.json")
        data = json.load(open(path))
        n_total = sum(len(data["samples"][lab]) for lab in data["samples"])
        print(f"loaded {atk}: {n_total} samples", flush=True)

        results[atk] = analyze_attack(data)
        results[atk]["optimum"] = find_optimum(results[atk]["per_t"])

        nd = results[atk]["no_defense"]
        op = results[atk]["optimum"]
        best_uni = min(results[atk]["per_t"], key=lambda r: r["uniform_y"])
        per_atk_gates = results[atk]["gate_t_candidates"]
        best_auc_t = max(per_atk_gates,
                         key=lambda t: results[atk]["gate_stats"][t]["auc"])
        best_auc = results[atk]["gate_stats"][best_auc_t]["auc"]
        print(f"\n=== {atk} ===", flush=True)
        print(f"  no-defense        y={nd['y']*100:6.2f}%  (c={nd['eer_c']*100:.2f}, adv={nd['eer_adv']*100:.2f})",
              flush=True)
        print(f"  best uniform      t_score={best_uni['t_score']:3d}                                  "
              f"y={best_uni['uniform_y']*100:6.2f}%  "
              f"(c={best_uni['uniform_eer_c']*100:.2f}, adv={best_uni['uniform_eer_adv']*100:.2f})",
              flush=True)
        print(f"  best DGAP         gate_t={op['gate_t']:3d}  t_score={op['t_score']:3d}  "
              f"gamma={op['gamma']:.2f}  tau={op['tau']:+.4f}  y={op['y']*100:6.2f}%  "
              f"(c={op['eer_c']*100:.2f}, adv={op['eer_adv']*100:.2f})", flush=True)
        print(f"  best detection    gate_t={best_auc_t:3d}  AUC={best_auc:.4f}", flush=True)

    # attack-agnostic optimum: a single (gate_t, t_score, gamma) shared by all attacks,
    # searched over the intersection of the attacks' t grids
    print(f"\n=== Attack-agnostic optimum (single config for all attacks) ===", flush=True)
    common_t = set(results[ATTACKS[0]]["config"]["t_high_list"])
    for atk in ATTACKS[1:]:
        common_t &= set(results[atk]["config"]["t_high_list"])
    common_t = sorted(common_t)
    print(f"  common t values across attacks: {common_t}", flush=True)
    common_gate_t = [t for t in ALL_GATE_T if t in common_t]
    print(f"  attack-agnostic gate_t candidates: {common_gate_t}", flush=True)

    best_agn = {"avg_y": float("inf")}
    for gate_t in common_gate_t:
        for t_score in common_t:
            for gamma in GAMMA_GRID:
                per_y = []
                tau_per_atk = {}
                for atk in ATTACKS:
                    g = results[atk]["per_t"]
                    row = next(r for r in g if r["t_score"] == t_score)
                    cell = next(c for c in row["dgap"] if c["gate_t"] == gate_t and c["gamma"] == gamma)
                    per_y.append(cell["y"])
                    tau_per_atk[atk] = cell["tau"]
                avg = float(np.mean(per_y))
                if avg < best_agn["avg_y"]:
                    best_agn = {"avg_y": avg, "gate_t": gate_t, "t_score": t_score, "gamma": float(gamma),
                                 "per_y": per_y, "tau_per_atk": tau_per_atk}
    bu = best_agn
    print(f"  gate_t={bu['gate_t']}  t_score={bu['t_score']}  gamma={bu['gamma']:.2f}  "
          f"avg y={bu['avg_y']*100:.2f}%", flush=True)
    for atk, y in zip(ATTACKS, bu["per_y"]):
        print(f"    {atk:<10}  y={y*100:.2f}%  tau={bu['tau_per_atk'][atk]:+.4f}", flush=True)

    out_path = os.path.join(results_dir, "calibration.json")
    out = {
        "attacks":           ATTACKS,
        "all_gate_t":        ALL_GATE_T,
        "agnostic_gate_t":   common_gate_t,
        "agnostic_t_score":  common_t,
        "gamma_grid":        GAMMA_GRID,
        "per_attack":        results,
        "agnostic":          bu,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
