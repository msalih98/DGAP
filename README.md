# DGAP, an adaptive diffusion-based defense approach for audio deepfake detection

Code for the paper *Detection-Guided Adaptive Purification with Diffusion Models
for Robust Audio Deepfake Detection*, accepted at the 4th EAI International
Conference on Security and Privacy in Cyber-Physical Systems and Smart Vehicles
(EAI SmartSP 2026).

**Authors:** Muhammed Salih Kayhan, Qiben Yan (Michigan State University)

A preprint of the paper is available at: https://arxiv.org/abs/2610.10752

---
## 1. Setup

### Conda environment

```bash
conda create -n dgap python=3.10
conda activate dgap
```

### Dependencies

```bash
pip install -r requirements.txt
```

Tested with Python 3.10 and CUDA 12.4. `mpi4py` and `blobfile` are needed only if
you fine-tune the diffusion model yourself.

---
## 2. Dataset

### Where to get it

Download the Logical Access partition from
[Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/3336).

### Where it goes

Unpack it under `data/` so that the protocols and audio resolve to:

```
data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{train.trn,dev.trl,eval.trl}.txt
data/LA/ASVspoof2019_LA_{train,dev,eval}/flac/
```

---
## 3. Purifier checkpoint

The purifier is a guided-diffusion model fine-tuned on 256x256 spectrogram windows
drawn from both bonafide and spoofed training utterances.

### Pretrained

The fine-tuned checkpoint is available
[here](https://drive.google.com/file/d/1ojRdPaEAO1_A7zuMin-8-ATbbwS6RZnj/view?usp=sharing);
save it as `models/model300000.pt`.

### Or fine-tune it yourself

Build the spectrogram windows first, then resume from the public 256x256
guided-diffusion checkpoint:

```bash
python prepare_spec_data.py

python -m guided_diffusion.scripts.image_train \
    --data_dir data/spec_train_256 \
    --resume_checkpoint models/256x256_diffusion_uncond.pt \
    --image_size 256 --num_channels 256 --num_res_blocks 2 \
    --attention_resolutions 32,16,8 --num_head_channels 64 \
    --resblock_updown True --use_scale_shift_norm True --learn_sigma True \
    --diffusion_steps 1000 --noise_schedule linear --lr 1e-4 --batch_size 1
```

---
## 4. Detectors

The three detectors are included as unmodified copies of their official
repositories, together with the pretrained weights those repositories release, so
nothing needs to be downloaded for them.

| Directory | Official repository | Version | License |
|---|---|---|---|
| `aasist/` | https://github.com/clovaai/aasist | commit `a04c986` | MIT (NAVER Corp.) |
| `rawgat_st/` | https://github.com/eurecom-asp/RawGAT-ST-antispoofing | `main` | MIT (EURECOM) |
| `tssdnet/` | https://github.com/ghua-ac/end-to-end-synthetic-speech-detection | `main` | GPL-3.0 |

`guided_diffusion/` is likewise taken from
https://github.com/openai/guided-diffusion (MIT, see
`guided_diffusion/LICENSE_GUIDED_DIFFUSION`).

Every script takes the same `--detector` flag, which accepts `aasist`, `rawgat` or
`res_tssdnet`. Detector-specific conventions are handled in `inference.py`.

---
## 5. Generating adversarial examples

Each run writes `bona/`, `clean_spoof/` and `adv/` directories plus an
`attack_summary.json` with the attack success rate.

```bash
# PGD, l2 (eps = 1.0) and linf (eps = 0.005)
python attacks/pgd_l2.py   --detector aasist --split eval --steps 10 --out_root out/aasist/pgd_l2
python attacks/pgd_linf.py --detector aasist --split eval --steps 20 --out_root out/aasist/linf

# Carlini-Wagner (kappa = 10, c = 1.0)
python attacks/cw.py       --detector aasist --split eval --out_root out/aasist/cw
```

The number of PGD iterations is tuned per detector to reach a comparable success
rate: 10/20 for AASIST, 50/100 for RawGAT-ST, 10/20 for Res-TSSDNet (`l2`/`linf`).

To keep the utterance sets identical across detectors, pass the same
`--spoof_keys_file` and `--bona_keys_file` to every run instead of relying on random
sampling.

---
## 6. Running the defense

`configs/dgap_{detector}.yaml` holds the attack-agnostic configuration reported in
the paper; `configs/ablations/{detector}_uniform.yaml` is the ablation with the gate
disabled (uniform purification at a fixed level, equivalent to DiffPure with the
same purifier).

Both benign and adversarial utterances pass through the defense before scoring, so
run it on all three directories produced by the attack:

```bash
for split in bona clean_spoof adv; do
  python purify.py --detector aasist --config configs/dgap_aasist.yaml \
      --input out/aasist/pgd_l2/$split \
      --out_json results/aasist_pgd_l2/$split.json \
      --save_purified_dir results/aasist_pgd_l2/$split
done
```

Then compute the metrics:

```bash
python evaluate.py --detector aasist \
    --bona_dir  results/aasist_pgd_l2/bona \
    --clean_dir results/aasist_pgd_l2/clean_spoof \
    --adv_dir   results/aasist_pgd_l2/adv \
    --out_dir   results/aasist_pgd_l2/eval
```

This reports `EER_clean`, `EER_adv` and the objective
`y = EER_clean + lambda * EER_adv`.

---
## 7. Defense-aware attack

`attacks/bpda_eot.py` runs a defense-aware adversary: the gate decision sits inside
the attack loop, the defense is approximated as the identity in the backward pass
(BPDA), and the gradient is averaged over several stochastic runs of the pipeline
(EOT). It reports `EER_adv` together with the fraction of benign and adversarial
inputs the gate flags under attack.

```bash
python attacks/bpda_eot.py --detector aasist --config configs/dgap_aasist.yaml \
    --split eval --eps 0.005 --steps 20 --eot 8 --n_spoof 120 --n_bona 120 \
    --out_root out/aasist/bpda_eot
```

Every EOT sample runs the full purification pipeline, so the cost grows with
`--steps` times `--eot`. Passing an ablation config attacks the uniform defense
instead, which reduces to plain BPDA since no gate is involved.

---
## 8. Parameter selection

The shipped configurations were selected on the development split. First craft the
adversarial examples on that split, using the same attack scripts with
`--split dev` and `--out_root out/aasist_dev/{attack}`. Then extract the score gaps
over the diffusion-level grid and calibrate:

```bash
for atk in pgd_l2 cw linf; do
  python probe_sweep.py --detector aasist --dev_root out/aasist_dev/$atk \
      --bona_sub bona --clean_sub clean_spoof --adv_sub adv --attack_name $atk \
      --t_high_list 20,40,60,80,100,120,140,160,180,200,220,240,260,280,300 \
      --N 500 --out_json dev/aasist/features_dev_${atk}.json
done

python calibrate.py --results_dir dev/aasist
```

`calibrate.py` expects the three files to be named `features_dev_{attack}.json` in
one directory, and writes its own `calibration.json` next to them.

`calibrate.py` sweeps the probe level `t_g`, the purification level `t_p` and the
gate coefficient `gamma` in [0.5, 3.0], and reports both the per-attack optima and
the single attack-agnostic configuration that minimizes the objective averaged over
the three attacks.

---
## Repository layout

```
purify.py              defense runner (probe -> gate -> purify)
inference.py           detector wrappers (loading, padding, scoring)
diffusion_model.py     reverse VP-SDE purifier
utils.py               audio/spectrogram helpers and the EER metric
probe_sweep.py         development-split score gaps over the t grid
calibrate.py           parameter selection
evaluate.py            EER_clean / EER_adv / y on the evaluation split
prepare_spec_data.py   spectrogram windows for diffusion fine-tuning
attacks/               pgd_l2.py, pgd_linf.py, cw.py, bpda_eot.py
configs/               attack-agnostic configurations and the uniform ablation
data/                  dataset goes here (see data/README.md)
models/                diffusion checkpoint goes here (see models/README.md)
```

Output directories are chosen on the command line and are not tracked by git. The
commands above use `out/` for adversarial audio, `results/` for purified audio and
metrics, and `dev/` for the calibration features.

---
## Citation

If you use this code, please cite:

```bibtex
@inproceedings{kayhan2026dgap,
  title     = {Detection-Guided Adaptive Purification with Diffusion Models
               for Robust Audio Deepfake Detection},
  author    = {Kayhan, Muhammed Salih and Yan, Qiben},
  booktitle = {Proceedings of the 4th EAI International Conference on Security and
               Privacy in Cyber-Physical Systems and Smart Vehicles (SmartSP)},
  year      = {2026},
  note      = {To appear}
}
```

---
## License

The code in this repository is released under the MIT License (see `LICENSE`).
The third-party directories listed in Section 4 keep their own licenses; in
particular, `tssdnet/` is distributed under GPL-3.0.
