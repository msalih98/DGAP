# This prepares the spectrogram training data for the diffusion model.

import os

from tqdm import tqdm

from utils import REPO_ROOT, read_wav, wav2spec, spec_to_windows, spec_to_img

LA_ROOT   = os.path.join(REPO_ROOT, "data", "LA")
PROTOCOL  = os.path.join(LA_ROOT, "ASVspoof2019_LA_cm_protocols",
                         "ASVspoof2019.LA.cm.train.trn.txt")
AUDIO_DIR = os.path.join(LA_ROOT, "ASVspoof2019_LA_train", "flac")
OUT_DIR   = os.path.join(REPO_ROOT, "data", "spec_train_256")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(PROTOCOL) as f:
        entries = [(line.split()[1], line.split()[4]) for line in f]
    print(f"{len(entries)} training utterances")

    for utt_id, label in tqdm(entries):
        path = os.path.join(AUDIO_DIR, utt_id + ".flac")
        if not os.path.exists(path):
            continue
        wav = read_wav(path)
        S, _ = wav2spec(wav)
        for i, win in enumerate(spec_to_windows(S)):
            spec_to_img(win).save(
                os.path.join(OUT_DIR, f"{utt_id}_{label}_w{i:02d}.png"))


if __name__ == "__main__":
    main()
