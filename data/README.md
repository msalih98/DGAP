# Data directory

Place the ASVspoof 2019 LA partition here, so that the paths resolve to:

```
data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{train.trn,dev.trl,eval.trl}.txt
data/LA/ASVspoof2019_LA_{train,dev,eval}/flac/
```

`prepare_spec_data.py` writes the diffusion training windows to
`data/spec_train_256/`.
