# Model directory

Place the fine-tuned diffusion purifier here:

```
models/model300000.pt
```

The detector weights are not stored here; they ship inside `aasist/`, `rawgat_st/`
and `tssdnet/` together with their official repositories.

If you fine-tune the purifier yourself, also place the public 256x256
guided-diffusion checkpoint here as `models/256x256_diffusion_uncond.pt` and pass it
to `--resume_checkpoint`.
