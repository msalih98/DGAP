# This is the diffusion model used for purification.

from types import SimpleNamespace

import numpy as np
import torch
import torchsde

from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

DEFAULT_CKPT = "models/model300000.pt"


def _model_config(use_fp16):
    cfg = model_and_diffusion_defaults()
    cfg.update({
        "image_size":            256,
        "num_channels":          256,
        "num_res_blocks":        2,
        "num_head_channels":     64,
        "attention_resolutions": "32,16,8",
        "resblock_updown":       True,
        "use_fp16":              use_fp16,
        "use_scale_shift_norm":  True,
        "class_cond":            False,
        "diffusion_steps":       1000,
        "noise_schedule":        "linear",
        "learn_sigma":           True,
    })
    return cfg


def window_to_tensor(window):
    arr = np.stack([window, window, window], axis=0)
    return torch.from_numpy(arr).float().unsqueeze(0) * 2.0 - 1.0


def tensor_to_window(tensor):
    x = tensor.detach().float().clamp(-1.0, 1.0)
    return ((x + 1.0) * 0.5)[0].mean(dim=0).cpu().numpy()


def _extract_into_tensor(arr_or_func, timesteps, broadcast_shape):
    if callable(arr_or_func):
        res = arr_or_func(timesteps).float()
    else:
        res = arr_or_func.to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


class RevVPSDE(torch.nn.Module):
    def __init__(self, model, beta_min=0.1, beta_max=20, N=1000,
                 img_shape=(3, 256, 256), model_kwargs=None):
        super().__init__()
        self.model = model
        self.model_kwargs = model_kwargs
        self.img_shape = img_shape

        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.alphas_cumprod_cont = lambda t: torch.exp(-0.5 * (beta_max - beta_min) * t**2 - beta_min * t)
        self.sqrt_1m_alphas_cumprod_neg_recip_cont = lambda t: -1. / torch.sqrt(1. - self.alphas_cumprod_cont(t))

        self.noise_type = "diagonal"
        self.sde_type = "ito"

    def _scale_timesteps(self, t):
        assert torch.all(t <= 1) and torch.all(t >= 0), f't has to be in [0, 1], but get {t} with shape {t.shape}'
        return (t.float() * self.N).long()

    def vpsde_fn(self, t, x):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def rvpsde_fn(self, t, x, return_type='drift'):
        drift, diffusion = self.vpsde_fn(t, x)

        if return_type == 'drift':
            assert x.ndim == 2 and np.prod(self.img_shape) == x.shape[1], x.shape
            x_img = x.view(-1, *self.img_shape)

            if self.model_kwargs is None:
                self.model_kwargs = {}

            # model predicts epsilon; with learn_sigma the output packs (mean, var)
            disc_steps = self._scale_timesteps(t)
            model_output = self.model(x_img, disc_steps, **self.model_kwargs)
            model_output, _ = torch.split(model_output, self.img_shape[0], dim=1)
            assert x_img.shape == model_output.shape, f'{x_img.shape}, {model_output.shape}'
            model_output = model_output.view(x.shape[0], -1)
            score = _extract_into_tensor(self.sqrt_1m_alphas_cumprod_neg_recip_cont, t, x.shape) * model_output

            drift = drift - diffusion[:, None] ** 2 * score
            return drift

        return diffusion

    def f(self, t, x):
        # drift of the reverse SDE via the t' = 1 - t substitution
        t = t.expand(x.shape[0])
        drift = self.rvpsde_fn(1 - t, x, return_type='drift')
        assert drift.shape == x.shape
        return -drift

    def g(self, t, x):
        t = t.expand(x.shape[0])
        diffusion = self.rvpsde_fn(1 - t, x, return_type='diffusion')
        assert diffusion.shape == (x.shape[0], )
        return diffusion[:, None].expand(x.shape)


class RevGuidedDiffusion(torch.nn.Module):
    def __init__(self, args, device=None):
        super().__init__()
        self.args = args
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        img_shape = (3, 256, 256)
        fp16 = bool(getattr(args, 'fp16', True) and self.device.type == "cuda")
        model_config = _model_config(fp16)
        model, _ = create_model_and_diffusion(**model_config)
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
        if model_config['use_fp16']:
            model.convert_to_fp16()
        model.eval().to(self.device)

        self.model = model
        self.rev_vpsde = RevVPSDE(model=model, img_shape=img_shape,
                                  model_kwargs=None).to(self.device)
        self.betas = self.rev_vpsde.discrete_betas.float().to(self.device)

    def image_editing_sample(self, img):
        assert isinstance(img, torch.Tensor)
        batch_size = img.shape[0]
        state_size = int(np.prod(img.shape[1:]))

        assert img.ndim == 4, img.ndim
        img = img.to(self.device)
        x0 = img

        xs = []
        for it in range(self.args.sample_step):
            e = torch.randn_like(x0).to(self.device)
            total_noise_levels = self.args.t
            if self.args.rand_t:
                total_noise_levels = self.args.t + np.random.randint(-self.args.t_delta, self.args.t_delta)
            a = (1 - self.betas).cumprod(dim=0).to(self.device)
            x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()

            epsilon_dt0, epsilon_dt1 = 0, 1e-5
            t0, t1 = 1 - self.args.t * 1. / 1000 + epsilon_dt0, 1 - epsilon_dt1
            t_size = 2
            ts = torch.linspace(t0, t1, t_size).to(self.device)

            x_ = x.view(batch_size, -1)
            if self.args.use_bm:
                bm = torchsde.BrownianInterval(t0=t0, t1=t1, size=(batch_size, state_size), device=self.device)
                xs_ = torchsde.sdeint_adjoint(self.rev_vpsde, x_, ts, method='euler', bm=bm)
            else:
                xs_ = torchsde.sdeint_adjoint(self.rev_vpsde, x_, ts, method='euler')
            x0 = xs_[-1].view(x.shape)

            xs.append(x0)

        return torch.cat(xs, dim=0)


def make_args(t, checkpoint=DEFAULT_CKPT, sample_step=1, use_bm=False,
              rand_t=False, t_delta=10, fp16=True):
    return SimpleNamespace(
        t=t, sample_step=sample_step, use_bm=use_bm, rand_t=rand_t, t_delta=t_delta,
        checkpoint=checkpoint, fp16=fp16,
    )
