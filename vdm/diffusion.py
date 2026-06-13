"""Gaussian diffusion (DDPM) with selectable prediction target, plus DDPM and DDIM
sampling. Schedules, q_sample, training loss and samplers are implemented from scratch.

Prediction target (`predict`):
  - "v"   : velocity (Salimans & Ho 2022). Well-conditioned at ALL noise levels —
            recovers x0 via a stable linear combo, with NO 1/sqrt(alpha_bar) blow-up.
            This is what makes generation-from-pure-noise actually work for small models.
  - "eps" : classic noise prediction (Ho 2020). x0 = (xt - sqrt(1-ab)*eps)/sqrt(ab)
            divides by sqrt(ab) ~ 0.006 at high t, amplifying tiny eps errors ~150x ->
            a small model cannot bootstrap sampling from pure noise (samples = noise).
  - "x0"  : predict the clean sample directly.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    acp = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    acp = acp / acp[0]
    betas = 1 - (acp[1:] / acp[:-1])
    return betas.clamp(1e-4, 0.999)


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    return torch.linspace(1e-4, 0.02, timesteps)


def _extract(a: torch.Tensor, t: torch.Tensor, shape) -> torch.Tensor:
    out = a.gather(0, t)
    return out.reshape(t.shape[0], *([1] * (len(shape) - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(self, model: nn.Module, timesteps: int = 1000, schedule: str = "cosine",
                 predict: str = "v"):
        super().__init__()
        assert predict in ("v", "eps", "x0"), f"unknown predict target: {predict}"
        self.model = model
        self.timesteps = timesteps
        self.predict = predict

        betas = cosine_beta_schedule(timesteps) if schedule == "cosine" else linear_beta_schedule(timesteps)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)
        acp_prev = F.pad(acp[:-1], (1, 0), value=1.0)

        def reg(name, val):
            self.register_buffer(name, val.float())

        reg("betas", betas)
        reg("acp", acp)
        reg("acp_prev", acp_prev)
        reg("sqrt_acp", torch.sqrt(acp))
        reg("sqrt_one_minus_acp", torch.sqrt(1.0 - acp))
        reg("sqrt_recip_acp", torch.sqrt(1.0 / acp))
        reg("sqrt_recipm1_acp", torch.sqrt(1.0 / acp - 1.0))
        post_var = betas * (1.0 - acp_prev) / (1.0 - acp)
        reg("post_log_var", torch.log(post_var.clamp(min=1e-20)))
        reg("post_mean_c1", betas * torch.sqrt(acp_prev) / (1.0 - acp))
        reg("post_mean_c2", (1.0 - acp_prev) * torch.sqrt(alphas) / (1.0 - acp))

    # ---- forward (training) ----
    def q_sample(self, x0, t, noise):
        return (_extract(self.sqrt_acp, t, x0.shape) * x0
                + _extract(self.sqrt_one_minus_acp, t, x0.shape) * noise)

    def _target(self, x0, noise, t):
        """The quantity the model is trained to predict, per `self.predict`."""
        if self.predict == "eps":
            return noise
        if self.predict == "x0":
            return x0
        # v = sqrt(ab)*eps - sqrt(1-ab)*x0
        return (_extract(self.sqrt_acp, t, x0.shape) * noise
                - _extract(self.sqrt_one_minus_acp, t, x0.shape) * x0)

    def p_losses(self, x0, t, y=None):
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred = self.model(xt, t, y)
        return F.mse_loss(pred, self._target(x0, noise, t))

    def _x0_from_out(self, xt, t, out):
        """Recover x0 from the model output, whatever the prediction target is.
        All branches are numerically stable except 'eps' at high t (kept for parity)."""
        if self.predict == "eps":
            return (_extract(self.sqrt_recip_acp, t, xt.shape) * xt
                    - _extract(self.sqrt_recipm1_acp, t, xt.shape) * out)
        if self.predict == "x0":
            return out
        # x0 = sqrt(ab)*xt - sqrt(1-ab)*v   (stable: no division by sqrt(ab))
        return (_extract(self.sqrt_acp, t, xt.shape) * xt
                - _extract(self.sqrt_one_minus_acp, t, xt.shape) * out)

    def predict_x0(self, xt, t, out):
        return self._x0_from_out(xt, t, out)

    def _eps_from_x0(self, xt, t, x0):
        """eps consistent with a (clamped) x0 estimate: eps = (xt - sqrt(ab)*x0)/sqrt(1-ab)."""
        return ((xt - _extract(self.sqrt_acp, t, xt.shape) * x0)
                / _extract(self.sqrt_one_minus_acp, t, xt.shape))

    # ---- reverse (sampling) ----
    @torch.no_grad()
    def p_sample(self, xt, t, y=None):
        out = self.model(xt, t, y)
        x0 = self._x0_from_out(xt, t, out).clamp(-1, 1)
        mean = _extract(self.post_mean_c1, t, xt.shape) * x0 + _extract(self.post_mean_c2, t, xt.shape) * xt
        if int(t[0]) == 0:
            return mean
        noise = torch.randn_like(xt)
        return mean + (0.5 * _extract(self.post_log_var, t, xt.shape)).exp() * noise

    @torch.no_grad()
    def sample(self, shape, y=None, device="cuda"):
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, y)
        return x

    @torch.no_grad()
    def ddim_sample(self, shape, y=None, steps=50, device="cuda"):
        """Deterministic DDIM (eta=0) — fewer steps than full DDPM."""
        seq = torch.linspace(0, self.timesteps - 1, steps).long().flip(0).tolist()
        x = torch.randn(shape, device=device)
        for idx, ti in enumerate(seq):
            t = torch.full((shape[0],), ti, device=device, dtype=torch.long)
            out = self.model(x, t, y)
            x0 = self._x0_from_out(x, t, out).clamp(-1, 1)
            eps = self._eps_from_x0(x, t, x0)             # re-derive eps from clamped x0
            t_next = seq[idx + 1] if idx + 1 < len(seq) else -1
            acp_next = self.acp[t_next] if t_next >= 0 else torch.tensor(1.0, device=device)
            x = acp_next.sqrt() * x0 + (1 - acp_next).sqrt() * eps
        return x
