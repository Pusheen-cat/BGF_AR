"""BGF: a Balanced Gradient Filter that pre-conditions gradients for AdamW.

Before every AdamW update, each parameter tensor's gradient ``g`` is blended with
a norm-matched exponential moving average (EMA) of past gradients -- the
"low-frequency" gradient component:

    que <- (1 - lambda) * que + lambda * g                 # EMA  (init: first g)
    g'  <- alpha * g + (1 - alpha) * (||g|| / ||que||) * que

``g'`` is then handed to the standard AdamW step, so **BGF = grad-adjust + AdamW**.

All norms are Frobenius / L2 norms computed *per parameter tensor* (one scalar per
"layer"), which keeps the overhead to a single norm pair per tensor.

Hyperparameters
    bgf_lambda : EMA weight (default 0.01). Smaller -> slower-moving low-freq term.
    bgf_alpha  : weight on the raw gradient (default 0.9); (1 - alpha) on the
                 norm-matched EMA direction.
    bgf_eps    : epsilon added to the ``||que||`` denominator of the layer-wise
                 norm ratio (default 1e-8). Guards divide-by-(near-)zero when a
                 tensor's EMA gradient is tiny.
    bgf_scale_max : hard clamp on the norm ratio (default 1e6). The added term
                 ``scale * que`` has norm ``||g||`` by construction, so the ratio
                 is only unbounded when ``||que||`` underflows; clamping avoids a
                 large*tiny overflow producing Inf.

NaN robustness
    Recurrent models (e.g. the DeepSEA Stack-/Tape-RNN over 1000 steps) can emit a
    non-finite gradient on a single batch (backward/BPTT overflow; note that
    ``clip_grad_norm_`` turns an Inf grad into NaN via ``inf * 0``). Two guards
    keep one bad batch from permanently poisoning training:
      1. If ANY parameter has a non-finite gradient this step, the AdamW update is
         SKIPPED and the EMA ``que`` is NOT updated -- params and the low-frequency
         buffer keep their last finite values (AMP-GradScaler-style skip). The
         count is exposed as ``self.bgf_num_skipped``.
      2. In the finite path, the norm ratio is eps-guarded and clamped, and ``g'``
         is sanitized as a last resort, so the normalization itself cannot
         introduce Inf/NaN.

On the very first step ``que`` is seeded with the first gradient, which makes the
scale 1 and ``g' == g`` -- i.e. BGF starts as plain AdamW and warms in.

The DeepSEA pipeline keeps a verbatim copy at
``DNA/deepsea_sequence_models/src/training/optim.py``; keep the two in sync.
"""
from __future__ import annotations

import torch


class BGF(torch.optim.AdamW):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, bgf_lambda=0.01, bgf_alpha=0.9,
                 bgf_eps=1e-8, bgf_scale_max=1e6, **kwargs):
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay, **kwargs)

        if not 0.0 <= bgf_lambda <= 1.0:
            raise ValueError(f"bgf_lambda must be in [0, 1], got {bgf_lambda}")
        if not 0.0 <= bgf_alpha <= 1.0:
            raise ValueError(f"bgf_alpha must be in [0, 1], got {bgf_alpha}")
        self.bgf_lambda = float(bgf_lambda)
        self.bgf_alpha = float(bgf_alpha)
        self.bgf_eps = float(bgf_eps)
        self.bgf_scale_max = float(bgf_scale_max)
        # EMA buffers kept *outside* AdamW's per-param state, so AdamW still
        # initializes its own exp_avg / exp_avg_sq on the first step.
        self._bgf_que: dict = {}
        self.bgf_num_skipped = 0  # steps skipped due to non-finite gradients

    @torch.no_grad()
    def _grads_all_finite(self) -> bool:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    return False
        return True

    @torch.no_grad()
    def _adjust_gradients(self) -> None:
        lam, alpha = self.bgf_lambda, self.bgf_alpha
        eps, scale_max = self.bgf_eps, self.bgf_scale_max
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                que = self._bgf_que.get(p)
                if que is None:
                    # seed the EMA with the first gradient; g' == g this step.
                    self._bgf_que[p] = g.clone()
                    continue
                que.mul_(1.0 - lam).add_(g, alpha=lam)          # EMA update (in place)
                # layer-wise Frobenius ratio: eps in the denominator guards a tiny
                # ||que||; clamp guards a large/tiny overflow to Inf.
                scale = g.norm() / (que.norm() + eps)
                scale = torch.clamp(scale, max=scale_max)
                # g' = alpha * g + (1 - alpha) * scale * que
                new_g = alpha * g + ((1.0 - alpha) * scale) * que
                # last-resort sanitiser: fall back to the (finite) raw gradient if
                # the blend somehow produced a non-finite value.
                if not torch.isfinite(new_g).all():
                    new_g = g
                p.grad = new_g

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Skip-on-non-finite: do not update params and do not poison the EMA que
        # if any gradient is Inf/NaN this step (recover on the next good batch).
        if not self._grads_all_finite():
            self.bgf_num_skipped += 1
            return loss
        self._adjust_gradients()
        super().step()  # AdamW update on the adjusted gradients
        return loss


def build_optimizer(params, *, name="adamw", lr, weight_decay,
                    betas=(0.9, 0.999), eps=1e-8,
                    bgf_lambda=0.01, bgf_alpha=0.9,
                    bgf_eps=1e-8, bgf_scale_max=1e6):
    """Construct ``adamw`` or ``bgf`` (BGF + AdamW) from a config name."""
    name = (name or "adamw").lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay,
                                 betas=betas, eps=eps)
    if name in ("bgf", "bgf_adamw"):
        return BGF(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps,
                   bgf_lambda=bgf_lambda, bgf_alpha=bgf_alpha,
                   bgf_eps=bgf_eps, bgf_scale_max=bgf_scale_max)
    raise ValueError(f"Unknown optimizer {name!r}; choose 'adamw' or 'bgf'")
