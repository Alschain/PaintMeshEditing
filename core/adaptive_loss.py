import math
import torch


class AdaptiveLossBalancer:
    """Automatic multi-loss weight balancer for PaintMeshEditing.

    Supports four strategies:
      - 'fixed': Keep base weights constant.
      - 'ema_scale': Magnitude-calibrated EMA weighting with smooth regularization decay
        so differently scaled losses (e.g. area vs image L1) stay balanced automatically.
      - 'grad_norm': Periodically measures ||d(w_i * L_i) / dV||_2 w.r.t. mesh vertices
        and smoothly adjusts weights toward balanced gradient contributions.
      - 'dwa': Dynamic Weight Average (Liu et al.) adjusting weights from relative loss
        decrease rates so slower-converging terms receive higher priority.
    """

    DATA_LOSS_KEYS = {
        'img_loss',
        'img_l2_loss',
        'ssim_loss',
        'vgg_loss',
        'normal_loss',
        'depth_loss',
        'silhouette_loss',
    }

    def __init__(
        self,
        base_weights,
        mode='fixed',
        adapt_interval=10,
        ema_momentum=0.9,
        min_scale=0.1,
        max_scale=10.0,
        temperature=2.0,
        decay_rate=0.0005,
        min_decay_factor=0.15,
    ):
        self.base_weights = {k: float(v) for k, v in base_weights.items()}
        self.weights = dict(self.base_weights)
        self.mode = (mode or 'fixed').lower()
        self.adapt_interval = max(1, int(adapt_interval))
        self.ema_momentum = float(ema_momentum)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.temperature = float(temperature)
        self.decay_rate = float(decay_rate)
        self.min_decay_factor = float(min_decay_factor)

        self._initial_scales = {}
        self._ema_losses = {}
        self._prev_losses = {}
        self._prev2_losses = {}

    def get_weights(self):
        return dict(self.weights)

    def _active_keys(self, losses):
        return [k for k in losses.keys() if self.base_weights.get(k, 0.0) > 0.0]

    def _clamp_weight(self, key, value):
        base = self.base_weights.get(key, 0.0)
        if base <= 0.0:
            return 0.0
        lo = base * self.min_scale
        hi = base * self.max_scale
        return float(max(lo, min(hi, value)))

    def update(self, iteration, losses, params=None):
        """Update and return active loss weights for the current iteration."""
        active = self._active_keys(losses)
        if not active or self.mode == 'fixed':
            return self.get_weights()

        raw_vals = {}
        for k in active:
            v = losses[k]
            val = float(v.detach().item()) if isinstance(v, torch.Tensor) else float(v)
            if math.isfinite(val):
                raw_vals[k] = max(val, 1e-12)

        if self.mode == 'ema_scale':
            self._update_ema_scale(iteration, active, raw_vals)
        elif self.mode == 'dwa':
            self._update_dwa(iteration, active, raw_vals)
        elif self.mode == 'grad_norm':
            self._update_grad_norm(iteration, active, losses, params)

        return self.get_weights()

    def _update_ema_scale(self, iteration, active, raw_vals):
        # Compute normalized target shares from base_weights
        weighted_init = {}
        for k in active:
            if k not in raw_vals:
                continue
            val = raw_vals[k]
            if k not in self._ema_losses:
                self._ema_losses[k] = val
            else:
                self._ema_losses[k] = self.ema_momentum * self._ema_losses[k] + (1.0 - self.ema_momentum) * val

            if iteration == 0 or k not in self._initial_scales:
                # Calibrate around the initial weighted magnitude so no term starts 1000x off
                self._initial_scales[k] = max(val, 1e-8)

            weighted_init[k] = self.base_weights[k] * self._initial_scales[k]

        if not weighted_init:
            return

        # Median weighted contribution serves as anchor scale
        sorted_vals = sorted(weighted_init.values())
        anchor = sorted_vals[len(sorted_vals) // 2]
        if anchor <= 1e-12:
            return

        for k in active:
            if k not in self._ema_losses:
                continue
            base = self.base_weights[k]
            cur_weighted = base * self._ema_losses[k]
            if cur_weighted <= 1e-12:
                continue

            # Softly pull imbalanced terms toward the anchor magnitude
            ratio = (anchor / cur_weighted) ** 0.35
            target_w = base * ratio

            # Gently decay regularization over time so fine details fit cleanly
            if k not in self.DATA_LOSS_KEYS:
                decay = self.min_decay_factor + (1.0 - self.min_decay_factor) * (10.0 ** (-iteration * self.decay_rate))
                target_w = target_w * decay

            target_w = self._clamp_weight(k, target_w)
            self.weights[k] = self.ema_momentum * self.weights[k] + (1.0 - self.ema_momentum) * target_w

    def _update_dwa(self, iteration, active, raw_vals):
        # Update 2-step history via smoothed EMA losses
        for k in active:
            if k not in raw_vals:
                continue
            val = raw_vals[k]
            if k not in self._ema_losses:
                self._ema_losses[k] = val
            else:
                self._ema_losses[k] = 0.7 * self._ema_losses[k] + 0.3 * val

        if iteration % self.adapt_interval != 0:
            return

        if len(self._prev_losses) > 0 and len(self._prev2_losses) > 0:
            rates = {}
            for k in active:
                if k in self._prev_losses and k in self._prev2_losses:
                    denom = max(self._prev2_losses[k], 1e-10)
                    rates[k] = min(max(self._prev_losses[k] / denom, 0.1), 10.0)

            if len(rates) > 1:
                T = max(self.temperature, 1e-3)
                exps = {k: math.exp(r / T) for k, r in rates.items()}
                exp_sum = sum(exps.values())
                n = len(rates)
                for k, e_val in exps.items():
                    factor = n * e_val / exp_sum
                    target_w = self._clamp_weight(k, self.base_weights[k] * factor)
                    self.weights[k] = self.ema_momentum * self.weights[k] + (1.0 - self.ema_momentum) * target_w

        self._prev2_losses = dict(self._prev_losses)
        self._prev_losses = {k: self._ema_losses[k] for k in active if k in self._ema_losses}

    def _update_grad_norm(self, iteration, active, losses, params):
        if params is None or (iteration % self.adapt_interval) != 0:
            return
        if not isinstance(params, torch.Tensor) or not params.requires_grad:
            return

        grad_norms = {}
        for k in active:
            loss_tensor = losses[k]
            if not isinstance(loss_tensor, torch.Tensor) or not loss_tensor.requires_grad:
                continue
            w = self.weights.get(k, 0.0)
            if w <= 0.0:
                continue
            try:
                g = torch.autograd.grad(w * loss_tensor, params, retain_graph=True, allow_unused=True)[0]
                if g is not None:
                    gn = float(g.detach().norm().item())
                    if math.isfinite(gn) and gn > 1e-12:
                        grad_norms[k] = gn
            except RuntimeError:
                continue

        if len(grad_norms) < 2:
            return

        mean_gn = sum(grad_norms.values()) / len(grad_norms)
        for k, gn in grad_norms.items():
            # Nudge weight so gradient norms move smoothly toward mean gradient norm
            ratio = (mean_gn / gn) ** 0.5
            target_w = self._clamp_weight(k, self.weights[k] * ratio)
            self.weights[k] = self.ema_momentum * self.weights[k] + (1.0 - self.ema_momentum) * target_w
