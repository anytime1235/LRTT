"""Monkey-patch for true gradient accumulation with LRTT analog tiles.

Problem: AIHWKit's AnalogOptimizer.step() concatenates all micro-batch
inputs/grads via _pad_and_cat() before a single tile.update(), so the
update kernel always processes full-batch-sized tensors regardless of
gradient accumulation. This defeats memory savings.

Solution: Group entries by grad_accum step, concatenate weight-sharing
entries within each group, then process each group via direct
controller.ab_weight_update() (bypassing LRTTSimulatorTile.update()'s
_update_handled guard). Snapshot A0/B0 at step start for consistent
projections across micro-batch groups.

Handles ALBERT-style weight sharing correctly:
- Weight-sharing entries (same forward pass, multiple depths) are
  concatenated within each grad_accum group.
- Grad_accum entries (different forward passes) are processed separately
  for memory savings.

Usage:
    from aihwkit.optim import AnalogSGD, AnalogAdam
    import lrtt_grad_accum_patch  # noqa: F401  (applies patches on import)

    optimizer = AnalogSGD(model.parameters(), lr=0.01)
    optimizer._grad_accum_steps = GRAD_ACCUM  # must set for correct grouping
"""

import math as _math
import torch as _torch
from torch.autograd import no_grad as _no_grad

from aihwkit.optim.analog_optimizer import AnalogOptimizerMixin
from aihwkit.optim.context import AnalogContext as _AnalogContext
from aihwkit.simulator.tiles.lrtt_controller import LRTTController


# ── Patch 1: _ab_weight_update_lora with A/B snapshot ────────────────
_orig_ab_update = LRTTController._ab_weight_update_lora


def _ab_update_with_snapshot(self, x, d, lr, in_trans=False, out_trans=False):
    """LoRA update that uses snapshot A0/B0 for projections when available."""
    if not hasattr(self, '_snapshot_ab'):
        return _orig_ab_update(self, x, d, lr, in_trans, out_trans)

    # ── Snapshot mode ──
    if in_trans:
        x = x.t()
    if out_trans:
        d = d.t()

    A0, B0 = self._snapshot_ab
    with _torch.no_grad():
        XB = x @ B0.t()   # [batch, rank] using snapshot B0
        DA = d @ A0        # [batch, rank] using snapshot A0

    # lr_eff
    if self._is_hardware_mode():
        lr_eff = 1.0
    else:
        lr_eff = lr * self.lora_alpha
        if self.correct_gradient_magnitudes:
            lr_eff /= _math.sqrt(self.rank)

    # Debug logging
    self._ab_update_step += 1
    if self.log_ab_scaling and (self._ab_update_step % self.log_ab_scaling_every == 1):
        print(f"  [AB-Scaling Step {self._ab_update_step}] "
              f"A: x_max={XB.abs().max().item():.4f}, d_max={d.abs().max().item():.4f} | "
              f"B: x_max={x.abs().max().item():.4f}, d_max={DA.abs().max().item():.4f} | "
              f"lr_eff={lr_eff:.4f}")

    # ΔA = -lr_eff · D^T · XB
    lr_a_old = self.tile_a.get_learning_rate()
    self.tile_a.set_learning_rate(lr_eff)
    if hasattr(self.tile_a, '_orig_update'):
        self.tile_a._orig_update(XB, d)
    else:
        self.tile_a.update(XB, d)
    self.tile_a.set_learning_rate(lr_a_old)
    self.num_a_updates += 1

    # ΔB = -lr_eff · DA^T · X
    if not self._b_frozen:
        lr_b_old = self.tile_b.get_learning_rate()
        self.tile_b.set_learning_rate(lr_eff)
        if hasattr(self.tile_b, '_orig_update'):
            self.tile_b._orig_update(x, DA)
        else:
            self.tile_b.update(x, DA)
        self.tile_b.set_learning_rate(lr_b_old)
        self.num_b_updates += 1

    self._update_dynamic_te(lr)
    self.transfer_counter += (x.shape[0] if self.units_in_mbatch else 1)


LRTTController._ab_weight_update_lora = _ab_update_with_snapshot


# ── Patch 2: step() with per-group tile update ──────────────────────
@_no_grad()
def _step_mem_opt(self, closure=None, **kwargs):
    # Digital parameter update — skip AnalogOptimizerMixin in MRO
    # so the real optimizer (SGD, Adam, AdamW, etc.) step() is called.
    ret = super(AnalogOptimizerMixin, self).step(closure, **kwargs)

    # How many grad_accum forward passes were done before this step().
    # Must be set by the training script: optimizer._grad_accum_steps = K
    # Default 1 = no grad_accum (all entries from weight sharing).
    grad_accum_steps = getattr(self, '_grad_accum_steps', 1)

    # Analog parameter update
    for group in self.param_groups:
        learning_rate = group.get("lr")
        for param in group["params"]:
            if not isinstance(param, _AnalogContext):
                continue
            analog_ctx = param
            analog_tile = analog_ctx.analog_tile
            if analog_ctx.use_torch_update or not analog_ctx.has_gradient():
                continue
            if learning_rate == 0.0:
                analog_ctx.reset()
                continue
            if learning_rate is not None:
                analog_tile.set_learning_rate(learning_rate)

            runtime = analog_tile.get_runtime()
            n_micro = len(analog_ctx.analog_input)

            if n_micro <= 1:
                # Single batch: original path (no overhead)
                if analog_ctx.use_indexed:
                    for x_i, d_i in zip(
                        analog_ctx.analog_input, analog_ctx.analog_grad_output
                    ):
                        analog_tile.update_indexed(
                            x_i.to(analog_tile.device) if runtime.offload_input else x_i,
                            d_i.to(analog_tile.device) if runtime.offload_gradient else d_i,
                        )
                else:
                    x_input = self._pad_and_cat(
                        analog_ctx.analog_input,
                        axis=-1 if analog_tile.in_trans else 0,
                    )
                    d_input = self._pad_and_cat(
                        analog_ctx.analog_grad_output,
                        axis=-1 if analog_tile.out_trans else 0,
                    )
                    analog_tile.update(
                        x_input.to(analog_tile.device) if runtime.offload_input else x_input,
                        d_input.to(analog_tile.device) if runtime.offload_gradient else d_input,
                    )
            else:
                # Multiple entries: weight-sharing and/or grad_accum
                controller = getattr(analog_tile, 'controller', None)
                is_lrtt = controller is not None and hasattr(controller, 'tile_a')

                if is_lrtt:
                    # ── LRTT tile: bypass _update_handled, group correctly ──
                    #
                    # n_micro = entries_per_fwd * grad_accum_steps
                    #   entries_per_fwd: weight-sharing depth (ALBERT=12, MobileBERT=1)
                    #   grad_accum_steps: number of forward passes before step()
                    #
                    # Within each grad_accum group, concatenate weight-sharing
                    # entries (same forward pass). Process groups separately
                    # for memory savings.
                    entries_per_fwd = n_micro // grad_accum_steps

                    # Snapshot A0, B0 for consistent projections
                    A0 = controller.tile_a.get_weights()[0].clone()
                    B0 = controller.tile_b.get_weights()[0].clone()
                    controller._snapshot_ab = (A0, B0)

                    # Save state that _update_dynamic_te / transfer_counter
                    # modify inside ab_weight_update. We'll restore and
                    # recompute once after the loop so that a single optimizer
                    # step counts as exactly 1 logical update.
                    te_step_before = controller.te_step_counter
                    te_before = controller.transfer_every
                    counter_before = controller.transfer_counter

                    lr = analog_tile.get_learning_rate()

                    for g in range(grad_accum_steps):
                        start = g * entries_per_fwd
                        end = start + entries_per_fwd

                        # Concatenate weight-sharing entries within this group
                        x_group = self._pad_and_cat(
                            analog_ctx.analog_input[start:end],
                            axis=-1 if analog_tile.in_trans else 0,
                        )
                        d_group = self._pad_and_cat(
                            analog_ctx.analog_grad_output[start:end],
                            axis=-1 if analog_tile.out_trans else 0,
                        )

                        if runtime.offload_input:
                            x_group = x_group.to(analog_tile.device)
                        if runtime.offload_gradient:
                            d_group = d_group.to(analog_tile.device)

                        # Direct controller call — bypasses _update_handled
                        # and should_transfer() inside LRTTSimulatorTile.update()
                        controller.ab_weight_update(
                            x=x_group, d=d_group, lr=lr,
                            in_trans=False, out_trans=False,
                        )

                    # ── Fix counters ──
                    # ab_weight_update called _update_dynamic_te N times,
                    # advancing te_step_counter by N and potentially setting
                    # transfer_every based on inflated counter. Restore and
                    # apply a single correct increment.
                    controller.te_step_counter = te_step_before
                    controller.transfer_every = te_before
                    controller._update_dynamic_te(lr)  # advances by +1

                    # transfer_counter:
                    # - units_in_mbatch=True: sum of group batch_sizes across
                    #   the loop is correct (equals total batch over all depths).
                    # - units_in_mbatch=False: each call added +1, should be +1
                    #   per optimizer step.
                    if not controller.units_in_mbatch:
                        controller.transfer_counter = counter_before + 1

                    # Check transfer once after all groups processed
                    if controller.should_transfer():
                        controller.ab_weight_transfer()

                    del controller._snapshot_ab

                else:
                    # Non-LRTT tile: per-micro-batch update (no _update_handled issue)
                    if analog_ctx.use_indexed:
                        for x_i, d_i in zip(
                            analog_ctx.analog_input, analog_ctx.analog_grad_output
                        ):
                            analog_tile.update_indexed(
                                x_i.to(analog_tile.device) if runtime.offload_input else x_i,
                                d_i.to(analog_tile.device) if runtime.offload_gradient else d_i,
                            )
                    else:
                        for x_i, d_i in zip(
                            analog_ctx.analog_input, analog_ctx.analog_grad_output
                        ):
                            analog_tile.update(
                                x_i.to(analog_tile.device) if runtime.offload_input else x_i,
                                d_i.to(analog_tile.device) if runtime.offload_gradient else d_i,
                            )

            analog_ctx.reset()

    # Post-update step (diffuse, decay, etc.)
    for group in self.param_groups:
        for param in group["params"]:
            if isinstance(param, _AnalogContext):
                param.analog_tile.post_update_step()
    return ret


AnalogOptimizerMixin.step = _step_mem_opt
