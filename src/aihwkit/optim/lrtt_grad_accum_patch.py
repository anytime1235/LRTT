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
    import aihwkit.optim.lrtt_grad_accum_patch  # noqa: F401  (applies patches on import)

    optimizer = AnalogSGD(model.parameters(), lr=0.01)
    optimizer._grad_accum_steps = GRAD_ACCUM  # must set for correct grouping
"""

import math as _math
import torch as _torch
from torch import cat as _cat
from torch.nn.functional import pad as _F_pad
from torch.autograd import no_grad as _no_grad

from aihwkit.optim.analog_optimizer import AnalogOptimizerMixin
from aihwkit.optim.context import AnalogContext as _AnalogContext
from aihwkit.simulator.tiles.lrtt_controller import LRTTController


# ── Patch 0: _pad_and_cat for dynamic-padding grad accumulation ──────
def _pad_and_cat(tensors, axis):
    """Cat tensors, zero-padding non-cat dimensions if shapes differ.

    When using dynamic padding with gradient accumulation, accumulated
    inputs may have different sequence lengths. Zero-padding before cat
    is safe because padded positions contribute zero to the update.
    """
    if len(tensors) <= 1:
        return _cat(tensors, axis=axis)

    ndim = tensors[0].dim()
    # Find max size for each dimension
    max_sizes = list(tensors[0].shape)
    needs_pad = False
    for t in tensors[1:]:
        for d in range(ndim):
            if d != (axis % ndim) and t.shape[d] != max_sizes[d]:
                needs_pad = True
            if t.shape[d] > max_sizes[d]:
                max_sizes[d] = t.shape[d]

    if not needs_pad:
        return _cat(tensors, axis=axis)

    cat_dim = axis % ndim
    padded = []
    for t in tensors:
        # _F_pad format: (last_dim_left, last_dim_right, ..., first_dim_left, first_dim_right)
        pad_sizes = []
        for d in range(ndim - 1, -1, -1):
            if d == cat_dim:
                pad_sizes.extend([0, 0])
            else:
                pad_sizes.extend([0, max_sizes[d] - t.shape[d]])
        padded.append(_F_pad(t, pad_sizes) if any(pad_sizes) else t)
    return _cat(padded, axis=axis)


AnalogOptimizerMixin._pad_and_cat = staticmethod(_pad_and_cat)


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
    m_batch = x.shape[0]
    self._last_m_batch = m_batch
    self.transfer_counter += m_batch


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
                # Detect LRTT sub-tiles via back-reference set in
                # LRTTSimulatorTile._hook_tile_updates()
                controller = getattr(analog_tile, '_lrtt_controller', None)
                is_lrtt = controller is not None
                tile_name = getattr(analog_tile, '_lrtt_tile_name', None)

                if is_lrtt and not controller.forward_inject_enabled:
                    # ── LRTT NFI mode ──
                    # In NFI mode, only tile_c's update triggers
                    # ab_weight_update; tile_a/tile_b hooks are no-ops.
                    if tile_name != 'tile_c':
                        # tile_a/tile_b: skip (no-op in NFI mode)
                        analog_ctx.reset()
                        continue

                    # tile_c: bypass _update_handled, group correctly
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
                    _dev = analog_tile.device
                    A0 = controller.tile_a.get_weights()[0].clone().to(_dev)
                    B0 = controller.tile_b.get_weights()[0].clone().to(_dev)
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
                    # Counter always += m_batch per call (TikiTaka convention).
                    # Loop called ab_weight_update N times with depth-inflated
                    # entries_per_fwd groups. Correct to 1 batch worth.
                    # _last_m_batch must equal the corrected increment for the
                    # modulo boundary-crossing check in should_transfer().
                    inflated = controller.transfer_counter - counter_before
                    corrected_increment = inflated // entries_per_fwd
                    controller.transfer_counter = counter_before + corrected_increment
                    controller._last_m_batch = corrected_increment

                    # Check transfer once after all groups processed
                    if controller.should_transfer():
                        controller.ab_weight_transfer()

                    del controller._snapshot_ab

                elif is_lrtt and controller.forward_inject_enabled:
                    # ── LRTT FI mode ──
                    # FI hooks pair tile_a + tile_b updates: tile_a.update()
                    # caches data, tile_b.update() triggers both _orig_update.
                    # Must process tile_a and tile_b together per group.
                    entries_per_fwd = n_micro // grad_accum_steps

                    if tile_name == 'tile_c':
                        # tile_c: frozen, only handle transfer counter once.
                        lr = analog_tile.get_learning_rate()
                        controller._last_lr_sgd = lr  # sync for effective_alpha
                        m_batch = analog_ctx.analog_input[0].shape[0]
                        controller._update_dynamic_te(lr)
                        corrected_increment = m_batch * grad_accum_steps
                        controller._last_m_batch = corrected_increment
                        controller.transfer_counter += corrected_increment
                        if controller.should_transfer():
                            controller.ab_weight_transfer()
                    elif tile_name == 'tile_a':
                        # Defer: save for paired processing with tile_b
                        controller._fi_ga_a_tile = analog_tile
                        controller._fi_ga_a_ctx = analog_ctx
                        controller._fi_ga_a_runtime = runtime
                        continue  # skip analog_ctx.reset() — tile_b will handle it
                    elif tile_name == 'tile_b':
                        # Process tile_a and tile_b together, group by group.
                        # This ensures the FI hook pairs matching groups.
                        a_tile = controller._fi_ga_a_tile
                        a_ctx = controller._fi_ga_a_ctx
                        a_runtime = controller._fi_ga_a_runtime

                        for g in range(grad_accum_steps):
                            start = g * entries_per_fwd
                            end = start + entries_per_fwd

                            # tile_a group
                            x_a = self._pad_and_cat(
                                a_ctx.analog_input[start:end],
                                axis=-1 if a_tile.in_trans else 0,
                            )
                            d_a = self._pad_and_cat(
                                a_ctx.analog_grad_output[start:end],
                                axis=-1 if a_tile.out_trans else 0,
                            )
                            if a_runtime.offload_input:
                                x_a = x_a.to(a_tile.device)
                            if a_runtime.offload_gradient:
                                d_a = d_a.to(a_tile.device)

                            # tile_b group
                            x_b = self._pad_and_cat(
                                analog_ctx.analog_input[start:end],
                                axis=-1 if analog_tile.in_trans else 0,
                            )
                            d_b = self._pad_and_cat(
                                analog_ctx.analog_grad_output[start:end],
                                axis=-1 if analog_tile.out_trans else 0,
                            )
                            if runtime.offload_input:
                                x_b = x_b.to(analog_tile.device)
                            if runtime.offload_gradient:
                                d_b = d_b.to(analog_tile.device)

                            # Call in order: tile_a then tile_b
                            # FI hook caches _fi_a_*, then _fi_b_* triggers _orig_update
                            a_tile.update(x_a, d_a)
                            analog_tile.update(x_b, d_b)

                        # Reset deferred tile_a context
                        a_ctx.reset()
                        del controller._fi_ga_a_tile, controller._fi_ga_a_ctx, controller._fi_ga_a_runtime

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
