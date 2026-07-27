"""Contextual dynamic-space GP for optuna's BoTorchSampler.

Fixes two defects of the stock sampler that both fail silently:

1. Mid-study suggest-range edits kill the GP. Optuna gates the GP through
   ``IntersectionSearchSpace``, which drops any parameter whose *distribution
   definition* ever differed across completed trials -- value containment is
   never checked. Editing a range therefore demotes that parameter to
   ``RandomSampler`` for the rest of the study, with nothing logged. Here the
   GP box is instead the bounding box of (history values union current range),
   so every completed trial keeps feeding the GP, while the acquisition is
   constrained to the *current* suggest region: width>0 params via
   ``optimize_acqf`` bounds, width-0 (fixed) params via ``fixed_features``.
   A parameter that is fixed now but was swept before stays in the GP as a
   context dimension, so observations from the other settings still transfer
   conditionally instead of being thrown away.

2. Parallel workers all receive the same proposal. A single-point acquisition
   is a deterministic function of the data, so workers asking before anyone
   finishes get the identical argmax and burn GPU time on near-duplicates.
   ``qLogExpectedImprovement`` with ``X_pending`` suppresses the acquisition
   around trials already in flight. Requires ``consider_running_trials=True``
   on the sampler, otherwise optuna never populates ``pending_x``.

Usage -- scripts with their own sampler subclass::

    class ConfigAwareBoTorchSampler(ContextualBoTorchMixin, BoTorchSampler):
        def sample_relative(self, study, trial, search_space):
            params = super().sample_relative(study, trial, search_space)
            ...                                   # existing jitter / config forcing
            return self._postprocess(params)      # must run last

Scripts with no subclass can use :class:`ContextualBoTorchSampler` directly.
Either way pass ``consider_running_trials=True`` when constructing it.

Verified against the live BERT-SQuAD journal (926 completed trials): all
completed trials reach the GP across three historical range edits, and with
6 running trials the proposal moves from 0.0142 to 0.8063 (normalised
distance) away from the nearest in-flight point.
"""

import torch
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from optuna.trial import TrialState
from optuna_integration import BoTorchSampler


class ContextualBoTorchMixin:
    """Mixin adding the dynamic-space contextual GP to a BoTorchSampler subclass.

    Must precede ``BoTorchSampler`` in the base list so its
    ``infer_relative_search_space`` wins over the stock one.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('candidates_func', self._contextual_candidates_func)
        super().__init__(*args, **kwargs)
        self._acq_ranges = {}    # name -> (low, high): current suggest range (width > 0)
        self._fixed_values = {}  # name -> value: fixed (width 0) in current code
        self._gp_space = {}      # name -> GP bounding-box distribution

    def infer_relative_search_space(self, study, trial):
        if self._study_id is None:
            self._study_id = study._study_id
        if self._study_id != study._study_id:
            raise RuntimeError("BoTorchSampler cannot handle multiple studies.")
        trials = [t for t in study.get_trials(deepcopy=False)
                  if t.state in (TrialState.COMPLETE, TrialState.RUNNING) and t.distributions]
        completed = [t for t in trials if t.state == TrialState.COMPLETE]
        if not completed:
            return {}
        newest = max(trials, key=lambda t: t.number)
        common = set(completed[0].params)
        for t in completed[1:]:
            common &= set(t.params)
        space, acq_ranges, fixed = {}, {}, {}
        for name in sorted(common & set(newest.distributions)):
            cur = newest.distributions[name]
            if isinstance(cur, CategoricalDistribution):
                continue  # keep categorical params on the independent path
            vals = [t.params[name] for t in completed]
            lo, hi = min(min(vals), cur.low), max(max(vals), cur.high)
            if lo == hi:
                continue  # true constant: never swept, nothing for the GP
            log = cur.log and lo > 0
            if isinstance(cur, IntDistribution):
                space[name] = IntDistribution(int(lo), int(hi), log=log)
            else:
                space[name] = FloatDistribution(float(lo), float(hi), log=log)
            if cur.single():
                fixed[name] = cur.low
            else:
                acq_ranges[name] = (cur.low, cur.high)
        self._gp_space, self._acq_ranges, self._fixed_values = space, acq_ranges, fixed
        return space

    def _contextual_candidates_func(self, train_x, train_obj, train_con, bounds, pending_x):
        from botorch.acquisition.logei import qLogExpectedImprovement
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.optim import optimize_acqf
        from botorch.sampling.normal import SobolQMCNormalSampler
        from botorch.utils.transforms import normalize, unnormalize
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from optuna._transform import _SearchSpaceTransform

        train_x = normalize(train_x, bounds=bounds)
        model = SingleTaskGP(train_x, train_obj,
                             outcome_transform=Standardize(m=train_obj.size(-1)))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        # In-flight (RUNNING) trials enter as X_pending: the acquisition marginalizes
        # over their unknown outcomes, so its argmax moves away from points other
        # workers are already evaluating. Without this every concurrent worker sees
        # the same data and gets the same argmax (herding -> near-duplicate trials).
        pending = None
        if pending_x is not None and len(pending_x):
            pending = normalize(pending_x, bounds=bounds)
        acqf = qLogExpectedImprovement(
            model=model, best_f=train_obj.max(), X_pending=pending,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])),
        )

        def to_unit(name, value, col):
            # map a raw value to the [0,1] coordinate of its GP-box column
            tval = float(_SearchSpaceTransform({name: self._gp_space[name]})
                         .transform({name: value})[0])
            lo, hi = float(bounds[0, col]), float(bounds[1, col])
            return min(1.0, max(0.0, (tval - lo) / (hi - lo)))

        acq_bounds = torch.zeros_like(bounds)
        acq_bounds[1] = 1.0
        fixed_features = {}
        for col, name in enumerate(self._gp_space):
            if name in self._fixed_values:
                fixed_features[col] = to_unit(name, self._fixed_values[name], col)
            else:
                a_lo, a_hi = self._acq_ranges[name]
                acq_bounds[0, col] = to_unit(name, a_lo, col)
                acq_bounds[1, col] = to_unit(name, a_hi, col)

        candidates, _ = optimize_acqf(
            acq_function=acqf, bounds=acq_bounds, q=1,
            num_restarts=10, raw_samples=512,
            fixed_features=fixed_features or None,
            options={"batch_limit": 5, "maxiter": 200},
        )
        print(f"[GP] contextual qLogEI: {train_x.size(0)} completed trials, "
              f"{train_x.size(1)}D, pending={0 if pending is None else len(pending)} "
              f"(fixed: {sorted(self._fixed_values)})", flush=True)
        return unnormalize(candidates.detach(), bounds=bounds)

    def _postprocess(self, params):
        """Final containment fixups. Call last, after any jitter the subclass applies."""
        # Snap width-0 (fixed) params to their exact value: the normalize ->
        # unnormalize round-trip drifts by ~1e-12, which would fail Trial._suggest's
        # containment check against the single-value distribution.
        for key, val in self._fixed_values.items():
            if key in params:
                params[key] = val
        # Clamp free params to the current suggest range. The GP box is generally
        # wider than what suggest() accepts, and jitter can overshoot; an
        # out-of-range value is silently demoted to independent (random) sampling.
        for key, (lo, hi) in self._acq_ranges.items():
            if key in params:
                params[key] = min(max(params[key], lo), hi)
        return params


class ContextualBoTorchSampler(ContextualBoTorchMixin, BoTorchSampler):
    """Drop-in BoTorchSampler with the contextual GP, for scripts with no subclass."""

    def sample_relative(self, study, trial, search_space):
        return self._postprocess(super().sample_relative(study, trial, search_space))
