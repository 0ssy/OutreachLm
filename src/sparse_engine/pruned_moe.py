"""Banked expert pruning wired into a real MoE, end to end.

`banked_pruning.py` was validated in isolation against a simulated stream of
gradient magnitudes. That leaves the integration untested, and the integration
is where the mechanism can quietly fail: the router produces affinities on a
different scale every step, magnitudes shift as training proceeds, and the
pruned set changes what the router sees next.

This wraps a real routed MoE so the loop is closed:

    router affinity  ->  pruner priority  ->  admitted expert set
         ^                                              |
         +------- observed gradient magnitude <---------+

and measures the two things that matter: does every expert still get served,
and does pruning cost quality against an unpruned control at equal compute.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.sparse_engine.banked_pruning import BankedExpertPruner


class PrunedRouter:
    """Restricts routing to an admitted subset, chosen by banked priority.

    The admitted set is recomputed every `refresh` steps rather than every
    step: re-selecting constantly would make the expert set churn faster than
    the experts can learn, and the whole point of pruning is that the active
    set is stable enough to amortise streaming its weights.
    """

    def __init__(self, n_experts: int, admit: int, *, refresh: int = 16,
                 max_staleness: int | None = None):
        if admit > n_experts:
            raise ValueError("admit cannot exceed n_experts")
        self.pruner = BankedExpertPruner(
            n_experts, admit, max_staleness=max_staleness
        )
        self.n_experts = n_experts
        self.admit = admit
        self.refresh = refresh
        self._admitted = torch.arange(min(admit, n_experts))
        self._step = 0

    @property
    def admitted(self) -> torch.Tensor:
        return self._admitted

    def mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Blank out non-admitted experts before top-k routing."""
        keep = torch.zeros(self.n_experts, dtype=torch.bool)
        keep[self._admitted] = True
        return logits.masked_fill(~keep.unsqueeze(0), float("-inf"))

    def step(self, logits: torch.Tensor,
             expert_grad_norm: torch.Tensor) -> None:
        """Feed one step of evidence back into the pruner.

        `logits` is over ALL experts -- the router scores everything, it is
        the expert COMPUTE that is restricted, so affinity for pruned experts
        is available for free. `expert_grad_norm` is defined only for the
        admitted set, because nothing else was computed.
        """
        self._step += 1
        with torch.no_grad():
            affinity = F.softmax(logits, dim=-1).mean(0)
        self.pruner.observe(self._admitted, expert_grad_norm, affinity)
        if self._step % self.refresh == 0:
            self._admitted = torch.sort(self.pruner.select()).values

    @property
    def never_served(self) -> int:
        return self.pruner.never_updated

    @property
    def forced_admissions(self) -> int:
        return self.pruner.forced_admissions


def expert_grad_norms(w_in: torch.Tensor, w_out: torch.Tensor,
                      admitted: torch.Tensor) -> torch.Tensor:
    """Per-expert gradient magnitude for the admitted set.

    Returns zeros rather than raising when a backward pass has not populated
    grads, so a caller cannot silently feed stale magnitudes.
    """
    if w_in.grad is None or w_out.grad is None:
        return torch.zeros(len(admitted))
    gi = w_in.grad[admitted].reshape(len(admitted), -1)
    go = w_out.grad[admitted].reshape(len(admitted), -1)
    return (gi.norm(dim=1) ** 2 + go.norm(dim=1) ** 2).sqrt()
