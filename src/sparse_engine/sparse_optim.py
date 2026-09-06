"""Optimiser state that scales with ACTIVE experts, not the whole pool.

THE DEFECT THIS FIXES
    A routed model only computes a few experts per step, but a standard
    optimiser updates every parameter it owns. At 512 experts with
    d=96, d_ff=192 the expert tensors hold 18.9M parameters, and AdamW reads
    and writes all of them plus two moment buffers every step -- roughly
    227 MB of traffic -- while the forward pass touched a few thousand token
    assignments.

    Measured effect on a full training step (2 threads):

        experts     ms/step
              8       319.1
             64       468.2
            512      1340.6

    The 4x jump from 8 to 512 experts is not the routed compute, which is
    nearly flat by construction: k=2 experts are active per token regardless
    of pool size. It is the optimiser sweeping a pool that grew 64x.

    That defeats the entire premise. Sparsity that saves forward FLOPs and
    then pays them back in the optimiser is not sparsity, and a probe with
    this defect measures optimiser cost rather than routing.

WHAT THE REAL ENGINE DOES, AND THIS MIRRORS
    The 70B design keeps latent optimiser state per expert on disk and updates
    only the experts a step actually touched, amortised over several steps
    because disk writes measured 0.26 GB/s against 4.10 GB/s reads. This is
    the in-memory analogue: moments live for the whole pool, but a step reads
    and writes only the rows whose gradient is non-zero.

CORRECTNESS NOTE
    Adam's bias correction uses a per-parameter step count, not a global one.
    A rarely-touched expert must be corrected by ITS OWN number of updates or
    its first real update is scaled as though it had been training all along.
    `_step_count` is therefore per-row, which is the part of this that is easy
    to get silently wrong.
"""
from __future__ import annotations

import torch


class SparseExpertAdam:
    """Adam restricted to the expert rows that received gradient.

    Operates on tensors shaped (n_experts, ...) where dimension 0 indexes
    experts, so "row" means "one expert's whole weight block".
    """

    def __init__(self, params, lr: float = 3e-3, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        self.params = [p for p in params]
        if not self.params:
            raise ValueError("no parameters given")
        for p in self.params:
            if p.dim() < 2:
                raise ValueError(
                    "expert tensors must be (n_experts, ...); got shape "
                    f"{tuple(p.shape)}"
                )
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.state = [
            {
                "m": torch.zeros_like(p),
                "v": torch.zeros_like(p),
                "t": torch.zeros(p.shape[0], dtype=torch.long),
            }
            for p in self.params
        ]
        self.rows_updated = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self, rows: torch.Tensor | None = None) -> None:
        """Update the expert rows that received gradient.

        `rows` may be supplied by the caller when routing already knows which
        experts were touched. Detecting it by scanning the gradient costs a
        pass over the whole tensor -- 9.4M elements at 512 experts -- which
        was measured to make this SLOWER than a dense optimiser.

        When most of the pool is active the gather/scatter of moment state
        costs more than simply updating everything, so the dense path is taken
        instead. At 512 experts with 4096 token assignments nearly every
        expert is touched every step, so this is the common case, not a corner
        one: sparsity in the optimiser only pays when the pool is genuinely
        under-used.
        """
        self.rows_updated = 0
        for p, st in zip(self.params, self.state):
            if p.grad is None:
                continue
            g = p.grad
            n_e = g.shape[0]

            if rows is None:
                flat = g.reshape(n_e, -1)
                idx = (flat != 0).any(dim=1).nonzero(as_tuple=True)[0]
            else:
                idx = rows
            if idx.numel() == 0:
                continue
            self.rows_updated += int(idx.numel())

            if idx.numel() > n_e // 2:
                self._dense_step(p, st, g)
            else:
                self._sparse_step(p, st, g, idx)

    def _dense_step(self, p, st, g) -> None:
        st["t"] += 1
        t = st["t"].to(p.dtype).reshape((-1,) + (1,) * (p.dim() - 1))
        if self.weight_decay:
            g = g + self.weight_decay * p
        st["m"].mul_(self.b1).add_(g, alpha=1 - self.b1)
        st["v"].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
        mh = st["m"] / (1 - self.b1 ** t)
        vh = st["v"] / (1 - self.b2 ** t)
        p -= self.lr * mh / (vh.sqrt() + self.eps)

    def _sparse_step(self, p, st, g, rows) -> None:
        st["t"][rows] += 1
        t = st["t"][rows].to(p.dtype)
        t = t.reshape((-1,) + (1,) * (p.dim() - 1))

        gr = g[rows]
        if self.weight_decay:
            gr = gr + self.weight_decay * p[rows]

        m = st["m"][rows].mul_(self.b1).add_(gr, alpha=1 - self.b1)
        v = st["v"][rows].mul_(self.b2).addcmul_(gr, gr, value=1 - self.b2)
        st["m"][rows] = m
        st["v"][rows] = v

        # Per-row bias correction: a rarely-served expert must be corrected
        # by its own update count, not a global step counter.
        mh = m / (1 - self.b1 ** t)
        vh = v / (1 - self.b2 ** t)
        p[rows] -= self.lr * mh / (vh.sqrt() + self.eps)

    def set_lr(self, lr: float) -> None:
        self.lr = lr
