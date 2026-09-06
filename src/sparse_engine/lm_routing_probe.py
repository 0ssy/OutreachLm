"""Routing probe on REAL language, not a synthetic task.

Every routing result so far came from a constructed task: tokens carry a
latent concept id and the target is a concept-specific linear map. That tests
the right STRUCTURE -- can a router discover which specialist a token needs --
but it is a task built to be solvable by routing, so a positive result there
is weaker than it looks.

This runs the same comparisons on next-character prediction over the project's
own corpus (1002 files, 3.22 MB). Nothing about the data was chosen to reward
routing, so the questions become honest:

    A. Does a larger expert pool still help at fixed active count?
    B. What does sharding the pool across two machines actually cost?

Held constant across every arm, so only the pool varies:
    * active experts per token (k), hence compute per token
    * token budget, batch, learning rate, schedule, seeds
    * total training steps

Reported on HELD-OUT text, because a bigger pool has more parameters and
training loss would reward memorisation rather than routing.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

CORPUS = Path(r"C:\Users\josep\OneDrive\Desktop\OutreachLM")


def load_corpus(max_chars: int = 3_000_000) -> str:
    parts, total = [], 0
    for p in sorted(CORPUS.rglob("*.txt")):
        try:
            t = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        parts.append(t)
        total += len(t)
        if total >= max_chars:
            break
    return "".join(parts)[:max_chars]


class CharMoE(nn.Module):
    """Character LM whose feed-forward block is a top-k routed expert pool.

    Deliberately small and shallow: the question is whether pool size helps,
    and a deep model would confound that with depth.
    """

    def __init__(self, vocab: int, d: int, n_experts: int, k: int,
                 context: int, *, shards: int = 1, seed: int = 0,
                 residual: bool = True):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.d, self.k, self.n_experts = d, k, n_experts
        self.shards, self.context = shards, context
        # With residual=False the expert block is the only path from context
        # to prediction, so the probe measures ROUTING rather than the
        # embedding and head. Measured with the skip connection in place, the
        # expert block contributed only 0.0955 of 2.46 nats -- 3.9% -- which
        # made every pool-size comparison on this model uninformative.
        self.residual = residual
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(context, d))
        self.norm = nn.LayerNorm(d)
        self.router = nn.Linear(d, n_experts)
        d_ff = max(8, (4 * d) // max(1, int(n_experts ** 0.25)))
        self.d_ff = d_ff
        self.w_in = nn.Parameter(
            torch.randn(n_experts, d, d_ff, generator=g) / math.sqrt(d)
        )
        self.w_out = nn.Parameter(
            torch.randn(n_experts, d_ff, d, generator=g) / math.sqrt(d_ff)
        )
        self.head = nn.Linear(d, vocab)
        with torch.no_grad():
            self.router.weight.mul_(0.02)
            self.router.bias.zero_()

    def forward(self, idx: torch.Tensor, shard_id: torch.Tensor | None = None):
        B, T = idx.shape
        h = self.emb(idx) + self.pos[:T]
        # Causal mean-pool context: cheap, and identical across arms.
        cum = h.cumsum(1) / torch.arange(
            1, T + 1, device=h.device, dtype=h.dtype
        ).view(1, T, 1)
        h = self.norm(h + cum)
        flat = h.reshape(B * T, self.d)

        logits = self.router(flat)
        if self.shards > 1 and shard_id is not None:
            owner = torch.arange(
                self.n_experts, device=flat.device
            ) // (self.n_experts // self.shards)
            sid = shard_id.reshape(-1, 1)
            logits = logits.masked_fill(owner.unsqueeze(0) != sid,
                                        float("-inf"))
        topv, topi = torch.topk(logits, self.k, dim=-1)
        gate = F.softmax(topv, dim=-1)

        out = torch.zeros_like(flat)
        for j in range(self.k):
            e = topi[:, j]
            wi = self.w_in[e]                       # (N, d, d_ff)
            wo = self.w_out[e]                      # (N, d_ff, d)
            z = torch.bmm(flat.unsqueeze(1), wi).relu()
            out = out + gate[:, j:j + 1] * torch.bmm(z, wo).squeeze(1)

        probs = F.softmax(
            logits.masked_fill(torch.isinf(logits), -1e9), dim=-1
        )
        frac = torch.zeros(self.n_experts, device=flat.device)
        frac.scatter_add_(0, topi.reshape(-1),
                          torch.ones(topi.numel(), device=flat.device))
        bal = self.n_experts * (frac / max(1, topi.numel())
                                * probs.mean(0)).sum()
        mixed = (flat + out) if self.residual else out
        return self.head(mixed.reshape(B, T, self.d)), bal


def make_batches(data: torch.Tensor, batch: int, context: int,
                 gen: torch.Generator):
    ix = torch.randint(0, len(data) - context - 1, (batch,), generator=gen)
    x = torch.stack([data[i:i + context] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + context] for i in ix])
    return x, y, ix


def run(n_experts: int, *, k: int = 2, shards: int = 1, steps: int = 700,
        d: int = 96, context: int = 64, batch: int = 32, seed: int = 0,
        lr: float = 3e-3, bal_weight: float = 1e-2, residual: bool = True,
        data: tuple | None = None) -> float:
    train_d, val_d, vocab = data
    torch.manual_seed(seed)
    model = CharMoE(vocab, d, n_experts, k, context, shards=shards,
                    seed=seed, residual=residual)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    gen = torch.Generator().manual_seed(seed + 7)

    for _ in range(steps):
        x, y, ix = make_batches(train_d, batch, context, gen)
        sid = (ix % shards).repeat_interleave(context) if shards > 1 else None
        logits, bal = model(x, sid)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1)) \
            + bal_weight * bal
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    model.eval()
    tot, n = 0.0, 0
    vgen = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        for _ in range(24):
            x, y, ix = make_batches(val_d, batch, context, vgen)
            sid = (ix % shards).repeat_interleave(context) \
                if shards > 1 else None
            logits, _ = model(x, sid)
            tot += float(F.cross_entropy(
                logits.reshape(-1, vocab), y.reshape(-1)
            ))
            n += 1
    return tot / n
