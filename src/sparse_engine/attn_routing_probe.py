"""Language routing probe with a context-carrying trunk and fast dispatch.

TWO DEFECTS IN THE PREVIOUS PROBE, BOTH FIXED HERE.

1. THE TRUNK FLATTENED CONTEXT.
   The old trunk mean-pooled the prefix, so a token's representation was
   almost independent of what surrounded it. A router fed that cannot tell
   "river bank" from "investment bank", and cannot discover structure the
   representation does not represent. That is the most likely explanation for
   the null result: 8, 64 and 512 experts scored 2.4724 / 2.4736 / 2.4729,
   indistinguishable, while on a synthetic task where routing is the
   bottleneck by construction the same comparison spread 3.8x.

   Replaced with causal LINEAR attention. Using the associativity
   phi(Q) (phi(K)^T V) with a causal cumulative sum costs O(T d^2) rather
   than the O(T^2 d) of softmax attention, which matters because this has to
   run on a CPU. Every token now carries a context-dependent signature, which
   is the thing the router is supposed to route on.

2. DISPATCH MATERIALISED A TENSOR PER TOKEN.
   `w_in[expert_of_token]` builds an (N, d, d_ff) tensor -- 178 MB at N=2048,
   d_ff=227. Measured at the probe's own shapes: forward bmm 54 ms, the gather
   feeding it 188.7 ms, and the backward scatter 409.5 ms. Arithmetic was ~10%
   of the step, and five runs took 3.2 hours against a 20-minute estimate.

   That is the same FLOPs-vs-wall-clock divergence this project has now
   measured three times, and it is the access pattern the engine design
   explicitly rejects. Replaced with per-expert dispatch: tokens are sorted by
   assigned expert and each expert runs ONE GEMM over its own contiguous
   block, so the weights are read once and never replicated per token.

   `test_dispatch_paths_agree` asserts the two produce identical output, so
   the speedup cannot come from a behaviour change.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def quadratic_causal_attention(q: torch.Tensor, k: torch.Tensor,
                               v: torch.Tensor) -> torch.Tensor:
    """Ordinary masked softmax attention. The right choice at these shapes.

    Linear attention is only cheaper when T > d. Its cost is O(T d^2) against
    O(T^2 d) for the quadratic form, so the crossover is exactly T = d. This
    probe runs T=64 with d=96, i.e. on the wrong side of it:

        linear   O(T d^2) = 589,824   988.3 ms   75.50 MB state
        quad     O(T^2 d) = 393,216     7.2 ms    0.52 MB attention

    A 137x wall-clock difference and 145x memory difference, in favour of the
    form usually described as the expensive one. The linear version's O(T d^2)
    FLOP advantage never existed here, and its (B, T, d, d) prefix-sum tensor
    is retained by autograd for the backward, which is where the time went.

    Keep `linear_causal_attention` for long contexts (T >> d); use this below
    the crossover.
    """
    t = q.shape[1]
    scale = q.shape[-1] ** -0.5
    att = (q @ k.transpose(1, 2)) * scale
    mask = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
    att = att.masked_fill(~mask, float("-inf")).softmax(dim=-1)
    return att @ v


def linear_causal_attention(q: torch.Tensor, k: torch.Tensor,
                            v: torch.Tensor) -> torch.Tensor:
    """Causal attention in O(T d^2) via prefix sums of outer products.

    phi = elu + 1 keeps the feature map positive so the denominator cannot
    change sign, which is what makes the normalisation stable.

    ONLY use this when T > d. Below that crossover it is slower and far more
    memory-hungry than the quadratic form -- see quadratic_causal_attention.
    It materialises a (B, T, d, d) tensor, which is 75 MB at B=32, T=64, d=96
    and is retained through the backward pass.
    """
    q = F.elu(q) + 1.0
    k = F.elu(k) + 1.0
    kv = torch.einsum("btd,bte->btde", k, v).cumsum(dim=1)
    num = torch.einsum("btd,btde->bte", q, kv)
    den = torch.einsum("btd,btd->bt", q, k.cumsum(dim=1)).clamp_min(1e-6)
    return num / den.unsqueeze(-1)


def dispatch_batched(x: torch.Tensor, expert_idx: torch.Tensor,
                     gate: torch.Tensor, w_in: torch.Tensor,
                     w_out: torch.Tensor) -> torch.Tensor:
    """One batched GEMM over all experts at once, via padded blocks.

    The two obvious dispatch strategies each fail at one end of the range,
    and both failures were measured at the probe's shapes (N=2048, d=96,
    d_ff=192, forward+backward):

        experts   per-token    per-expert loop
              8     566.9 ms          12.5 ms      loop wins 45x
            512     812.3 ms       23127.8 ms      loop loses 25x

    Per-token replicates the weights for every token -- a 151 MB tensor per
    step regardless of pool size. Per-expert avoids that but degenerates into
    one tiny GEMM per expert: at 512 experts each holds ~8 tokens, so the
    Python loop and its 512 index_add_ calls dominate completely. That is the
    same fragmentation that made Method K2 lose 3.9x on wall-clock despite
    0.05x the FLOPs.

    This pads each expert's token block to a common length and issues a
    single bmm over the whole pool: weights are read once, the work is one
    contiguous operation, and there is no per-expert Python. Padding costs
    E * max_count * d floats, which stays small precisely because the
    load-balancing loss keeps max_count near the mean.
    """
    n, d = x.shape
    k = expert_idx.shape[1]
    n_e, _, d_ff = w_in.shape

    flat_e = expert_idx.reshape(-1)
    flat_g = gate.reshape(-1)
    flat_tok = torch.arange(n, device=x.device).repeat_interleave(k)

    order = torch.argsort(flat_e)
    e_sorted = flat_e[order]
    tok_sorted = flat_tok[order]
    g_sorted = flat_g[order]

    counts = torch.bincount(e_sorted, minlength=n_e)
    max_c = int(counts.max()) if n_e else 0
    if max_c == 0:
        return torch.zeros_like(x)

    # Position of each assignment within its own expert's block.
    starts = torch.cumsum(counts, 0) - counts
    within = torch.arange(e_sorted.numel(), device=x.device) - starts[e_sorted]

    buf = torch.zeros(n_e, max_c, d, dtype=x.dtype, device=x.device)
    buf[e_sorted, within] = x[tok_sorted]

    h = torch.bmm(buf, w_in).relu()
    y = torch.bmm(h, w_out)

    vals = y[e_sorted, within] * g_sorted.unsqueeze(1)
    out = torch.zeros_like(x)
    out.index_add_(0, tok_sorted, vals)
    return out


def dispatch_per_expert(x: torch.Tensor, expert_idx: torch.Tensor,
                        gate: torch.Tensor, w_in: torch.Tensor,
                        w_out: torch.Tensor) -> torch.Tensor:
    """One GEMM per expert. Fast for small pools, catastrophic for large ones
    (25x slower than per-token at 512 experts). Kept as a reference path."""
    n, d = x.shape
    k = expert_idx.shape[1]
    out = torch.zeros_like(x)
    flat_e = expert_idx.reshape(-1)
    flat_g = gate.reshape(-1)
    flat_tok = torch.arange(n, device=x.device).repeat_interleave(k)

    order = torch.argsort(flat_e)
    e_sorted = flat_e[order]
    tok_sorted = flat_tok[order]
    g_sorted = flat_g[order]

    uniq, counts = torch.unique_consecutive(e_sorted, return_counts=True)
    start = 0
    for e, c in zip(uniq.tolist(), counts.tolist()):
        sl = slice(start, start + c)
        start += c
        toks = tok_sorted[sl]
        h = (x[toks] @ w_in[e]).relu() @ w_out[e]
        out.index_add_(0, toks, g_sorted[sl].unsqueeze(1) * h)
    return out


def dispatch_per_token(x: torch.Tensor, expert_idx: torch.Tensor,
                       gate: torch.Tensor, w_in: torch.Tensor,
                       w_out: torch.Tensor) -> torch.Tensor:
    """Reference implementation. Correct, and the reason the probe was slow."""
    out = torch.zeros_like(x)
    for j in range(expert_idx.shape[1]):
        e = expert_idx[:, j]
        h = torch.bmm(x.unsqueeze(1), w_in[e]).relu()
        out = out + gate[:, j:j + 1] * torch.bmm(h, w_out[e]).squeeze(1)
    return out


class AttnMoE(nn.Module):
    """Character LM: linear-attention trunk, then a routed expert pool."""

    def __init__(self, vocab: int, d: int, n_experts: int, k: int,
                 context: int, *, shards: int = 1, seed: int = 0,
                 d_ff: int | None = None, fast: bool = True,
                 attention: str = "auto"):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.d, self.k, self.n_experts = d, k, n_experts
        self.shards, self.context, self.fast = shards, context, fast
        # Crossover is T = d: linear attention is O(T d^2), quadratic is
        # O(T^2 d). Picking the wrong side cost 137x wall-clock and 145x
        # memory when this probe was first written.
        if attention == "auto":
            attention = "linear" if context > d else "quadratic"
        if attention not in ("linear", "quadratic"):
            raise ValueError(f"unknown attention: {attention}")
        self.attention = attention

        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(context, d))
        self.qkv = nn.Linear(d, 3 * d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.router = nn.Linear(d, n_experts)

        # Expert width is held FIXED across pool sizes. Scaling it with the
        # pool (as the previous probe did) makes small-pool arms far more
        # expensive and confounds routing with per-expert capacity.
        self.d_ff = d_ff if d_ff is not None else 2 * d
        self.w_in = nn.Parameter(
            torch.randn(n_experts, d, self.d_ff, generator=g) / math.sqrt(d)
        )
        self.w_out = nn.Parameter(
            torch.randn(n_experts, self.d_ff, d, generator=g)
            / math.sqrt(self.d_ff)
        )
        self.head = nn.Linear(d, vocab)
        with torch.no_grad():
            self.router.weight.mul_(0.02)
            self.router.bias.zero_()

    def forward(self, idx: torch.Tensor, shard_id: torch.Tensor | None = None):
        B, T = idx.shape
        h = self.emb(idx) + self.pos[:T]
        q, k, v = self.qkv(self.norm1(h)).chunk(3, dim=-1)
        attn = (linear_causal_attention if self.attention == "linear"
                else quadratic_causal_attention)
        h = h + attn(q, k, v)
        flat = self.norm2(h).reshape(B * T, self.d)

        logits = self.router(flat)
        if self.shards > 1 and shard_id is not None:
            owner = torch.arange(
                self.n_experts, device=flat.device
            ) // (self.n_experts // self.shards)
            # shard_id may be given per SEQUENCE (B,) or per TOKEN (B*T,).
            # Silently broadcasting the wrong one would assign tokens to the
            # wrong shard rather than raise, so expand explicitly.
            sid = shard_id.reshape(-1)
            if sid.numel() == B:
                sid = sid.repeat_interleave(T)
            elif sid.numel() != B * T:
                raise ValueError(
                    f"shard_id must have {B} or {B * T} entries, "
                    f"got {sid.numel()}"
                )
            logits = logits.masked_fill(
                owner.unsqueeze(0) != sid.unsqueeze(1), float("-inf")
            )
        topv, topi = torch.topk(logits, self.k, dim=-1)
        gate = F.softmax(topv, dim=-1)

        fn = dispatch_batched if self.fast else dispatch_per_token
        out = fn(flat, topi, gate, self.w_in, self.w_out)

        probs = F.softmax(
            logits.masked_fill(torch.isinf(logits), -1e9), dim=-1
        )
        frac = torch.zeros(self.n_experts, device=flat.device)
        frac.scatter_add_(0, topi.reshape(-1),
                          torch.ones(topi.numel(), device=flat.device))
        bal = self.n_experts * (frac / max(1, topi.numel())
                                * probs.mean(0)).sum()
        return self.head(out.reshape(B, T, self.d)), bal
