"""Faster routed dispatch: closing the gap to an isolated GEMM.

Routed dispatch measures 78 GFLOP/s where an isolated GEMM on the same
machine reaches 173-190. profile_dispatch.py attributed the gap to two
things, both measured rather than assumed:

    gather + scatter + unscatter   28% of the forward pass
    padding waste                  16-36%, because every expert's block is
                                   padded to the largest expert's count

This module tests replacements for both. Correctness is not negotiable: each
variant must produce values identical to the reference within fp tolerance,
so any speed difference is attributable to memory access rather than to doing
less work.
"""
from __future__ import annotations

import time

import torch

torch.set_num_threads(6)


def dispatch_current(x, expert_idx, gate, w_in, w_out):
    """The shipped version, for reference."""
    n, d = x.shape
    k = expert_idx.shape[1]
    n_e = w_in.shape[0]
    flat_e = expert_idx.reshape(-1)
    flat_g = gate.reshape(-1)
    flat_tok = torch.arange(n, device=x.device).repeat_interleave(k)

    order = torch.argsort(flat_e)
    e_sorted = flat_e[order]
    tok_sorted = flat_tok[order]
    g_sorted = flat_g[order]

    counts = torch.bincount(e_sorted, minlength=n_e)
    max_c = int(counts.max())
    starts = torch.cumsum(counts, 0) - counts
    within = torch.arange(e_sorted.numel(), device=x.device) - starts[e_sorted]

    buf = torch.zeros(n_e, max_c, d, dtype=x.dtype, device=x.device)
    buf[e_sorted, within] = x[tok_sorted]
    y = torch.bmm(torch.bmm(buf, w_in).relu(), w_out)
    vals = y[e_sorted, within] * g_sorted.unsqueeze(1)
    out = torch.zeros_like(x)
    out.index_add_(0, tok_sorted, vals)
    return out


def dispatch_flat(x, expert_idx, gate, w_in, w_out):
    """Same algorithm, but every 2-D advanced index replaced by a flat
    index_select / index_copy_ on a reshaped view.

    Advanced indexing with two index tensors builds intermediate index
    structures; a single flat index into a reshaped buffer does not.
    """
    n, d = x.shape
    k = expert_idx.shape[1]
    n_e = w_in.shape[0]
    flat_e = expert_idx.reshape(-1)
    flat_g = gate.reshape(-1)
    flat_tok = torch.arange(n, device=x.device).repeat_interleave(k)

    order = torch.argsort(flat_e)
    e_sorted = flat_e[order]
    tok_sorted = flat_tok[order]
    g_sorted = flat_g[order]

    counts = torch.bincount(e_sorted, minlength=n_e)
    max_c = int(counts.max())
    starts = torch.cumsum(counts, 0) - counts
    within = torch.arange(e_sorted.numel(), device=x.device) - starts[e_sorted]
    flat_pos = e_sorted * max_c + within

    buf = torch.zeros(n_e * max_c, d, dtype=x.dtype, device=x.device)
    buf.index_copy_(0, flat_pos, x.index_select(0, tok_sorted))
    y = torch.bmm(
        torch.bmm(buf.view(n_e, max_c, d), w_in).relu(), w_out
    ).view(n_e * max_c, d)
    vals = y.index_select(0, flat_pos) * g_sorted.unsqueeze(1)
    out = torch.zeros_like(x)
    out.index_add_(0, tok_sorted, vals)
    return out


def dispatch_capacity(x, expert_idx, gate, w_in, w_out, capacity_factor=1.25):
    """Fixed per-expert capacity, as in Switch Transformer.

    Padding is bounded by construction instead of being set by whichever
    expert happened to be most popular. Assignments beyond capacity are
    dropped, which is standard practice and is what makes the buffer a fixed
    shape -- so this variant does strictly LESS work than the others and is
    not a like-for-like comparison on quality. Reported separately for that
    reason.
    """
    n, d = x.shape
    k = expert_idx.shape[1]
    n_e = w_in.shape[0]
    cap = max(1, int(capacity_factor * n * k / n_e))

    flat_e = expert_idx.reshape(-1)
    flat_g = gate.reshape(-1)
    flat_tok = torch.arange(n, device=x.device).repeat_interleave(k)

    order = torch.argsort(flat_e)
    e_sorted = flat_e[order]
    tok_sorted = flat_tok[order]
    g_sorted = flat_g[order]

    counts = torch.bincount(e_sorted, minlength=n_e)
    starts = torch.cumsum(counts, 0) - counts
    within = torch.arange(e_sorted.numel(), device=x.device) - starts[e_sorted]

    keep = within < cap
    e_sorted = e_sorted[keep]
    tok_sorted = tok_sorted[keep]
    g_sorted = g_sorted[keep]
    within = within[keep]
    flat_pos = e_sorted * cap + within

    buf = torch.zeros(n_e * cap, d, dtype=x.dtype, device=x.device)
    buf.index_copy_(0, flat_pos, x.index_select(0, tok_sorted))
    y = torch.bmm(
        torch.bmm(buf.view(n_e, cap, d), w_in).relu(), w_out
    ).view(n_e * cap, d)
    vals = y.index_select(0, flat_pos) * g_sorted.unsqueeze(1)
    out = torch.zeros_like(x)
    out.index_add_(0, tok_sorted, vals)
    return out


VARIANTS = {
    "current": dispatch_current,
    "flat_index": dispatch_flat,
    "capacity_1.25": dispatch_capacity,
}


def bench(fn, x, topi, gate, w_in, w_out, reps=4):
    def step():
        w_in.grad = None
        w_out.grad = None
        fn(x, topi, gate, w_in, w_out).square().sum().backward()

    step()
    b = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        step()
        b = min(b, time.perf_counter() - t0)
    return b


def main() -> None:
    print("Routed dispatch variants, forward + backward.\n")
    hdr = (f"{'experts':>8}{'tok/exp':>9}" +
           "".join(f"{n:>15}" for n in VARIANTS) + f"{'best GF/s':>11}")
    print(hdr)
    print("-" * len(hdr))

    for n_e, d, d_ff, n_t, k in (
        (128, 256, 512, 8192, 2),
        (128, 256, 512, 16384, 8),
        (512, 256, 512, 32768, 8),
    ):
        g = torch.Generator().manual_seed(0)
        x = torch.randn(n_t, d, generator=g)
        w_in = (torch.randn(n_e, d, d_ff, generator=g)
                / d**0.5).requires_grad_(True)
        w_out = (torch.randn(n_e, d_ff, d, generator=g)
                 / d_ff**0.5).requires_grad_(True)
        topi = torch.topk(torch.randn(n_t, n_e, generator=g), k,
                          dim=-1).indices
        gate = torch.softmax(torch.randn(n_t, k, generator=g), dim=-1)

        ref = dispatch_current(x, topi, gate, w_in.detach(), w_out.detach())
        cells, best_t = [], float("inf")
        for name, fn in VARIANTS.items():
            got = fn(x, topi, gate, w_in.detach(), w_out.detach())
            exact = torch.allclose(ref, got, atol=1e-4)
            t = bench(fn, x, topi, gate, w_in, w_out)
            best_t = min(best_t, t)
            mark = "" if exact else "*"
            cells.append(f"{t * 1000:>13.1f}{mark:>2}")
        useful = 3.0 * 2.0 * n_t * k * (2 * d * d_ff)
        print(f"{n_e:>8}{n_t * k / n_e:>9.0f}" + "".join(cells)
              + f"{useful / best_t / 1e9:>11.1f}")

    print("\n  * = output differs from reference (capacity variant drops")
    print("  over-capacity assignments, so it does less work by design).")


if __name__ == "__main__":
    main()
