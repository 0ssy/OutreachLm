"""Where does routed dispatch lose 3.8x against an isolated GEMM?

The budget's dominant term is expert math at 94% of CPU time, priced at the
measured routed throughput of 45.4 GFLOP/s. An isolated GEMM on the same
machine reaches 173. Closing any part of that gap is worth more than every
other optimisation combined, and the optimiser -- which looked like the
bottleneck in an unamortised step decomposition -- is 1% of the budget.

Candidate causes, measured here rather than guessed:

  PADDING       dispatch_batched pads every expert's block to the largest
                count, so the batched GEMM does E * max_count work instead of
                sum(count). Under-utilisation shows up directly as wasted
                FLOPs.
  GATHER        x[tok_sorted] materialises a permuted copy of the input.
  SCATTER       index_add_ writes results back through an indirection.
  SMALL-M GEMM  each expert's block has few rows, so the batched GEMM runs at
                a much lower fraction of peak than a single large one.
"""
from __future__ import annotations

import time

import torch

torch.set_num_threads(6)


def bench(fn, reps=6):
    fn()
    b = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        b = min(b, time.perf_counter() - t0)
    return b * 1000


def analyse(n_experts, d, d_ff, n_tokens, k):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n_tokens, d, generator=g)
    w_in = torch.randn(n_experts, d, d_ff, generator=g) / d**0.5
    w_out = torch.randn(n_experts, d_ff, d, generator=g) / d_ff**0.5
    logits = torch.randn(n_tokens, n_experts, generator=g)
    topi = torch.topk(logits, k, dim=-1).indices

    flat_e = topi.reshape(-1)
    order = torch.argsort(flat_e)
    e_sorted = flat_e[order]
    tok_sorted = torch.arange(n_tokens).repeat_interleave(k)[order]
    counts = torch.bincount(e_sorted, minlength=n_experts)
    max_c = int(counts.max())
    mean_c = float(counts.float().mean())

    starts = torch.cumsum(counts, 0) - counts
    within = torch.arange(e_sorted.numel()) - starts[e_sorted]
    buf = torch.zeros(n_experts, max_c, d)

    t_gather = bench(lambda: x[tok_sorted])
    src = x[tok_sorted]

    def scatter():
        b = torch.zeros(n_experts, max_c, d)
        b[e_sorted, within] = src
        return b

    t_scatter = bench(scatter)
    buf = scatter()

    t_gemm = bench(lambda: torch.bmm(torch.bmm(buf, w_in).relu(), w_out))
    y = torch.bmm(torch.bmm(buf, w_in).relu(), w_out)

    out = torch.zeros_like(x)
    t_back = bench(lambda: out.index_add_(
        0, tok_sorted, y[e_sorted, within]))

    useful = n_tokens * k
    padded = n_experts * max_c
    pad_waste = 1.0 - useful / padded

    total = t_gather + t_scatter + t_gemm + t_back
    return {
        "mean_c": mean_c, "max_c": max_c, "pad_waste": pad_waste,
        "gather": t_gather, "scatter": t_scatter, "gemm": t_gemm,
        "unscatter": t_back, "total": total,
        "useful_gflops": 2.0 * useful * (d * d_ff + d_ff * d) / (
            total / 1000) / 1e9,
        "gemm_gflops": 2.0 * padded * (d * d_ff + d_ff * d) / (
            t_gemm / 1000) / 1e9,
    }


def main() -> None:
    print("Forward-only breakdown of dispatch_batched.\n")
    hdr = (f"{'experts':>8}{'tok/exp':>9}{'max/mean':>10}{'pad waste':>11}"
           f"{'gather':>8}{'scatter':>9}{'gemm':>8}{'unscat':>8}"
           f"{'useful':>9}{'gemm':>8}")
    print(hdr)
    print(f"{'':>8}{'':>9}{'':>10}{'':>11}{'ms':>8}{'ms':>9}{'ms':>8}"
          f"{'ms':>8}{'GF/s':>9}{'GF/s':>8}")
    print("-" * len(hdr))
    for n_e, d, d_ff, n_t, k in (
        (64, 128, 256, 2048, 2),
        (512, 256, 512, 8192, 2),
        (1024, 256, 512, 16384, 2),
        (512, 256, 512, 65536, 2),
    ):
        r = analyse(n_e, d, d_ff, n_t, k)
        print(f"{n_e:>8}{n_t * k / n_e:>9.0f}"
              f"{r['max_c'] / r['mean_c']:>10.2f}{r['pad_waste']:>10.0%}"
              f"{r['gather']:>8.1f}{r['scatter']:>9.1f}{r['gemm']:>8.1f}"
              f"{r['unscatter']:>8.1f}{r['useful_gflops']:>9.1f}"
              f"{r['gemm_gflops']:>8.1f}")

    print("\n  'useful GF/s' counts only real assignments; 'gemm GF/s' counts")
    print("  padded work, so the gap between them IS the padding tax.")


if __name__ == "__main__":
    main()
