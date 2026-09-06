"""What throughput does the ROUTED path actually achieve?

SPARSE_GFLOPS was set to 173 from bench_expert_tuned.py, which times a clean
isolated GEMM: one contiguous (512 x 2048) @ (2048 x 512) product. A real
routed step does not look like that. It sorts assignments, pads each expert's
block, issues a batched GEMM, scatters results back, and runs the whole thing
again in reverse for the backward pass.

Using the isolated number made the budget ~4x optimistic, and validate_budget
caught it. This is the same lesson the project has now hit repeatedly: the
isolated benchmark overstates what the integrated path delivers. It was
measured for Methods C/D/E, for K2's GEMM fragmentation, and for the
contiguous-column scatter -- and then repeated here at the level of the whole
cost model.

This measures the throughput of the path the budget is actually pricing:
forward plus backward, through dispatch_batched, counting only useful FLOPs.
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from src.sparse_engine.attn_routing_probe import dispatch_batched

torch.set_num_threads(6)
TRIALS = 5


def measure(n_experts, d, d_ff, n_tokens, k):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n_tokens, d, generator=g)
    tgt = torch.randn(n_tokens, d, generator=g)
    w_in = (torch.randn(n_experts, d, d_ff, generator=g)
            / d**0.5).requires_grad_(True)
    w_out = (torch.randn(n_experts, d_ff, d, generator=g)
             / d_ff**0.5).requires_grad_(True)
    logits = torch.randn(n_tokens, n_experts, generator=g)
    topv, topi = torch.topk(logits, k, dim=-1)
    gate = F.softmax(topv, dim=-1)

    def step():
        w_in.grad = None
        w_out.grad = None
        out = dispatch_batched(x, topi, gate, w_in, w_out)
        F.mse_loss(out, tgt).backward()

    step()
    best = float("inf")
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        step()
        best = min(best, time.perf_counter() - t0)

    # Useful FLOPs only: every assignment passes d->d_ff->d, and the backward
    # costs about twice the forward. Padding and gather are overhead, which
    # is exactly what this measurement is meant to capture.
    assignments = n_tokens * k
    fwd = 2.0 * assignments * (d * d_ff + d_ff * d)
    return (3.0 * fwd) / best / 1e9


def main() -> None:
    print("Routed dispatch throughput, forward + backward, useful FLOPs only.")
    print("Isolated GEMM benchmark gave 173 GFLOP/s.\n")
    print(f"{'experts':>8}{'d':>6}{'d_ff':>6}{'tokens':>8}{'k':>4}"
          f"{'tok/expert':>12}{'GFLOP/s':>10}")
    print("-" * 54)
    vals = []
    for n_e, d, d_ff, n_t, k in (
        (64, 128, 256, 2048, 2),
        (256, 128, 256, 2048, 2),
        (64, 256, 512, 4096, 2),
        (256, 256, 256, 4096, 4),
        (512, 256, 512, 8192, 2),
        (1024, 256, 512, 16384, 2),
    ):
        gf = measure(n_e, d, d_ff, n_t, k)
        vals.append(gf)
        print(f"{n_e:>8}{d:>6}{d_ff:>6}{n_t:>8}{k:>4}"
              f"{n_t * k / n_e:>12.0f}{gf:>10.1f}")

    print(f"\n  median {sorted(vals)[len(vals) // 2]:.1f} GFLOP/s, "
          f"range {min(vals):.1f}-{max(vals):.1f}")
    print("  This is what the budget should use, not the isolated 173.")


if __name__ == "__main__":
    main()
