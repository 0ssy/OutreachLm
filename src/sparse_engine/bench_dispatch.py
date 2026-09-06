"""Is per-expert dispatch actually faster, and by how much?

The previous probe took 3.2 hours against a 20-minute estimate because
`w_in[expert_of_token]` materialised an (N, d, d_ff) tensor per step: 178 MB
at N=2048, with a 409 ms backward scatter against a 54 ms forward matmul.

This measures the replacement at the shapes the probe actually uses, forward
and backward, so the run can be costed before it is started rather than
after -- which is the mistake that produced the 10x estimate miss.
"""
from __future__ import annotations

import time

import torch

from src.sparse_engine.attn_routing_probe import (
    dispatch_batched,
    dispatch_per_expert,
    dispatch_per_token,
)

torch.set_num_threads(6)
N, D, D_FF, K = 2048, 96, 192, 2


def bench(fn, w_in, w_out, x, topi, gate, reps=3):
    def step():
        if w_in.grad is not None:
            w_in.grad = None
            w_out.grad = None
        out = fn(x, topi, gate, w_in, w_out)
        out.sum().backward()

    step()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        step()
        best = min(best, time.perf_counter() - t0)
    return best * 1000


def main() -> None:
    print(f"N={N} tokens, d={D}, d_ff={D_FF}, k={K}. Forward + backward.\n")
    print(f"{'experts':>9}{'per-token':>12}{'per-expert':>13}"
          f"{'batched':>11}{'best speedup':>14}")
    print("-" * 59)
    for e in (8, 64, 512):
        g = torch.Generator().manual_seed(0)
        x = torch.randn(N, D, generator=g)
        w_in = (torch.randn(e, D, D_FF, generator=g) / D**0.5
                ).requires_grad_(True)
        w_out = (torch.randn(e, D_FF, D, generator=g) / D_FF**0.5
                 ).requires_grad_(True)
        topi = torch.randint(0, e, (N, K), generator=g)
        gate = torch.full((N, K), 0.5)

        slow = bench(dispatch_per_token, w_in, w_out, x, topi, gate)
        loop = bench(dispatch_per_expert, w_in, w_out, x, topi, gate)
        batch = bench(dispatch_batched, w_in, w_out, x, topi, gate)
        print(f"{e:>9}{slow:>12.1f}{loop:>13.1f}{batch:>11.1f}"
              f"{slow / batch:>13.2f}x")

    print("\n  Projected cost of a 3-arm, 2500-step probe at the fast path:")
    g = torch.Generator().manual_seed(1)
    total = 0.0
    for e in (8, 64, 512):
        x = torch.randn(N, D, generator=g)
        w_in = (torch.randn(e, D, D_FF, generator=g) / D**0.5
                ).requires_grad_(True)
        w_out = (torch.randn(e, D_FF, D, generator=g) / D_FF**0.5
                 ).requires_grad_(True)
        topi = torch.randint(0, e, (N, K), generator=g)
        gate = torch.full((N, K), 0.5)
        total += bench(dispatch_batched, w_in, w_out, x, topi, gate)
    print(f"    dispatch alone: {total * 2500 / 1000 / 60:.1f} min")
    print("    (attention trunk and optimiser add to this)")


if __name__ == "__main__":
    main()
