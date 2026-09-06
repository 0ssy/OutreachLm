"""Paths to 16 days, tested rather than proposed.

FIRST, A CORRECTION THAT CHANGES THE TARGET
    The premise "42.4 days single-machine, so dual-machine is ~21.2" is not
    right: 42.4 IS the two-node figure. Both nodes are already in it, and
    halving it again is not available. Verified below.

WHY THE TWO PROPOSED MODULES CANNOT DELIVER, IN ONE LINE EACH
    Module A optimises routing, measured at 0-1% of a step. Even a free
    router saves under 1%. Its real content -- 70 to 50 active experts -- is
    a smaller model, already available as a parameter.
    Module B optimises cross-node communication, measured at 0.00 days of
    42.4 and already overlapped behind compute. There is nothing to remove.
    Both also assume AVX-512 (`vprold`/`vprolq`) and POSIX FIFOs; this is an
    AVX2 Zen 3 CPU running Windows.

WHAT IS ACTUALLY UNTESTED, AND IS TESTED HERE
    Every measurement in this project has been fp32. The expert GEMM is 25.0
    of the 42.4 days, so its arithmetic precision and thread configuration
    are the two largest levers nobody has tried. If bf16 or fp16 GEMM is
    faster on this CPU -- even without AVX512-BF16, through halved memory
    traffic -- that is a direct multiplier on the dominant term.
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from src.sparse_engine.attn_routing_probe import dispatch_batched
from src.sparse_engine.budget import Config


def verify_node_count() -> None:
    print("CORRECTION -- is 42.4 days one machine or two?\n")
    base = dict(total_params=70e9, active_params_per_token=70e6,
                tokens=1.4e9, expert_params=1e6, tokens_per_step=4194304,
                micro_batch=65536, strategy="data_parallel_delta")
    one = Config(nodes=1, **base)
    two = Config(nodes=2, **base)
    print(f"  one node   {one.days:.1f} days")
    print(f"  two nodes  {two.days:.1f} days   <- the figure reported")
    print(f"\n  The second machine is already counted. The remaining work is"
          f"\n  {two.compute_seconds / 86400:.1f} days of expert GEMM plus a "
          f"1.63x measured residual.")


def bench_precision(n_e=128, d=256, d_ff=512, n_t=16384, k=8, reps=4):
    print("\n\nLEVER 1 -- arithmetic precision of the expert GEMM\n")
    g = torch.Generator().manual_seed(0)
    x32 = torch.randn(n_t, d, generator=g)
    tgt32 = torch.randn(n_t, d, generator=g)
    wi32 = torch.randn(n_e, d, d_ff, generator=g) / d**0.5
    wo32 = torch.randn(n_e, d_ff, d, generator=g) / d_ff**0.5
    topi = torch.topk(torch.randn(n_t, n_e, generator=g), k, dim=-1).indices
    gate = F.softmax(torch.randn(n_t, k, generator=g), dim=-1)

    flops = 12.0 * n_t * k * d * d_ff
    print(f"{'dtype':>12}{'ms':>10}{'GFLOP/s':>11}{'vs fp32':>10}"
          f"{'rel error':>12}")
    print("-" * 55)

    ref_out = None
    base = None
    for name, dt in (("float32", torch.float32),
                     ("bfloat16", torch.bfloat16),
                     ("float16", torch.float16)):
        x = x32.to(dt)
        tgt = tgt32.to(dt)
        wi = wi32.to(dt).requires_grad_(True)
        wo = wo32.to(dt).requires_grad_(True)
        gt = gate.to(dt)

        def step():
            wi.grad = None
            wo.grad = None
            out = dispatch_batched(x, topi, gt, wi, wo)
            F.mse_loss(out.float(), tgt.float()).backward()
            return out

        try:
            for _ in range(2):
                out = step()
            b = float("inf")
            for _ in range(reps):
                t0 = time.perf_counter()
                step()
                b = min(b, time.perf_counter() - t0)
        except Exception as e:                                # noqa: BLE001
            print(f"{name:>12}{'unsupported':>10}  {type(e).__name__}")
            continue

        gf = flops / b / 1e9
        if base is None:
            base = gf
            ref_out = out.float()
            err = 0.0
        else:
            err = float((out.float() - ref_out).norm() / ref_out.norm())
        print(f"{name:>12}{b * 1000:>10.1f}{gf:>11.1f}{gf / base:>9.2f}x"
              f"{err:>12.2e}")


def bench_threads(n_e=128, d=256, d_ff=512, n_t=16384, k=8, reps=4):
    print("\n\nLEVER 2 -- thread count (6 physical / 12 logical cores)\n")
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n_t, d, generator=g)
    tgt = torch.randn(n_t, d, generator=g)
    wi = (torch.randn(n_e, d, d_ff, generator=g) / d**0.5).requires_grad_(True)
    wo = (torch.randn(n_e, d_ff, d, generator=g)
          / d_ff**0.5).requires_grad_(True)
    topi = torch.topk(torch.randn(n_t, n_e, generator=g), k, dim=-1).indices
    gate = F.softmax(torch.randn(n_t, k, generator=g), dim=-1)
    flops = 12.0 * n_t * k * d * d_ff

    print(f"{'threads':>9}{'ms':>10}{'GFLOP/s':>11}{'vs 6':>9}")
    print("-" * 39)
    base = None
    for nt in (4, 6, 8, 12):
        torch.set_num_threads(nt)

        def step():
            wi.grad = None
            wo.grad = None
            F.mse_loss(dispatch_batched(x, topi, gate, wi, wo), tgt).backward()

        for _ in range(2):
            step()
        b = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            step()
            b = min(b, time.perf_counter() - t0)
        gf = flops / b / 1e9
        if nt == 6:
            base = gf
        print(f"{nt:>9}{b * 1000:>10.1f}{gf:>11.1f}"
              f"{(gf / base if base else float('nan')):>8.2f}x")
    torch.set_num_threads(6)


def main() -> None:
    torch.set_num_threads(6)
    verify_node_count()
    bench_precision()
    bench_threads()


if __name__ == "__main__":
    main()
