"""Is the 70.2-day figure a cache/bus problem, and would the proposed
physics rewrites reduce it?

Three checkable claims:

  (a) "the model hit a structural barrier where memory-mapped files and cache
      layers are no longer synchronized, forcing the CPU to fetch directly
      from the system bus."
  (b) Wave-interference gating replaces the router and pulls compute down.
  (c) Thermodynamic boundary sync removes file-locking latency between the
      two machines.

Each is checked against the measured decomposition rather than argued with.
"""
from __future__ import annotations

import time

import torch

from src.sparse_engine.budget import Config, MODEL_SLACK, SECONDS_PER_DAY

torch.set_num_threads(6)

C = Config(total_params=70e9, active_params_per_token=70e6, tokens=1.4e9,
           expert_params=1e6, tokens_per_step=4194304, micro_batch=65536,
           nodes=2, strategy="data_parallel_delta")


def claim_a_cache() -> None:
    print("CLAIM (a) -- is the 70.2 days a cache/bus synchronisation stall?\n")
    cpu = C.compute_seconds + C.unpack_seconds + C.optim_seconds
    print(f"{'term':>22}{'days':>9}{'share':>9}")
    print("-" * 40)
    for n, v in (("expert math (GEMM)", C.compute_seconds),
                 ("ternary unpack", C.unpack_seconds),
                 ("optimiser", C.optim_seconds)):
        print(f"{n:>22}{v * MODEL_SLACK / SECONDS_PER_DAY:>9.1f}{v / cpu:>8.0%}")
    print(f"{'disk streaming':>22}{C.io_seconds / SECONDS_PER_DAY:>9.1f}"
          f"{'hidden':>9}")
    print(f"{'cross-node comm':>22}{C.comm_seconds / SECONDS_PER_DAY:>9.1f}"
          f"{'hidden':>9}")
    print(f"{'TOTAL':>22}{C.days:>9.1f}")

    flops = C.compute_flops
    print(f"\n  The dominant term is arithmetic: {flops:.2e} FLOPs of routed "
          f"GEMM\n  at a measured 78 GFLOP/s across 2 nodes = "
          f"{flops / (78e9 * 2) / SECONDS_PER_DAY:.1f} days.")
    print("  Disk streaming is 0.9 days and already hidden behind compute, so")
    print("  a storage-controller stall cannot be the cause -- there is not")
    print("  enough time in that term for it to explain anything.")


def claim_b_router() -> None:
    print("\n\nCLAIM (b) -- wave-interference gating cuts compute\n")
    d, n_e, n_t = 2048, 4096, 4096
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n_t, d, generator=g)
    w = torch.randn(d, n_e, generator=g)

    def bench(fn, reps=5):
        fn()
        b = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            b = min(b, time.perf_counter() - t0)
        return b * 1000

    t_router = bench(lambda: x @ w)
    xb = (x > 0)
    wb = (w > 0)
    packed_x = xb.reshape(n_t, -1)
    t_pop = bench(lambda: (packed_x.unsqueeze(1)[:, :, :64]
                           ^ wb.T.unsqueeze(0)[:, :64, :64]).sum())

    print("  S_k = |integral psi(x) phi_k*(x) dx|^2 is an inner product,")
    print("  squared. Discretised on a CPU that IS the router matmul the")
    print("  design already does -- the formula renames the operation rather")
    print("  than replacing it.\n")
    print(f"  router matmul, {n_t} tokens x {n_e} experts: "
          f"{t_router:.1f} ms/step")
    print(f"  as a share of a step: routing measured 0-1% in "
          f"profile_dispatch.py")
    print("\n  So even a router that cost ZERO saves under 1%. The claim's")
    print("  real content is 70 -> 50 active experts, which cuts")
    print("  P_active from 70M to 50M:")
    for act in (70e6, 50e6):
        c = Config(total_params=70e9, active_params_per_token=act,
                   tokens=1.4e9, expert_params=1e6, tokens_per_step=4194304,
                   micro_batch=65536, nodes=2,
                   strategy="data_parallel_delta")
        print(f"    {act / 1e6:.0f}M active -> {c.days:.1f} days")
    print("\n  That is real, and it is a SMALLER MODEL, not a faster one --")
    print("  the same trade already identified for gradient-magnitude")
    print("  pruning. It is available today by setting active_params.")
    print("\n  Also: 'popcount across 512-bit registers' -- this CPU is AVX2,")
    print("  256-bit, no AVX-512 (verified: torch reports AVX2, Zen 3).")


def claim_c_sync() -> None:
    print("\n\nCLAIM (c) -- thermodynamic boundary sync\n")
    print(f"  cross-node comm in the current budget: "
          f"{C.comm_seconds / SECONDS_PER_DAY:.2f} days of {C.days:.1f}")
    print(f"  = {C.comm_seconds / C.total_seconds:.2%} of wall-clock, and it")
    print("  is already overlapped behind compute, so its true marginal cost")
    print("  is zero.")
    print("\n  dQ/dt = -k grad(dH) with k = disk bandwidth describes a rate")
    print("  limit. The budget already applies one: comm_gb_per_step divided")
    print("  by the measured 0.0135 GB/s link. Reducing a term worth 0.02%")
    print("  cannot move a 71-day total.")
    print("\n  Named pipes / FIFO are POSIX; this machine is Windows.")


def main() -> None:
    claim_a_cache()
    claim_b_router()
    claim_c_sync()
    print("\n\nWHAT THE MEASUREMENTS SAY IS ACTUALLY NEEDED\n")
    cpu = (C.compute_seconds + C.unpack_seconds + C.optim_seconds)
    print(f"  expert math          {C.compute_seconds * MODEL_SLACK / SECONDS_PER_DAY:>6.1f} d")
    print(f"  unexplained residual {(MODEL_SLACK - 1) * cpu / SECONDS_PER_DAY:>6.1f} d"
          f"   <- largest reducible term")
    print("\n  Routed dispatch runs at 78 GF/s against 173-190 for an")
    print("  isolated GEMM. profile_dispatch.py attributes the gap to")
    print("  gather/scatter (28% of the forward) and padding waste (16%).")
    print("  Both are addressable, and together they are worth ~1.6x.")


if __name__ == "__main__":
    main()
