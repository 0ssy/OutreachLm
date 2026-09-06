"""Optimiser state costs 48-55% of a step. What actually makes it cheap?

Adam keeps two fp32 moments per parameter, so an update reads and writes
3 x 4 bytes per parameter (m, v, and the parameter) plus the gradient. At the
expert scale in this design that is pure memory traffic, and it dominates the
step -- more than the expert math it exists to support.

Four candidates, each measured rather than reasoned about:

  adam_fp32     current: m and v in fp32
  adam_bf16     moments stored bf16, computed fp32 -- halves state traffic
  momentum_only single fp32 buffer, no second moment
  factored      Adafactor-style: second moment kept as row and column
                factors instead of a full tensor, so state falls from
                2*P to P + (rows + cols)

The comparison that matters is throughput in parameters per second, because
the budget prices this term as served_params / OPTIM_PARAMS_PER_SEC.
Correctness is checked separately -- a faster optimiser that trains worse is
not a saving, so each variant is also run on a small regression to confirm it
converges.
"""
from __future__ import annotations

import time

import torch

torch.set_num_threads(6)
E, D, F_ = 512, 128, 256


def _state(shape, dtype=torch.float32):
    return torch.zeros(shape, dtype=dtype)


def adam_fp32(p, g, st, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    st["m"].mul_(b1).add_(g, alpha=1 - b1)
    st["v"].mul_(b2).addcmul_(g, g, value=1 - b2)
    p.addcdiv_(st["m"], st["v"].sqrt().add_(eps), value=-lr)


def adam_bf16(p, g, st, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    m = st["m"].float().mul_(b1).add_(g, alpha=1 - b1)
    v = st["v"].float().mul_(b2).addcmul_(g, g, value=1 - b2)
    st["m"].copy_(m)
    st["v"].copy_(v)
    p.addcdiv_(m, v.sqrt().add_(eps), value=-lr)


def momentum_only(p, g, st, lr=1e-3, b1=0.9):
    st["m"].mul_(b1).add_(g, alpha=1 - b1)
    p.add_(st["m"], alpha=-lr)


def factored(p, g, st, lr=1e-3, b1=0.9, b2=0.999, eps=1e-30):
    """Adafactor-style: the second moment is kept as row and column factors.

    For an (E, D, F) expert tensor the factors are (E, D) and (E, F), so
    state falls from 2*E*D*F to E*D*F + E*(D+F) -- close to half.
    """
    g2 = g * g
    st["vr"].mul_(b2).add_(g2.mean(dim=2), alpha=1 - b2)
    st["vc"].mul_(b2).add_(g2.mean(dim=1), alpha=1 - b2)
    r = st["vr"] / st["vr"].mean(dim=1, keepdim=True).clamp_min(eps)
    v = r.unsqueeze(2) * st["vc"].unsqueeze(1)
    st["m"].mul_(b1).add_(g, alpha=1 - b1)
    p.addcdiv_(st["m"], v.sqrt().add_(1e-8), value=-lr)


VARIANTS = {
    "adam_fp32": (adam_fp32,
                  lambda: {"m": _state((E, D, F_)), "v": _state((E, D, F_))},
                  2.0),
    "adam_bf16": (adam_bf16,
                  lambda: {"m": _state((E, D, F_), torch.bfloat16),
                           "v": _state((E, D, F_), torch.bfloat16)},
                  1.0),
    "momentum_only": (momentum_only,
                      lambda: {"m": _state((E, D, F_))},
                      1.0),
    "factored": (factored,
                 lambda: {"m": _state((E, D, F_)),
                          "vr": _state((E, D)), "vc": _state((E, F_))},
                 1.0),
}


def bench(fn, make_state, reps=8):
    g = torch.Generator().manual_seed(0)
    p = torch.randn(E, D, F_, generator=g)
    grad = torch.randn(E, D, F_, generator=g) * 0.01
    st = make_state()
    fn(p, grad, st)
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(p, grad, st)
        best = min(best, time.perf_counter() - t0)
    return best


def converges(fn, make_state, steps=300) -> float:
    """Small regression: fit a fixed target and report final error."""
    g = torch.Generator().manual_seed(1)
    target = torch.randn(64, 32, generator=g)
    p = torch.zeros(64, 32)
    st = {k: (torch.zeros(64, 32, dtype=v.dtype) if v.dim() == 3
              else torch.zeros(64 if v.shape[-1] != 32 else 64))
          for k, v in make_state().items()}
    # Rebuild state at the small shape.
    if "vr" in st:
        st = {"m": torch.zeros(1, 64, 32), "vr": torch.zeros(1, 64),
              "vc": torch.zeros(1, 32)}
        pp = p.unsqueeze(0)
        tt = target.unsqueeze(0)
        for _ in range(steps):
            fn(pp, (pp - tt), st, lr=0.05)
        return float((pp[0] - target).abs().mean())
    st = {k: torch.zeros(64, 32, dtype=v.dtype)
          for k, v in make_state().items()}
    for _ in range(steps):
        fn(p, (p - target), st, lr=0.05)
    return float((p - target).abs().mean())


def main() -> None:
    n = E * D * F_
    print(f"Optimiser state cost, {n / 1e6:.1f}M parameters "
          f"({E} experts x {D} x {F_})\n")
    print(f"{'variant':>15}{'state/param':>13}{'ms':>9}"
          f"{'M param/s':>12}{'speedup':>10}{'converges':>12}")
    print("-" * 71)
    base = None
    for name, (fn, mk, state_mult) in VARIANTS.items():
        t = bench(fn, mk)
        rate = n / t / 1e6
        if base is None:
            base = rate
        err = converges(fn, mk)
        print(f"{name:>15}{state_mult:>12.1f}x{t * 1000:>9.2f}"
              f"{rate:>12.0f}{rate / base:>9.2f}x{err:>12.2e}")

    print("\n  state/param is relative to Adam's two fp32 moments.")
    print("  'converges' is final error on a fixed-target regression; a")
    print("  faster optimiser that does not converge is not a saving.")


if __name__ == "__main__":
    main()
