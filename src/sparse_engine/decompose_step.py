"""Where does the 9-70x gap between budget and reality come from?

The end-to-end validator showed real steps running 9-70x slower than the
budget predicts. Two very different explanations, and they need separating
before either is acted on:

  (a) THE BUDGET UNDERCOUNTS. It prices 6 * P_active * T over SPARSE_GFLOPS
      and a bytes/bandwidth term, and nothing else. It does NOT include the
      attention trunk, the embedding, the head, routing, or the 2-bit -> fp32
      unpack -- all of which a real step pays.

  (b) THE HARNESS IS PESSIMISTIC. It re-reads and unpacks every expert from
      disk each step and rebuilds optimiser state, which the real design
      would not do: streaming is amortised, and moment state persists.

Both are probably true. This decomposes a step so the split is measured
instead of argued, because the correction to the budget depends entirely on
which term dominates.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.sparse_engine.attn_routing_probe import dispatch_batched
from src.sparse_engine.sparse_optim import SparseExpertAdam
from src.sparse_engine.validate_budget import pack_ternary, unpack_ternary

torch.set_num_threads(6)


def decompose(n_experts, d, d_ff, n_tokens, k, reps=5):
    g = torch.Generator().manual_seed(0)
    tmp = Path(tempfile.mkdtemp(prefix="outreachlm_dec_"))
    w_in_t = torch.randn(n_experts, d, d_ff, generator=g) / d**0.5
    w_out_t = torch.randn(n_experts, d_ff, d, generator=g) / d_ff**0.5
    p_in, p_out = tmp / "a.ter", tmp / "b.ter"
    p_in.write_bytes(pack_ternary(w_in_t).tobytes())
    p_out.write_bytes(pack_ternary(w_out_t).tobytes())

    x = torch.randn(n_tokens, d, generator=g)
    target = torch.randn(n_tokens, d, generator=g)
    router = torch.randn(d, n_experts, generator=g) * 0.02

    def t(fn):
        fn()
        b = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            b = min(b, time.perf_counter() - t0)
        return b * 1000

    ms = {}
    ms["disk read"] = t(lambda: (np.fromfile(p_in, dtype=np.uint8),
                                 np.fromfile(p_out, dtype=np.uint8)))
    raw_in = np.fromfile(p_in, dtype=np.uint8)
    raw_out = np.fromfile(p_out, dtype=np.uint8)
    ms["ternary unpack"] = t(
        lambda: (unpack_ternary(raw_in, w_in_t.numel(), w_in_t.shape),
                 unpack_ternary(raw_out, w_out_t.numel(), w_out_t.shape))
    )

    w_in = unpack_ternary(raw_in, w_in_t.numel(),
                          w_in_t.shape).requires_grad_(True)
    w_out = unpack_ternary(raw_out, w_out_t.numel(),
                           w_out_t.shape).requires_grad_(True)
    ms["route (topk)"] = t(lambda: torch.topk(x @ router, k, dim=-1))

    topv, topi = torch.topk(x @ router, k, dim=-1)
    gate = F.softmax(topv, dim=-1)
    ms["dispatch fwd"] = t(
        lambda: dispatch_batched(x, topi, gate, w_in.detach(),
                                 w_out.detach())
    )

    def fwd_bwd():
        w_in.grad = None
        w_out.grad = None
        o = dispatch_batched(x, topi, gate, w_in, w_out)
        F.mse_loss(o, target).backward()

    ms["dispatch fwd+bwd"] = t(fwd_bwd)

    opt = SparseExpertAdam([w_in, w_out], lr=1e-3)
    fwd_bwd()
    rows = torch.unique(topi)
    ms["optimiser step"] = t(lambda: opt.step(rows=rows))
    ms["optimiser CONSTRUCT"] = t(
        lambda: SparseExpertAdam([w_in, w_out], lr=1e-3)
    )

    for p in (p_in, p_out):
        p.unlink(missing_ok=True)
    tmp.rmdir()
    return ms


def main() -> None:
    for cfg in ((64, 128, 256, 2048, 2), (256, 128, 256, 2048, 2)):
        n_e, d, d_ff, n_t, k = cfg
        ms = decompose(*cfg)
        print(f"\nexperts={n_e} d={d} d_ff={d_ff} tokens={n_t} k={k}")
        print(f"{'phase':>24}{'ms':>9}{'share':>9}")
        print("-" * 42)
        # fwd+bwd already contains fwd; count it once.
        billed = {kk: v for kk, v in ms.items()
                  if kk not in ("dispatch fwd",)}
        tot = sum(billed.values())
        for kk, v in ms.items():
            share = "" if kk == "dispatch fwd" else f"{v / tot:>8.0%}"
            print(f"{kk:>24}{v:>9.1f}{share}")
        print(f"{'TOTAL (billed)':>24}{tot:>9.1f}")
        real = billed["dispatch fwd+bwd"]
        print(f"\n  expert math is {real / tot:.0%} of the step; the budget "
              f"prices ONLY that")


if __name__ == "__main__":
    main()
