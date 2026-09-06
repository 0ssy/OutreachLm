"""End-to-end validation: does the budget predict a REAL measured run?

WHY THIS EXISTS
    Every constant in budget.py has provenance -- each came from a benchmark
    in this package. But provenance of the INPUTS is not validation of the
    MODEL. The 20.5-day figure comes from an analytical formula that had never
    been checked against an actual training step of the actual architecture,
    which means a structurally wrong model would produce a confident number
    and nothing would catch it.

    This runs a real configuration -- ternary weights packed on disk, streamed
    per step, routed MoE forward and backward, sparse optimiser -- measures
    wall-clock, and compares against what the budget predicts for the SAME
    configuration. Agreement is evidence the model composes its constants
    correctly; disagreement localises which term is wrong.

WHAT WOULD MAKE THIS A RIGGED TEST, AND IS THEREFORE AVOIDED
    * Predicting and measuring at the same scale the constants were fitted at
      would only show the constants were copied correctly. The run uses
      shapes and expert counts that were NOT used to derive SPARSE_GFLOPS.
    * Timing only the GEMM would hide exactly the overheads that have
      repeatedly dominated in this project. The whole step is timed:
      streaming, unpack, route, dispatch, backward, optimiser.
    * Comparing FLOP counts rather than seconds would restate the model
      instead of testing it.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.sparse_engine.attn_routing_probe import dispatch_batched
from src.sparse_engine.budget import Config
from src.sparse_engine.sparse_optim import SparseExpertAdam
from src.sparse_engine.ternary_delta import unpack_states

torch.set_num_threads(6)


def pack_ternary(w: torch.Tensor) -> np.ndarray:
    """4 weights per byte, matching the 2-bit storage the budget assumes."""
    q = torch.sign(w) * (w.abs() > w.abs().median())
    codes = np.zeros(q.numel(), dtype=np.uint8)
    flat = q.reshape(-1).numpy()
    codes[flat > 0] = 1
    codes[flat < 0] = 2
    pad = (-codes.size) % 4
    if pad:
        codes = np.concatenate([codes, np.zeros(pad, dtype=np.uint8)])
    return (codes[0::4] | (codes[1::4] << 2) | (codes[2::4] << 4)
            | (codes[3::4] << 6))


def unpack_ternary(buf: np.ndarray, n: int, shape) -> torch.Tensor:
    """Uses the byte-wide LUT from ternary_delta -- 155 M params/s against
    94 M for nibble shifts. The validator must use the same code path the
    budget prices, or it measures a strawman."""
    return torch.from_numpy(
        unpack_states(buf, n).astype(np.float32)
    ).reshape(shape)


class RealRun:
    """A minimal but complete instance of the architecture being budgeted."""

    def __init__(self, n_experts: int, d: int, d_ff: int, n_tokens: int,
                 k: int, seed: int = 0):
        self.n_experts, self.d, self.d_ff = n_experts, d, d_ff
        self.n_tokens, self.k = n_tokens, k
        g = torch.Generator().manual_seed(seed)

        self.tmp = Path(tempfile.mkdtemp(prefix="outreachlm_val_"))
        w_in = torch.randn(n_experts, d, d_ff, generator=g) / d**0.5
        w_out = torch.randn(n_experts, d_ff, d, generator=g) / d_ff**0.5
        self.path_in = self.tmp / "w_in.ter"
        self.path_out = self.tmp / "w_out.ter"
        self.path_in.write_bytes(pack_ternary(w_in).tobytes())
        self.path_out.write_bytes(pack_ternary(w_out).tobytes())
        self.n_in, self.n_out = w_in.numel(), w_out.numel()
        self.shape_in, self.shape_out = w_in.shape, w_out.shape

        self.router = torch.randn(d, n_experts, generator=g) * 0.02
        self.x = torch.randn(n_tokens, d, generator=g)
        self.target = torch.randn(n_tokens, d, generator=g)
        self.bytes_streamed = (
            self.path_in.stat().st_size + self.path_out.stat().st_size
        )
        # Optimiser state persists across steps, as it would in training.
        # Rebuilding it step cost 6-8% and is not something the real loop
        # would do.
        self._opt = None

    def step(self) -> None:
        """One complete step: stream, unpack, route, dispatch, bwd, update."""
        raw_in = np.fromfile(self.path_in, dtype=np.uint8)
        raw_out = np.fromfile(self.path_out, dtype=np.uint8)
        w_in = unpack_ternary(raw_in, self.n_in, self.shape_in
                              ).requires_grad_(True)
        w_out = unpack_ternary(raw_out, self.n_out, self.shape_out
                               ).requires_grad_(True)
        if self._opt is None:
            self._opt = SparseExpertAdam([w_in, w_out], lr=1e-3)
        opt = self._opt
        opt.params = [w_in, w_out]

        logits = self.x @ self.router
        topv, topi = torch.topk(logits, self.k, dim=-1)
        gate = F.softmax(topv, dim=-1)
        out = dispatch_batched(self.x, topi, gate, w_in, w_out)
        loss = F.mse_loss(out, self.target)
        loss.backward()
        opt.step(rows=torch.unique(topi))

    def cleanup(self) -> None:
        for p in (self.path_in, self.path_out):
            p.unlink(missing_ok=True)
        self.tmp.rmdir()


def measure(n_experts, d, d_ff, n_tokens, k, steps=6) -> tuple[float, int]:
    run = RealRun(n_experts, d, d_ff, n_tokens, k)
    try:
        run.step()
        t0 = time.perf_counter()
        for _ in range(steps):
            run.step()
        return (time.perf_counter() - t0) / steps, run.bytes_streamed
    finally:
        run.cleanup()


def predict(n_experts, d, d_ff, n_tokens, k) -> float:
    """Budget's per-step prediction for the same shape."""
    expert_params = d * d_ff * 2
    cfg = Config(
        total_params=n_experts * expert_params,
        active_params_per_token=k * expert_params,
        tokens=n_tokens,
        expert_params=expert_params,
        tokens_per_step=n_tokens,
        nodes=1,
        strategy="expert_sharded",
    )
    return cfg.total_seconds


def main() -> None:
    print("Budget model vs a real measured step.")
    print(f"Constants were fitted at d_model=2048, d_ff=512, 512 "
          f"tokens/expert;\nthe cases below deliberately differ.\n")
    print(f"{'experts':>8}{'d':>6}{'d_ff':>6}{'tokens':>8}"
          f"{'predicted':>12}{'measured':>11}{'ratio':>8}")
    print("-" * 59)
    for n_e, d, d_ff, n_t, k in (
        (64, 128, 256, 2048, 2),
        (256, 128, 256, 2048, 2),
        (64, 256, 512, 4096, 2),
        (256, 256, 256, 4096, 4),
    ):
        meas, _ = measure(n_e, d, d_ff, n_t, k)
        pred = predict(n_e, d, d_ff, n_t, k)
        print(f"{n_e:>8}{d:>6}{d_ff:>6}{n_t:>8}"
              f"{pred * 1000:>11.1f}m{meas * 1000:>10.1f}m"
              f"{meas / pred:>8.2f}x")

    print("\n  ratio > 1 means the real run is SLOWER than budgeted, i.e. the")
    print("  budget is optimistic and the 20.5-day figure is a lower bound.")


if __name__ == "__main__":
    main()
