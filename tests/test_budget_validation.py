"""The budget model, checked against real measured runs.

Every other test in this suite checks that the model is internally consistent.
None of them would catch a model that is coherently wrong -- and the model WAS
coherently wrong: it priced only 17-27% of a step and reported 20.5 days for
work that measured 9-70x slower.

These tests close that loop. They are slower than the rest of the suite
because they actually train.
"""
import numpy as np
import pytest
import torch

from src.sparse_engine.budget import (
    ISOLATED_GEMM_GFLOPS,
    MODEL_SLACK,
    Config,
    routed_gflops,
)
from src.sparse_engine.validate_budget import measure, predict

# Shapes deliberately unlike those any constant was fitted at.
CASES = [
    (64, 128, 256, 2048, 2),
    (256, 128, 256, 2048, 2),
]

# MODEL_SLACK is measured at >=512 tokens/expert, the regime the 2B and 70B
# configurations occupy. Below that, fixed per-step overhead has not
# amortised and the residual is larger (2.1-2.6x at 64-128 tokens/expert
# against 1.59-1.67x at 512-2048). Accuracy claims are therefore made in the
# regime the constant was fitted for, and the small-shape optimism is
# asserted explicitly rather than left as an unnoticed gap.
IN_REGIME = [
    (128, 256, 512, 16384, 4),
    (128, 256, 512, 32768, 8),
]


def test_prediction_is_within_a_stated_factor_of_measurement():
    """The headline guarantee. If this drifts, the reported timeline is
    wrong and MODEL_SLACK needs re-measuring rather than the test relaxing.
    """
    for case in CASES:
        meas, _ = measure(*case, steps=3)
        pred = predict(*case)
        ratio = meas / pred
        assert 0.5 < ratio < 2.0, f"{case}: measured/predicted = {ratio:.2f}"


def test_model_is_accurate_in_the_regime_it_was_fitted_for():
    """A model that under-predicts is worse than one that over-predicts: it
    turns a 'feasible' verdict into a missed deadline. Checked at >=512
    tokens/expert, which is where MODEL_SLACK was measured and where the
    real configurations run."""
    ratios = []
    for case in IN_REGIME:
        meas, _ = measure(*case, steps=2)
        ratios.append(meas / predict(*case))
    mean = sum(ratios) / len(ratios)
    assert 0.70 < mean < 1.30, f"mean measured/predicted = {mean:.2f}"


def test_small_configurations_are_priced_optimistically():
    """Documented rather than hidden: below the fitted regime the residual is
    larger, so small configs are UNDER-predicted. Anyone budgeting a small
    run needs to know that."""
    ratios = []
    for case in CASES:
        meas, _ = measure(*case, steps=2)
        ratios.append(meas / predict(*case))
    assert min(ratios) > 1.0


def test_routed_throughput_stays_below_the_isolated_gemm():
    """Routing costs something real -- sort, pad, gather, scatter -- so the
    routed path must never be priced at the isolated-GEMM figure. The gap is
    now 1.3x at large blocks (136 vs 173) and 2.8x at small ones, where
    padding waste and fixed overhead dominate."""
    assert routed_gflops(32) < ISOLATED_GEMM_GFLOPS / 2
    assert routed_gflops(2048) < ISOLATED_GEMM_GFLOPS
    assert routed_gflops(2048) > routed_gflops(32) * 2


def test_routed_throughput_rises_with_tokens_per_expert():
    """Measured 30.7 GF/s at 32 tokens/expert against 78.0 at 2048, because
    that ratio sets both the GEMM row count and the padding waste. A single
    constant is wrong in both directions."""
    assert routed_gflops(32) < routed_gflops(512) < routed_gflops(2048)
    assert routed_gflops(2048) / routed_gflops(32) > 2.0


def test_throughput_curve_is_clamped_not_extrapolated():
    """Out-of-range configs must be priced conservatively rather than
    extrapolated into numbers nothing measured."""
    assert routed_gflops(1) == routed_gflops(32)
    assert routed_gflops(1e9) == routed_gflops(2048)


def test_slack_is_documented_and_bounded():
    """MODEL_SLACK is a measured residual, not a free parameter. If it ever
    needs to exceed 4x, the model is missing a term rather than a constant."""
    assert 1.0 <= MODEL_SLACK <= 4.0


def test_unpack_and_optimiser_terms_are_actually_material():
    """Both were absent from the first model. This asserts they matter, so
    nobody removes them as noise."""
    c = Config(
        total_params=70e9, active_params_per_token=70e6, tokens=1.4e9,
        expert_params=1e6, tokens_per_step=65536, nodes=2,
        strategy="data_parallel_delta",
    )
    cpu = c.compute_seconds + c.unpack_seconds + c.optim_seconds
    assert c.unpack_seconds / cpu > 0.1
    assert c.optim_seconds / cpu > 0.01
    assert c.compute_seconds / cpu < 0.9


def test_packed_ternary_round_trips_through_the_fast_unpack():
    """The byte-LUT unpack is on the critical path at 24-27% of a step, so a
    correctness bug there would corrupt every weight silently."""
    from src.sparse_engine.ternary_delta import unpack_states
    from src.sparse_engine.validate_budget import pack_ternary

    g = torch.Generator().manual_seed(0)
    w = torch.randn(4096, generator=g)
    packed = pack_ternary(w)
    got = unpack_states(packed, w.numel())
    expect = (torch.sign(w) * (w.abs() > w.abs().median())).to(torch.int8)
    assert np.array_equal(got, expect.numpy())
