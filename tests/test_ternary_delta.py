"""Tests for ternary delta coding.

Round-trip fidelity is the load-bearing property: a lossy sync desynchronises
the two machines silently, and the divergence would surface much later as a
quality problem with no obvious cause.
"""
import pytest
import torch

from src.sparse_engine.ternary_delta import (
    compression_ratio,
    decode_delta,
    encode_delta,
    quantize,
)


def _states(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return quantize(torch.randn(n, generator=g), 0.5)


def test_round_trip_is_exact_for_random_deltas():
    for seed in range(5):
        before = _states(4096, seed)
        after = _states(4096, seed + 100)
        blob = encode_delta(before, after)
        assert torch.equal(decode_delta(before, blob), after)


def test_round_trip_handles_no_change_and_total_change():
    before = _states(1024)
    empty = encode_delta(before, before.clone())
    assert torch.equal(decode_delta(before, empty), before)

    flipped = ((before.int() + 1) % 3).to(torch.int8) - 1
    flipped = torch.where(flipped == before, before + 1, flipped)
    flipped = flipped.clamp(-1, 1).to(torch.int8)
    blob = encode_delta(before, flipped)
    assert torch.equal(decode_delta(before, blob), flipped)


def test_round_trip_survives_very_large_gaps():
    """Gaps far beyond a single byte must cascade correctly. A fixed 6-bit
    field cannot represent these at all, which is why the encoding is
    variable-length."""
    n = 1 << 20
    before = torch.zeros(n, dtype=torch.int8)
    after = before.clone()
    after[0] = 1
    after[n - 1] = -1
    blob = encode_delta(before, after)
    assert torch.equal(decode_delta(before, blob), after)


def test_shape_is_preserved():
    before = _states(64).reshape(8, 8)
    after = _states(64, 7).reshape(8, 8)
    blob = encode_delta(before, after)
    out = decode_delta(before, blob)
    assert out.shape == (8, 8)
    assert torch.equal(out, after)


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError):
        encode_delta(_states(16), _states(32))


def test_compression_is_measured_against_the_packed_form():
    """Comparing against fp32 would inflate every ratio 16x for free; the
    design already stores ternary, so that is the flattering denominator."""
    n = 1 << 16
    assert compression_ratio(n, n * 2 // 8) == pytest.approx(1.0)


def test_sparse_deltas_compress_and_dense_ones_do_not():
    n = 1 << 16
    g = torch.Generator().manual_seed(3)
    before = _states(n)

    sparse = before.clone()
    idx = torch.randperm(n, generator=g)[: n // 1000]
    sparse[idx] = -before[idx]
    r_sparse = compression_ratio(n, len(encode_delta(before, sparse)))

    dense = -before
    r_dense = compression_ratio(n, len(encode_delta(before, dense)))

    assert r_sparse > 20.0
    assert r_dense < 1.0        # gap-coding a dense change is a pessimisation


def test_escape_path_handles_gaps_beyond_a_uint16():
    """Gaps >= 65535 are escaped rather than truncated. Rare at realistic
    flip rates, but silent truncation would corrupt every later position."""
    n = 400_000
    before = torch.zeros(n, dtype=torch.int8)
    after = before.clone()
    after[5] = 1
    after[200_000] = -1        # gap far beyond 0xFFFF
    after[n - 1] = 1
    blob = encode_delta(before, after)
    assert torch.equal(decode_delta(before, blob), after)


def test_encoder_cost_does_not_scale_with_flip_count():
    """The property that distinguishes vectorised from a Python loop.

    A loop over flips costs O(flips); vectorised numpy costs O(array). So a
    100x increase in flips at constant array size must NOT produce anything
    like a 100x increase in time. The previous implementation looped and would
    have needed ~630 million iterations per sync in production.
    """
    import time

    n = 1 << 22
    g = torch.Generator().manual_seed(5)
    before = _states(n)

    def timed(frac):
        after = before.clone()
        idx = torch.randperm(n, generator=g)[: int(frac * n)]
        after[idx] = torch.where(
            before[idx] > 0, torch.tensor(-1, dtype=torch.int8),
            torch.tensor(1, dtype=torch.int8),
        )
        flips = int((before != after).sum())
        t0 = time.perf_counter()
        blob = encode_delta(before, after)
        el = time.perf_counter() - t0
        assert torch.equal(decode_delta(before, blob), after)
        return flips, el

    f_lo, t_lo = timed(0.0001)
    f_hi, t_hi = timed(0.01)
    flip_ratio = f_hi / max(1, f_lo)
    time_ratio = t_hi / max(1e-9, t_lo)
    assert flip_ratio > 50
    assert time_ratio < flip_ratio / 10, (
        f"flips x{flip_ratio:.0f} but time x{time_ratio:.1f} -- looks like a "
        f"per-flip loop"
    )


def test_fixed_width_format_beats_the_varint_it_replaced():
    """2.25 bytes/flip against the varint encoder's measured 2.88."""
    from src.sparse_engine.ternary_delta import bytes_per_flip

    n = 1 << 20
    g = torch.Generator().manual_seed(6)
    before = _states(n)
    after = before.clone()
    idx = torch.randperm(n, generator=g)[: n // 1000]
    after[idx] = -before[idx]
    blob = encode_delta(before, after)
    flips = int((before != after).sum())
    assert bytes_per_flip(flips, len(blob)) < 2.88
