"""Ternary delta coding: vectorised, and a better format than varint gaps.

WHY THIS REPLACES THE FIRST VERSION
    The first encoder walked flips in a Python loop. At a 0.9% flip rate over
    70e9 weights that is ~630 million iterations per sync -- hours of pure
    interpreter overhead for an operation that has to fit inside a training
    step. Correct, and unusable.

    Rewriting it also improved the format. Varint gaps plus a whole byte for
    2 bits of state measured 2.88 bytes per flip. Fixed-width uint16 gaps with
    states packed 4-per-byte costs 2.25 bytes per flip AND vectorises, because
    every field sits at a known offset:

        compression = 0.25 bytes/weight / (2.25 bytes/flip * flip_rate)
                    = 0.111 / flip_rate

    which is ~48x at a 0.231% flip rate against the varint encoder's 39.2x.

    Gaps of 65535 or more are escaped: the field value 0xFFFF means "add
    65535 and read another field for this same flip". At the flip rates that
    make the scheme worthwhile the mean gap is ~100-1000, so escapes are
    vanishingly rare -- but they are handled exactly rather than assumed away,
    and the escape path is covered by tests.

WHY THE COMPARISON IS AGAINST THE PACKED 2-BIT FORM
    Comparing against fp32 would multiply every ratio by 16 for free. The
    design already stores ternary, so 2 bits per weight is the honest
    denominator.
"""
from __future__ import annotations

import numpy as np
import torch

STATE_BITS = 2
RAW_BITS_PER_WEIGHT = STATE_BITS
_ESCAPE = 0xFFFF
_FROM_CODE = np.array([0, 1, -1, 0], dtype=np.int8)


def quantize(latent: torch.Tensor, threshold: float) -> torch.Tensor:
    """Ternary quantisation: -1, 0, +1 as int8."""
    return (torch.sign(latent) * (latent.abs() > threshold)).to(torch.int8)


def _codes(vals: np.ndarray) -> np.ndarray:
    """-1 -> 2, 0 -> 0, +1 -> 1. Two bits, four per byte."""
    out = np.zeros(vals.shape, dtype=np.uint8)
    out[vals > 0] = 1
    out[vals < 0] = 2
    return out


def encode_delta(before: torch.Tensor, after: torch.Tensor) -> bytes:
    """Encode every position whose ternary state changed.

    Layout: uint32 count | uint16 gap fields (escaped) | packed 2-bit states.
    """
    if before.shape != after.shape:
        raise ValueError("shapes must match")
    b = before.reshape(-1).numpy()
    a = after.reshape(-1).numpy()
    idx = np.nonzero(b != a)[0]
    n = int(idx.size)
    if n == 0:
        return np.uint32(0).tobytes()

    gaps = np.diff(np.concatenate(([-1], idx))).astype(np.int64) - 1
    n_esc = gaps // _ESCAPE
    total_esc = int(n_esc.sum())

    if total_esc == 0:
        fields = gaps.astype(np.uint16)
    else:
        fields = np.empty(n + total_esc, dtype=np.uint16)
        pos = 0
        for g in gaps.tolist():
            while g >= _ESCAPE:
                fields[pos] = _ESCAPE
                pos += 1
                g -= _ESCAPE
            fields[pos] = g
            pos += 1

    codes = _codes(a[idx])
    pad = (-n) % 4
    if pad:
        codes = np.concatenate([codes, np.zeros(pad, dtype=np.uint8)])
    packed = (
        codes[0::4] | (codes[1::4] << 2) | (codes[2::4] << 4)
        | (codes[3::4] << 6)
    ).astype(np.uint8)
    return np.uint32(n).tobytes() + fields.tobytes() + packed.tobytes()


def decode_delta(before: torch.Tensor, blob: bytes) -> torch.Tensor:
    """Exact inverse of encode_delta.

    Round-trip fidelity is not optional: a lossy sync desynchronises the two
    machines silently, and surfaces much later as an unexplained quality loss
    with no obvious cause.
    """
    out = before.clone().reshape(-1)
    n = int(np.frombuffer(blob, dtype=np.uint32, count=1)[0])
    if n == 0:
        return out.reshape(before.shape)

    body = np.frombuffer(blob, dtype=np.uint8, offset=4)
    n_state_bytes = (n + 3) // 4
    n_field_bytes = body.size - n_state_bytes
    fields = np.frombuffer(
        body.tobytes(), dtype=np.uint16, count=n_field_bytes // 2
    )

    if fields.size == n:
        gaps = fields.astype(np.int64)
    else:
        gaps = np.empty(n, dtype=np.int64)
        acc, k = 0, 0
        for f in fields.tolist():
            if f == _ESCAPE:
                acc += _ESCAPE
            else:
                gaps[k] = acc + f
                acc = 0
                k += 1

    idx = np.cumsum(gaps + 1) - 1

    packed = body[n_field_bytes:]
    codes = np.empty(n_state_bytes * 4, dtype=np.uint8)
    codes[0::4] = packed & 0b11
    codes[1::4] = (packed >> 2) & 0b11
    codes[2::4] = (packed >> 4) & 0b11
    codes[3::4] = (packed >> 6) & 0b11
    vals = _FROM_CODE[codes[:n]]

    out[torch.from_numpy(idx.copy())] = torch.from_numpy(vals.copy())
    return out.reshape(before.shape)


def compression_ratio(n_weights: int, blob_len: int) -> float:
    """Against the packed 2-bit representation, not against fp32."""
    raw_bytes = n_weights * RAW_BITS_PER_WEIGHT / 8
    return raw_bytes / max(1, blob_len)


def bytes_per_flip(n_flips: int, blob_len: int) -> float:
    return blob_len / max(1, n_flips)
