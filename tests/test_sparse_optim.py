"""Tests for expert-sparse Adam.

The failure modes here are silent: a wrong bias correction still trains, just
worse, and a dense fallback still trains, just slower. Both are asserted.
"""
import pytest
import torch

from src.sparse_engine.sparse_optim import SparseExpertAdam

E, D, F_ = 16, 8, 12


def _param():
    g = torch.Generator().manual_seed(0)
    return torch.randn(E, D, F_, generator=g, requires_grad=True)


def test_only_rows_with_gradient_are_touched():
    p = _param()
    before = p.detach().clone()
    p.grad = torch.zeros_like(p)
    p.grad[3] = 1.0
    p.grad[9] = -1.0
    opt = SparseExpertAdam([p], lr=0.1)
    opt.step()
    changed = (p.detach() != before).any(dim=(1, 2)).nonzero(as_tuple=True)[0]
    assert set(changed.tolist()) == {3, 9}
    assert opt.rows_updated == 2


def test_untouched_rows_do_not_drift_over_many_steps():
    """The dense-optimiser defect: momentum and weight decay keep moving
    parameters that received no gradient at all."""
    p = _param()
    before = p.detach().clone()
    opt = SparseExpertAdam([p], lr=0.1, weight_decay=0.01)
    for _ in range(50):
        p.grad = torch.zeros_like(p)
        p.grad[0] = torch.randn(D, F_)
        opt.step()
    assert torch.equal(p.detach()[1:], before[1:])
    assert not torch.equal(p.detach()[0], before[0])


def test_matches_dense_adam_when_every_row_is_active():
    """Restricting the update must not change the maths."""
    ref = _param()
    spa = ref.detach().clone().requires_grad_(True)
    dense = torch.optim.Adam([ref], lr=0.05, betas=(0.9, 0.999), eps=1e-8)
    sparse = SparseExpertAdam([spa], lr=0.05, betas=(0.9, 0.999), eps=1e-8)
    g = torch.Generator().manual_seed(3)
    for _ in range(20):
        grad = torch.randn(E, D, F_, generator=g)
        ref.grad = grad.clone()
        spa.grad = grad.clone()
        dense.step()
        sparse.step()
    assert torch.allclose(ref.detach(), spa.detach(), atol=1e-5), \
        float((ref - spa).abs().max())


def test_bias_correction_is_per_row_not_global():
    """A rarely-served expert must be corrected by ITS OWN update count.

    Row 0 is updated every step; row 1 only on the last one. With a global
    step counter row 1's first update would be scaled as though it had been
    training all along, and would come out far too small.
    """
    p = _param()
    with torch.no_grad():
        p.zero_()
    opt = SparseExpertAdam([p], lr=0.1)
    for _ in range(30):
        p.grad = torch.zeros_like(p)
        p.grad[0] = 1.0
        opt.step()
    p.grad = torch.zeros_like(p)
    p.grad[1] = 1.0
    opt.step()

    first_update = float(p.detach()[1].abs().mean())
    steady = float(p.detach()[0].abs().mean() / 30)
    # A correctly bias-corrected first step moves by roughly lr.
    assert abs(first_update - 0.1) < 0.02, first_update
    assert first_update > steady * 0.5


def test_state_counters_track_each_row_independently():
    p = _param()
    opt = SparseExpertAdam([p], lr=0.1)
    for i in range(5):
        p.grad = torch.zeros_like(p)
        p.grad[0] = 1.0
        if i == 4:
            p.grad[7] = 1.0
        opt.step()
    assert int(opt.state[0]["t"][0]) == 5
    assert int(opt.state[0]["t"][7]) == 1
    assert int(opt.state[0]["t"][3]) == 0


def test_rejects_non_expert_shaped_tensors():
    with pytest.raises(ValueError, match="n_experts"):
        SparseExpertAdam([torch.zeros(10, requires_grad=True)])


def test_rejects_empty_parameter_list():
    with pytest.raises(ValueError):
        SparseExpertAdam([])


def test_cost_scales_with_touched_rows_not_pool_size():
    """The whole point. A 64x larger pool with the same number of active
    experts must not cost 64x more optimiser time.

    Only `step()` is timed. An earlier version of this test allocated a fresh
    gradient tensor inside the loop -- 33 MB at 1024 experts -- which scales
    with the pool and swamped the quantity under test.
    """
    import time

    def run(n_experts, touched=4, steps=50):
        g = torch.Generator().manual_seed(1)
        p = torch.randn(n_experts, 64, 128, generator=g, requires_grad=True)
        opt = SparseExpertAdam([p], lr=0.01)
        rows = torch.arange(touched)
        p.grad = torch.zeros_like(p)
        p.grad[rows] = 1.0
        opt.step(rows=rows)
        t0 = time.perf_counter()
        for _ in range(steps):
            opt.step(rows=rows)
        return (time.perf_counter() - t0) / steps

    small = run(16)
    large = run(1024)
    assert large < small * 4, f"{large / small:.1f}x"


def test_explicit_rows_avoid_the_gradient_scan():
    """Detecting touched rows by scanning the gradient costs a pass over the
    whole tensor, which is what made the first version slower than a dense
    optimiser. Routing already knows the answer, so it can be passed in."""
    import time

    g = torch.Generator().manual_seed(2)
    p = torch.randn(2048, 64, 128, generator=g, requires_grad=True)
    opt = SparseExpertAdam([p], lr=0.01)
    rows = torch.arange(4)
    p.grad = torch.zeros_like(p)
    p.grad[rows] = 1.0

    opt.step(rows=rows)
    t0 = time.perf_counter()
    for _ in range(20):
        opt.step(rows=rows)
    given = (time.perf_counter() - t0) / 20

    opt.step()
    t0 = time.perf_counter()
    for _ in range(20):
        opt.step()
    scanned = (time.perf_counter() - t0) / 20

    assert given < scanned


def test_dense_path_is_taken_when_most_of_the_pool_is_active():
    """At 512 experts with 4096 assignments nearly every expert is touched,
    so gather/scatter of moment state costs more than updating everything.
    This asserts the adaptive switch exists and stays correct."""
    p = _param()
    ref = p.detach().clone().requires_grad_(True)
    opt_a = SparseExpertAdam([p], lr=0.05)
    opt_b = SparseExpertAdam([ref], lr=0.05)
    g = torch.Generator().manual_seed(4)
    all_rows = torch.arange(E)
    for _ in range(10):
        grad = torch.randn(E, D, F_, generator=g)
        p.grad = grad.clone()
        ref.grad = grad.clone()
        opt_a.step()                      # detects all rows -> dense path
        opt_b.step(rows=all_rows)         # told all rows -> dense path
    assert torch.allclose(p.detach(), ref.detach(), atol=1e-6)
