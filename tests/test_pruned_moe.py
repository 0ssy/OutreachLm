"""Tests for banked pruning wired into a real routed MoE.

Isolated tests of the pruner cannot catch integration failures: the router's
affinity scale drifts, gradient magnitudes change as training proceeds, and
the admitted set alters what the router observes next. These close that loop.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sparse_engine.pruned_moe import (
    PrunedRouter,
    expert_grad_norms,
)

D, E, ADMIT, N = 16, 64, 8, 96


class TinyMoE(nn.Module):
    def __init__(self, d=D, n_experts=E, k=2, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.router = nn.Linear(d, n_experts)
        self.w_in = nn.Parameter(
            torch.randn(n_experts, d, d, generator=g) / d**0.5
        )
        self.w_out = nn.Parameter(
            torch.randn(n_experts, d, d, generator=g) / d**0.5
        )
        self.k = k

    def forward(self, x, router: PrunedRouter | None):
        logits = self.router(x)
        masked = router.mask_logits(logits) if router else logits
        topv, topi = torch.topk(masked, self.k, dim=-1)
        gate = F.softmax(topv, dim=-1)
        out = torch.zeros_like(x)
        for j in range(self.k):
            e = topi[:, j]
            z = torch.bmm(x.unsqueeze(1), self.w_in[e]).relu()
            out = out + gate[:, j:j + 1] * torch.bmm(z, self.w_out[e]
                                                     ).squeeze(1)
        return out, logits


def _train(steps=400, admit=ADMIT, max_staleness=None, seed=0):
    torch.manual_seed(seed)
    model = TinyMoE(seed=seed)
    router = PrunedRouter(E, admit, refresh=8, max_staleness=max_staleness)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(seed + 5)
    target = torch.randn(N, D, generator=g)
    x = torch.randn(N, D, generator=g)
    for _ in range(steps):
        out, logits = model(x, router)
        loss = F.mse_loss(out, target)
        opt.zero_grad()
        loss.backward()
        gn = expert_grad_norms(model.w_in, model.w_out, router.admitted)
        opt.step()
        router.step(logits, gn)
    return model, router, float(loss.detach())


def test_only_admitted_experts_receive_tokens():
    model = TinyMoE()
    router = PrunedRouter(E, ADMIT)
    x = torch.randn(N, D)
    logits = model.router(x)
    masked = router.mask_logits(logits)
    chosen = torch.topk(masked, 2, dim=-1).indices.unique()
    assert set(chosen.tolist()).issubset(set(router.admitted.tolist()))


def test_admitted_set_actually_changes_over_training():
    """If the set never moves, banking is doing nothing and the mechanism is
    silently inert."""
    _, router, _ = _train()
    assert router.pruner.updates.sum() > 0
    assert int((router.pruner.updates > 0).sum()) > ADMIT


def test_pruned_experts_are_served_eventually_not_frozen():
    """The integrated version of the freeze test. Naive magnitude pruning
    left 95.8% of experts permanently unserved."""
    _, router, _ = _train(steps=800)
    frozen = router.never_served / E
    assert frozen < 0.5, frozen


def test_age_bound_guarantees_every_expert_is_served():
    """The bound is a ceiling on staleness, not evidence that forcing was
    required. Banking alone often suffices, in which case forced_admissions
    stays 0 and that is the desired outcome -- the guarantee still holds."""
    _, router, _ = _train(steps=800, max_staleness=200)
    assert router.never_served == 0
    assert int(router.pruner.staleness().max()) <= 200 + router.refresh


def test_age_bound_fires_when_banking_alone_would_starve():
    """A bound tight enough to bite must actually force admissions, otherwise
    the mechanism is inert and the guarantee is accidental."""
    torch.manual_seed(0)
    router = PrunedRouter(E, 2, refresh=1, max_staleness=E // 2)
    g = torch.Generator().manual_seed(1)
    # Affinity concentrated on a handful of experts: without forcing, the
    # rest would wait far beyond the bound.
    logits = torch.full((4, E), -8.0)
    logits[:, :3] = 8.0
    for _ in range(400):
        router.step(logits, torch.full((len(router.admitted),), 1.0))
    assert router.forced_admissions > 0
    assert int(router.pruner.staleness().max()) <= E // 2 + router.refresh


def test_router_affinity_is_available_for_pruned_experts():
    """The reason banking is free here: the router scores every expert, it is
    the expert COMPUTE that is restricted. If affinity were only defined on
    the admitted set there would be nothing to bank."""
    model = TinyMoE()
    router = PrunedRouter(E, ADMIT)
    x = torch.randn(N, D)
    _, logits = model(x, router)
    assert logits.shape[-1] == E
    assert torch.isfinite(logits).all()


def test_grad_norms_are_zero_before_backward_not_stale():
    model = TinyMoE()
    router = PrunedRouter(E, ADMIT)
    gn = expert_grad_norms(model.w_in, model.w_out, router.admitted)
    assert gn.shape == (ADMIT,)
    assert float(gn.abs().sum()) == 0.0


def test_pruning_still_learns():
    """Sanity: the loop must not merely be well-behaved, it must train."""
    _, _, final = _train(steps=400)
    _, _, early = _train(steps=20)
    assert final < early


def test_admit_larger_than_pool_is_rejected():
    with pytest.raises(ValueError):
        PrunedRouter(8, 16)
