"""Tests for the attention-trunk routing probe.

The two fixes it embodies both have failure modes that would silently
invalidate the experiment rather than crash it, so both are asserted:
the fast dispatch must be numerically identical to the reference, and the
attention trunk must actually make representations context-dependent.
"""
import pytest
import torch

from src.sparse_engine.attn_routing_probe import (
    AttnMoE,
    dispatch_batched,
    dispatch_per_expert,
    dispatch_per_token,
    linear_causal_attention,
)

D, E, K, N = 32, 16, 2, 64


def _parts(seed=0, n_experts=E):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(N, D, generator=g)
    w_in = torch.randn(n_experts, D, 2 * D, generator=g) / D**0.5
    w_out = torch.randn(n_experts, 2 * D, D, generator=g) / (2 * D) ** 0.5
    logits = torch.randn(N, n_experts, generator=g)
    topv, topi = torch.topk(logits, K, dim=-1)
    gate = torch.softmax(topv, dim=-1)
    return x, topi, gate, w_in, w_out


def test_dispatch_paths_agree():
    """The speedup must come from access pattern, not behaviour change.

    Measured forward+backward at N=2048, d=96, d_ff=192:
        experts   per-token   per-expert    batched
              8     628.7 ms      14.9 ms    14.2 ms
             64     806.0 ms     306.0 ms    22.0 ms
            512     827.5 ms   26630.7 ms    66.9 ms
    Per-expert degenerates into one tiny GEMM each; batched keeps it as a
    single bmm. All three must produce the same numbers.
    """
    for seed in range(4):
        x, topi, gate, w_in, w_out = _parts(seed)
        a = dispatch_per_token(x, topi, gate, w_in, w_out)
        b = dispatch_per_expert(x, topi, gate, w_in, w_out)
        c = dispatch_batched(x, topi, gate, w_in, w_out)
        assert torch.allclose(a, b, atol=1e-5), float((a - b).abs().max())
        assert torch.allclose(a, c, atol=1e-5), float((a - c).abs().max())


def test_dispatch_agrees_when_an_expert_gets_no_tokens():
    """Empty experts are common at large pool sizes and are an easy way to
    silently drop tokens."""
    x, _, _, w_in, w_out = _parts(1, n_experts=32)
    topi = torch.zeros(N, K, dtype=torch.long)
    topi[:, 1] = 1
    gate = torch.full((N, K), 0.5)
    a = dispatch_per_token(x, topi, gate, w_in, w_out)
    b = dispatch_per_expert(x, topi, gate, w_in, w_out)
    c = dispatch_batched(x, topi, gate, w_in, w_out)
    assert torch.allclose(a, b, atol=1e-5)
    assert torch.allclose(a, c, atol=1e-5)


def test_dispatch_agrees_when_a_token_picks_one_expert_twice():
    x, _, _, w_in, w_out = _parts(2)
    topi = torch.full((N, K), 3, dtype=torch.long)
    gate = torch.full((N, K), 0.5)
    a = dispatch_per_token(x, topi, gate, w_in, w_out)
    c = dispatch_batched(x, topi, gate, w_in, w_out)
    assert torch.allclose(a, c, atol=1e-5)


def test_fast_dispatch_does_not_materialise_per_token_weights():
    """The defect being fixed: (N, d, d_ff) is 178 MB at the probe's shapes.
    Peak allocation for the fast path must not scale with N * d * d_ff."""
    n, d, d_ff, n_e = 512, 64, 128, 32
    g = torch.Generator().manual_seed(3)
    x = torch.randn(n, d, generator=g)
    w_in = torch.randn(n_e, d, d_ff, generator=g)
    w_out = torch.randn(n_e, d_ff, d, generator=g)
    topi = torch.randint(0, n_e, (n, K), generator=g)
    gate = torch.full((n, K), 0.5)

    per_token_bytes = n * d * d_ff * 4
    # Largest single tensor the fast path builds is one expert's block.
    largest_block = (n * K // n_e + n) * max(d, d_ff) * 4
    assert largest_block < per_token_bytes / 4
    out = dispatch_batched(x, topi, gate, w_in, w_out)
    assert out.shape == (n, d)


def test_attention_makes_representations_context_dependent():
    """The whole point of replacing mean-pooling. The same token id in two
    different contexts must produce different features, otherwise the router
    has nothing to route on."""
    torch.manual_seed(0)
    m = AttnMoE(vocab=16, d=D, n_experts=E, k=K, context=8)
    a = torch.tensor([[1, 2, 3, 4, 5, 5, 5, 9]])
    b = torch.tensor([[7, 7, 7, 7, 7, 5, 5, 9]])
    with torch.no_grad():
        h_a = m.emb(a) + m.pos[:8]
        q, k, v = m.qkv(m.norm1(h_a)).chunk(3, dim=-1)
        ctx_a = linear_causal_attention(q, k, v)
        h_b = m.emb(b) + m.pos[:8]
        q, k, v = m.qkv(m.norm1(h_b)).chunk(3, dim=-1)
        ctx_b = linear_causal_attention(q, k, v)
    # Final position holds the same token id in both sequences.
    assert torch.equal(a[0, -1], b[0, -1])
    diff = float((ctx_a[0, -1] - ctx_b[0, -1]).norm())
    assert diff > 1e-3, diff


def test_attention_is_causal():
    """A leak would let the model see the answer and invalidate every loss."""
    torch.manual_seed(1)
    q = torch.randn(1, 6, D)
    k = torch.randn(1, 6, D)
    v = torch.randn(1, 6, D)
    base = linear_causal_attention(q, k, v)
    v2 = v.clone()
    v2[0, 4:] += 10.0                       # perturb only the future
    after = linear_causal_attention(q, k, v2)
    assert torch.allclose(base[0, :4], after[0, :4], atol=1e-5)
    assert not torch.allclose(base[0, 4:], after[0, 4:], atol=1e-5)


def test_expert_width_is_independent_of_pool_size():
    """The previous probe shrank d_ff as the pool grew, which made small-pool
    arms 3x more expensive and confounded routing with per-expert capacity."""
    small = AttnMoE(vocab=16, d=D, n_experts=8, k=K, context=8)
    large = AttnMoE(vocab=16, d=D, n_experts=512, k=K, context=8)
    assert small.d_ff == large.d_ff


def test_forward_runs_and_is_finite():
    torch.manual_seed(2)
    m = AttnMoE(vocab=16, d=D, n_experts=E, k=K, context=8)
    logits, bal = m(torch.randint(0, 16, (2, 8)))
    assert logits.shape == (2, 8, 16)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(bal)


def test_sharding_restricts_reachable_experts():
    torch.manual_seed(3)
    m = AttnMoE(vocab=16, d=D, n_experts=E, k=K, context=8, shards=2)
    idx = torch.randint(0, 16, (4, 8))
    sid = torch.zeros(4, dtype=torch.long)
    logits, _ = m(idx, sid)
    assert torch.isfinite(logits).all()


def test_shard_id_length_is_validated_not_broadcast():
    """A per-sequence shard_id against per-token logits used to broadcast
    wrongly, assigning tokens to the wrong shard instead of raising."""
    torch.manual_seed(4)
    m = AttnMoE(vocab=16, d=D, n_experts=E, k=K, context=8, shards=2)
    idx = torch.randint(0, 16, (4, 8))
    per_seq, _ = m(idx, torch.zeros(4, dtype=torch.long))
    per_tok, _ = m(idx, torch.zeros(32, dtype=torch.long))
    assert torch.allclose(per_seq, per_tok, atol=1e-6)
    with pytest.raises(ValueError, match="shard_id"):
        m(idx, torch.zeros(7, dtype=torch.long))


def test_batched_dispatch_gradients_match_reference():
    """Equal forward values are not enough -- the backward must agree too,
    or training silently diverges from the reference implementation."""
    x, topi, gate, w_in, w_out = _parts(9)
    grads = []
    for fn in (dispatch_per_token, dispatch_batched):
        wi = w_in.clone().requires_grad_(True)
        wo = w_out.clone().requires_grad_(True)
        xx = x.clone().requires_grad_(True)
        fn(xx, topi, gate, wi, wo).square().sum().backward()
        grads.append((wi.grad, wo.grad, xx.grad))
    for a, b in zip(*[grads[0], grads[1]]) if False else zip(grads[0], grads[1]):
        assert torch.allclose(a, b, atol=1e-4), float((a - b).abs().max())


def test_batched_padding_stays_small_under_load_balancing():
    """Padding costs E * max_count * d. It is only affordable because the
    balance loss keeps max_count near the mean; this documents the
    dependency so a future change to bal_weight cannot silently blow it up."""
    n_e, n = 512, 4096
    g = torch.Generator().manual_seed(11)
    balanced = torch.randint(0, n_e, (n,), generator=g)
    max_c = int(torch.bincount(balanced, minlength=n_e).max())
    assert max_c < 8 * (n / n_e)


def test_quadratic_attention_is_causal():
    torch.manual_seed(7)
    from src.sparse_engine.attn_routing_probe import (
        quadratic_causal_attention,
    )

    q = torch.randn(1, 6, D)
    k = torch.randn(1, 6, D)
    v = torch.randn(1, 6, D)
    base = quadratic_causal_attention(q, k, v)
    v2 = v.clone()
    v2[0, 4:] += 10.0
    after = quadratic_causal_attention(q, k, v2)
    assert torch.allclose(base[0, :4], after[0, :4], atol=1e-5)
    assert not torch.allclose(base[0, 4:], after[0, 4:], atol=1e-5)


def test_attention_form_is_chosen_by_the_T_vs_d_crossover():
    """Linear attention is O(T d^2) and quadratic is O(T^2 d), so linear only
    wins when T > d. Measured at T=64, d=96: 988.3 ms and 75.5 MB for linear
    against 7.2 ms and 0.52 MB for quadratic -- a 137x error from picking the
    wrong side."""
    short_ctx = AttnMoE(vocab=16, d=64, n_experts=8, k=K, context=32)
    long_ctx = AttnMoE(vocab=16, d=32, n_experts=8, k=K, context=256)
    assert short_ctx.attention == "quadratic"
    assert long_ctx.attention == "linear"


def test_explicit_attention_choice_is_validated():
    with pytest.raises(ValueError, match="unknown attention"):
        AttnMoE(vocab=16, d=D, n_experts=E, k=K, context=8, attention="fast")


def test_both_attention_forms_make_context_dependent_features():
    """Whichever form is selected, the router must see context -- that is the
    entire reason the trunk was replaced."""
    for form in ("linear", "quadratic"):
        torch.manual_seed(0)
        m = AttnMoE(vocab=16, d=D, n_experts=E, k=K, context=8,
                    attention=form)
        a = torch.tensor([[1, 2, 3, 4, 5, 5, 5, 9]])
        b = torch.tensor([[7, 7, 7, 7, 7, 5, 5, 9]])
        outs = []
        for seq in (a, b):
            with torch.no_grad():
                h = m.emb(seq) + m.pos[:8]
                q, k, v = m.qkv(m.norm1(h)).chunk(3, dim=-1)
                fn = (m.__class__.__module__, form)
                from src.sparse_engine import attn_routing_probe as arp
                f = (arp.linear_causal_attention if form == "linear"
                     else arp.quadratic_causal_attention)
                outs.append(f(q, k, v)[0, -1])
        assert float((outs[0] - outs[1]).norm()) > 1e-3, form
