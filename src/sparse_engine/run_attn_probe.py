"""The decisive test: does a context-carrying trunk let routing pay off?

Prior state of the evidence:
  * On a synthetic task where routing IS the bottleneck by construction,
    larger pools win clearly and the gap widens with training (relative loss
    0.00018 / 0.00007 / 0.00005 at 128 / 512 / 2048 experts).
  * On real text with a MEAN-POOLED trunk, pool size made no difference at
    all: 2.4724 / 2.4736 / 2.4729 at 8 / 64 / 512 experts.

The hypothesis for that gap is that mean-pooling destroys the contextual
signal a router needs -- "river bank" and "investment bank" arrive at the
router looking the same, so no pool size can help. This replaces the trunk
with causal linear attention and repeats the comparison.

Controls, so the only thing varying is the pool:
  * k=2 active experts per token in every arm -> identical compute per token
  * EXPERT WIDTH HELD FIXED. The previous probe shrank d_ff as the pool grew,
    which made small-pool arms 3x more expensive and confounded routing with
    per-expert capacity.
  * same corpus, steps, batch, schedule, seeds
  * held-out loss, because larger pools hold more parameters and training
    loss would reward memorisation

A trajectory is reported rather than a single endpoint: every earlier null in
this project turned out to be an unconverged reading.
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from src.sparse_engine.attn_routing_probe import AttnMoE
from src.sparse_engine.lm_routing_probe import load_corpus, make_batches
from src.sparse_engine.sparse_optim import SparseExpertAdam

torch.set_num_threads(6)
MARKS = (500, 1400, 2600)
K = 2
D = 96
CONTEXT = 64
BATCH = 32


def prepare():
    text = load_corpus()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    cut = int(0.9 * len(data))
    return data[:cut], data[cut:], len(chars)


def train(n_experts, train_d, val_d, vocab, *, shards=1, seed=0, lr=3e-3,
          bal_weight=1e-2, expert_accum=8):
    torch.manual_seed(seed)
    model = AttnMoE(vocab, D, n_experts, K, CONTEXT, shards=shards, seed=seed)
    # Expert tensors get an optimiser whose cost scales with TOUCHED experts.
    # A dense AdamW over the pool made the 512-expert arm 4x slower than the
    # 8-expert one (1340.6 vs 319.1 ms/step) purely in optimiser traffic --
    # measuring optimiser cost instead of routing.
    expert_params = [model.w_in, model.w_out]
    expert_ids = {id(p) for p in expert_params}
    trunk_params = [p for p in model.parameters() if id(p) not in expert_ids]
    opt = torch.optim.AdamW(trunk_params, lr=lr, weight_decay=0.01)
    eopt = SparseExpertAdam(expert_params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(MARKS))
    gen = torch.Generator().manual_seed(seed + 7)
    out = {}
    for t in range(1, max(MARKS) + 1):
        x, y, ix = make_batches(train_d, BATCH, CONTEXT, gen)
        sid = (ix % shards) if shards > 1 else None
        logits, bal = model(x, sid)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab), y.reshape(-1)
        ) + bal_weight * bal
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trunk_params, 1.0)
        opt.step()
        # Expert updates are AMORTISED, exactly as the 70B design amortises
        # latent writes: measured 907.9 ms of a 1409 ms step went to the
        # expert optimiser at 512 experts, because 4096 assignments over 512
        # experts touch essentially all of them. Accumulating gradients and
        # stepping every `expert_accum` steps divides that cost directly.
        if t % expert_accum == 0:
            eopt.set_lr(opt.param_groups[0]["lr"] * expert_accum)
            eopt.step()
            eopt.zero_grad()
        sched.step()
        if t in MARKS:
            model.eval()
            vg = torch.Generator().manual_seed(1234)
            tot = 0.0
            with torch.no_grad():
                for _ in range(16):
                    vx, vy, vix = make_batches(val_d, BATCH, CONTEXT, vg)
                    vsid = (vix % shards) if shards > 1 else None
                    vl, _ = model(vx, vsid)
                    tot += float(F.cross_entropy(
                        vl.reshape(-1, vocab), vy.reshape(-1)
                    ))
            out[t] = tot / 16
            model.train()
    return out


def main() -> None:
    train_d, val_d, vocab = prepare()
    print("Attention-trunk routing probe on the real corpus.")
    print(f"vocab {vocab}, train {len(train_d):,}, val {len(val_d):,}")
    print(f"k={K} active per token in every arm; expert width fixed at "
          f"d_ff={2 * D}.")
    print("Mean-pooled trunk gave 2.4724 / 2.4736 / 2.4729 at 8 / 64 / 512 "
          "experts.")
    print(f"\n{max(MARKS)} steps per arm, 4 arms. Progress is printed as each")
    print("arm finishes, because estimating this from micro-benchmarks has")
    print("missed by 10x twice.\n", flush=True)

    t_start = time.perf_counter()
    results = {}
    for e in (8, 64, 512):
        t0 = time.perf_counter()
        results[e] = train(e, train_d, val_d, vocab)
        el = time.perf_counter() - t0
        print(f"  [{el / 60:5.1f} min] {e:>4} experts done, "
              f"val {results[e][max(MARKS)]:.4f}", flush=True)

    t0 = time.perf_counter()
    sharded = train(512, train_d, val_d, vocab, shards=2)
    print(f"  [{(time.perf_counter() - t0) / 60:5.1f} min] 512 sharded done, "
          f"val {sharded[max(MARKS)]:.4f}", flush=True)
    print(f"  total {(time.perf_counter() - t_start) / 60:.1f} min\n")

    print("A. POOL SIZE with a context-carrying trunk")
    hdr = f"{'experts':>9}{'ratio':>8}" + "".join(
        f"{f'@{m}':>10}" for m in MARKS
    ) + f"{'vs 8':>9}"
    print(hdr)
    print("-" * len(hdr))
    ref = results[8][max(MARKS)]
    for e in (8, 64, 512):
        c = results[e]
        print(f"{e:>9}{e // K:>7}x"
              + "".join(f"{c[m]:>10.4f}" for m in MARKS)
              + f"{c[max(MARKS)] / ref:>9.4f}")

    print("\nB. SHARDING with the same trunk")
    print(f"{'experts':>9}{'shards':>8}{'reachable':>11}{'val loss':>11}"
          f"{'vs full':>9}")
    print("-" * 48)
    # Reuse the arm already trained above rather than retraining it -- the
    # previous version trained 512-full twice, wasting a whole arm.
    full = results[512][max(MARKS)]
    sh = sharded[max(MARKS)]
    print(f"{512:>9}{1:>8}{512:>11}{full:>11.4f}{1.0:>9.4f}")
    print(f"{512:>9}{2:>8}{256:>11}{sh:>11.4f}{sh / full:>9.4f}")


if __name__ == "__main__":
    main()
