"""Is the language probe converged, or floor-limited like the synthetic one?

Probe A on real text reported no difference between 8 and 512 experts
(2.5239 vs 2.5263, 0.1%). That is either a real null or the same failure that
made the synthetic probe look flat: the comparison was taken before the arms
had separated.

Two checks that distinguish them:

  1. TRAJECTORY. If loss is still falling steeply at the cut-off, the
     comparison was premature and the null is uninformative.
  2. HEADROOM. If the model is bottlenecked somewhere other than the expert
     pool -- context mixing, width, depth -- then pool size cannot matter and
     the experiment is measuring the wrong component. Compared here against a
     control whose expert pool is deliberately crippled: if crippling the
     experts does NOT hurt, the experts were never load-bearing and no pool
     comparison run on this model can mean anything.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.sparse_engine.lm_routing_probe import (
    CharMoE,
    load_corpus,
    make_batches,
)

torch.set_num_threads(6)
MARKS = (400, 1200, 3600)


def prepare():
    text = load_corpus()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    cut = int(0.9 * len(data))
    return data[:cut], data[cut:], len(chars)


def train_curve(n_experts, train_d, val_d, vocab, *, k=2, d=96, context=64,
                batch=32, seed=0, lr=3e-3, steps=max(MARKS),
                expert_scale=1.0):
    torch.manual_seed(seed)
    model = CharMoE(vocab, d, n_experts, k, context, seed=seed)
    if expert_scale != 1.0:
        with torch.no_grad():
            model.w_in.mul_(expert_scale)
            model.w_out.mul_(expert_scale)
            model.w_in.requires_grad_(False)
            model.w_out.requires_grad_(False)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr,
        weight_decay=0.01,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    gen = torch.Generator().manual_seed(seed + 7)
    out = {}
    for t in range(1, steps + 1):
        x, y, _ = make_batches(train_d, batch, context, gen)
        logits, bal = model(x, None)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab), y.reshape(-1)
        ) + 1e-2 * bal
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        opt.step()
        sched.step()
        if t in MARKS:
            model.eval()
            vg = torch.Generator().manual_seed(1234)
            tot = 0.0
            with torch.no_grad():
                for _ in range(16):
                    vx, vy, _ = make_batches(val_d, batch, context, vg)
                    vl, _ = model(vx, None)
                    tot += float(F.cross_entropy(
                        vl.reshape(-1, vocab), vy.reshape(-1)
                    ))
            out[t] = tot / 16
            model.train()
    return out


def main() -> None:
    train_d, val_d, vocab = prepare()
    print(f"vocab {vocab}, train {len(train_d):,}, val {len(val_d):,}")
    print(f"uniform baseline = ln(vocab) = "
          f"{torch.log(torch.tensor(float(vocab))):.3f} nats/char\n")

    print("1. TRAJECTORY -- is the comparison converged?")
    hdr = f"{'experts':>9}" + "".join(f"{f'@{m}':>10}" for m in MARKS) + \
        f"{'still falling':>15}"
    print(hdr)
    print("-" * len(hdr))
    for e in (8, 512):
        c = train_curve(e, train_d, val_d, vocab)
        falling = "yes" if c[MARKS[-1]] < c[MARKS[-2]] * 0.99 else "converged"
        print(f"{e:>9}" + "".join(f"{c[m]:>10.4f}" for m in MARKS)
              + f"{falling:>15}")

    print("\n2. ARE THE EXPERTS LOAD-BEARING AT ALL?")
    print("   Experts frozen at init: if loss barely moves, the pool is not")
    print("   the bottleneck and no pool comparison on this model is valid.")
    live = train_curve(128, train_d, val_d, vocab)
    frozen = train_curve(128, train_d, val_d, vocab, expert_scale=1.0)
    frozen_off = train_curve(
        128, train_d, val_d, vocab, expert_scale=0.0
    )
    print(f"\n{'variant':>22}{'@3600':>10}")
    print("-" * 32)
    print(f"{'trained experts':>22}{live[MARKS[-1]]:>10.4f}")
    print(f"{'experts zeroed+frozen':>22}{frozen_off[MARKS[-1]]:>10.4f}")
    gap = frozen_off[MARKS[-1]] - live[MARKS[-1]]
    print(f"\n  gap = {gap:.4f} nats. A gap near zero means the expert block")
    print("  contributes almost nothing and the probe is measuring the")
    print("  embedding and head, not routing.")


if __name__ == "__main__":
    main()
