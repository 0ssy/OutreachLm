"""Run the language-model routing probe on the real corpus."""
from __future__ import annotations

import torch

from src.sparse_engine.lm_routing_probe import load_corpus, run

torch.set_num_threads(6)
SEEDS = (0, 1)
K = 2


def prepare():
    text = load_corpus()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    cut = int(0.9 * len(data))
    return (data[:cut], data[cut:], len(chars)), len(text)


def agg(payload, **kw) -> float:
    vals = [run(seed=s, data=payload, **kw) for s in SEEDS]
    return sum(vals) / len(vals)


def main() -> None:
    payload, n_chars = prepare()
    train_d, val_d, vocab = payload
    print(f"Real corpus: {n_chars:,} chars, vocab {vocab}, "
          f"train {len(train_d):,} / val {len(val_d):,}")
    print(f"k={K} active experts per token in EVERY arm, so compute per token")
    print("is identical and only the pool size varies. Held-out loss, 2 seeds.")
    print("Expert width shrinks as the pool grows, so total parameters stay")
    print("comparable and the comparison is routing, not capacity.\n")

    print("A. POOL SIZE at fixed active count")
    print(f"{'experts':>9}{'ratio':>8}{'d_ff':>7}{'val loss':>11}"
          f"{'vs 8':>9}")
    print("-" * 44)
    ref = None
    for e in (8, 32, 128, 512):
        v = agg(payload, n_experts=e, k=K)
        if ref is None:
            ref = v
        print(f"{e:>9}{e // K:>7}x"
              f"{max(4, (4 * 96) // int(e ** 0.5)):>7}{v:>11.4f}"
              f"{v / ref:>9.3f}")

    print("\nB. SHARDING -- each token may route only within its own half")
    print(f"{'experts':>9}{'shards':>8}{'reachable':>11}{'val loss':>11}"
          f"{'vs full':>9}")
    print("-" * 48)
    for e in (128, 512):
        full = agg(payload, n_experts=e, k=K)
        sh = agg(payload, n_experts=e, k=K, shards=2)
        print(f"{e:>9}{1:>8}{e:>11}{full:>11.4f}{1.0:>9.3f}")
        print(f"{e:>9}{2:>8}{e // 2:>11}{sh:>11.4f}{sh / full:>9.3f}")


if __name__ == "__main__":
    main()
