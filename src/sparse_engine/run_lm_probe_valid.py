"""The valid version of the language routing probe.

Two defects invalidated the first run, both measured rather than suspected:

  1. NOT CONVERGED. Both arms were still falling at 3600 steps, so a
     comparison taken at 700 was premature -- the same floor error that made
     the synthetic probe look flat until it was trained past it.

  2. EXPERTS WERE NOT LOAD-BEARING. With a residual path around the expert
     block, zeroing and freezing every expert cost only 0.0955 of 2.46 nats
     (3.9%). A pool comparison on that model measures the embedding and head,
     not routing, and no pool size could have made a difference.

Both are fixed here: the residual bypass is removed so the expert block is the
only path from context to prediction, expert width shrinks more slowly with
pool size so the arms stay comparable, and the run is long enough to report a
trajectory rather than a single point.
"""
from __future__ import annotations

import torch

from src.sparse_engine.lm_routing_probe import load_corpus, run

torch.set_num_threads(6)
STEPS = 2500
K = 2


def prepare():
    text = load_corpus()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    cut = int(0.9 * len(data))
    return (data[:cut], data[cut:], len(chars))


def main() -> None:
    payload = prepare()
    _, _, vocab = payload
    print(f"Valid LM routing probe. vocab {vocab}, {STEPS} steps,")
    print("expert block is the ONLY path to the head (no residual bypass),")
    print(f"k={K} active per token in every arm. Held-out loss.\n")

    print("A. POOL SIZE at fixed active count")
    print(f"{'experts':>9}{'ratio':>8}{'val loss':>11}{'vs 8':>9}")
    print("-" * 37)
    ref = None
    for e in (8, 64, 512):
        v = run(e, k=K, steps=STEPS, residual=False, data=payload, seed=0)
        if ref is None:
            ref = v
        print(f"{e:>9}{e // K:>7}x{v:>11.4f}{v / ref:>9.4f}")

    print("\nB. SHARDING at the same setting")
    print(f"{'experts':>9}{'shards':>8}{'reachable':>11}{'val loss':>11}"
          f"{'vs full':>9}")
    print("-" * 48)
    for e in (512,):
        full = run(e, k=K, steps=STEPS, residual=False, data=payload, seed=0)
        sh = run(e, k=K, shards=2, steps=STEPS, residual=False,
                 data=payload, seed=0)
        print(f"{e:>9}{1:>8}{e:>11}{full:>11.4f}{1.0:>9.4f}")
        print(f"{e:>9}{2:>8}{e // 2:>11}{sh:>11.4f}{sh / full:>9.4f}")


if __name__ == "__main__":
    main()
