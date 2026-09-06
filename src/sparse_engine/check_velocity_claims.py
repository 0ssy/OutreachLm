"""Checking the four proposed accelerations against measured facts.

  1. Pre-staging / pipeline overlap: 3-4 days.
  2. BF16 latent halves a 70 GB buffer to 35 GB: 2-3 days.
  3. Sequence packing removes padding waste: 3-5 days.
  4. Stream-and-delete decouples storage from dataset size.

Claims 1 and 4 are structural and can be checked arithmetically against the
current budget. Claim 2 rests on a stated baseline that can be verified
directly. Claim 3 is data-dependent, so it is measured on the actual corpus
rather than assumed.
"""
from __future__ import annotations

from pathlib import Path

from src.sparse_engine.budget import (
    DISK_FREE_GB,
    Config,
    SECONDS_PER_DAY,
)

CORPUS = Path(r"C:\Users\josep\OneDrive\Desktop\OutreachLM")

TARGET = Config(
    total_params=70e9, active_params_per_token=70e6, tokens=1.4e9,
    expert_params=1e6, tokens_per_step=65536, nodes=2,
    strategy="data_parallel_delta",
)


def claim_1_overlap() -> None:
    print("CLAIM 1 -- pipeline overlap of streaming with compute\n")
    c = TARGET
    comp = c.compute_seconds / SECONDS_PER_DAY
    io = c.io_seconds / SECONDS_PER_DAY
    comm = c.comm_seconds / SECONDS_PER_DAY
    print(f"  current model ADDS the phases: "
          f"{comp:.1f} + {io:.1f} + {comm:.1f} = {c.days:.1f} d")
    overlapped = max(comp, io + comm)
    print(f"  perfect overlap would give max(compute, io+comm) = "
          f"{overlapped:.1f} d")
    print(f"  ceiling on this optimisation: {c.days - overlapped:.1f} days")
    print("\n  VERDICT: real, and the claim of 3-4 days is close to the")
    print("  arithmetic ceiling. The budget should model overlap rather than")
    print("  summing, since double-buffered streaming is standard and the")
    print("  Phase L work already built it.")


def claim_2_bf16() -> None:
    print("\n\nCLAIM 2 -- BF16 latent halves the 70 GB buffer to 35 GB\n")
    p = 70e9
    print(f"{'latent dtype':>16}{'bytes/param':>13}{'latent GB':>12}"
          f"{'+ternary':>11}{'fits 119.5':>12}")
    print("-" * 64)
    for name, b in (("int8 (current)", 1), ("fp16/bf16", 2), ("fp32", 4)):
        gb = p * b / 1e9
        total = gb + 17.5
        print(f"{name:>16}{b:>13}{gb:>12.1f}{total:>11.1f}"
              f"{'yes' if total < DISK_FREE_GB else 'NO':>12}")
    print("\n  The claim's baseline is inverted. This budget already stores")
    print("  the latent master as int8, which is where the 70 GB figure comes")
    print("  from -- it is not an FP32 map. Moving to BF16 DOUBLES it to")
    print("  140 GB, and 140 + 17.5 = 157.5 GB does not fit in 119.5 GB free.")
    print("\n  Separately: this CPU is Zen 3 (AVX2), with no AVX512-BF16, so")
    print("  BF16 buys no arithmetic acceleration here either -- it would be")
    print("  storage-only, and the storage moves the wrong way.")


def claim_3_packing() -> None:
    print("\n\nCLAIM 3 -- sequence packing removes padding waste\n")
    sizes = []
    for f in sorted(CORPUS.rglob("*.txt")):
        try:
            sizes.append(f.stat().st_size)
        except OSError:
            pass
    if not sizes:
        print("  corpus not found")
        return
    total = sum(sizes)
    print(f"  {len(sizes):,} files, {total / 1e6:.2f} MB, "
          f"mean {total / len(sizes):,.0f} B, "
          f"median {sorted(sizes)[len(sizes) // 2]:,} B")
    print(f"\n{'block':>8}{'padded tokens':>16}{'waste':>9}{'days saved':>12}")
    print("-" * 45)
    for block in (256, 1024, 4096, 65536):
        padded = sum(-(-s // block) * block for s in sizes)
        waste = 1.0 - total / padded
        saved = TARGET.days * waste
        print(f"{block:>8}{padded:>16,}{waste:>8.1%}{saved:>12.1f}")
    print("\n  Waste is real and rises with block size, because these files")
    print("  are small relative to a 65,536-token block.")


def claim_4_streaming() -> None:
    print("\n\nCLAIM 4 -- stream and delete decouples storage from dataset\n")
    c = TARGET
    print(f"  ternary weights      {c.weight_gb:>7.1f} GB  (static)")
    print(f"  int8 latent master   {c.latent_gb:>7.1f} GB  (static)")
    print(f"  data window          {2.0:>7.1f} GB  (recycled)")
    print(f"  total footprint      {c.weight_gb + c.latent_gb + 2:>7.1f} GB "
          f"of {DISK_FREE_GB} free")
    text_gb = c.tokens * 4 / 1e9
    print(f"\n  VERDICT: correct, and it matters. 1.4e9 tokens is ~"
          f"{text_gb:.1f} GB of text at 4 bytes/token, which would not need")
    print("  to be resident at any moment.")
    print("\n  The real consequence is not storage, it is EPOCHS: deleting a")
    print("  chunk after one pass means single-epoch training, so the corpus")
    print(f"  must actually supply {text_gb:.1f} GB. The local corpus is "
          f"0.003 GB, i.e. {text_gb / 0.0032:,.0f}x short.")


def main() -> None:
    claim_1_overlap()
    claim_2_bf16()
    claim_3_packing()
    claim_4_streaming()


if __name__ == "__main__":
    main()
