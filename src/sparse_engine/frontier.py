"""The compute frontier: which (active_params, tokens) pairs land under a deadline.

This exists because the same three candidate recipes keep getting quoted with
numbers computed at MODEL_SLACK = 1.63.  The residual was raised to 1.92 after
in-regime validation showed the model 11-25% optimistic, which moves every row.
Rather than re-deriving by hand each time, ask the budget model.

Nothing here introduces new constants.  Every number is Config.days from
budget.py, so the frontier cannot drift away from the cost model.
"""

from __future__ import annotations

from dataclasses import dataclass

from .budget import Config, MODEL_SLACK

# The 70B-capacity reference point.  Capacity is held fixed across every row in
# this module -- only the work per token and the number of tokens vary.
TOTAL_PARAMS = 70e9
EXPERT_PARAMS = 1e6
NODES = 2
STRATEGY = "expert_sharded"


def recipe(active_params: float, tokens: float) -> Config:
    """A two-node, expert-sharded 70B-capacity config at a given work level."""
    return Config(
        total_params=TOTAL_PARAMS,
        active_params_per_token=active_params,
        tokens=tokens,
        expert_params=EXPERT_PARAMS,
        tokens_per_step=4194304,
        micro_batch=65536,
        nodes=NODES,
        strategy=STRATEGY,
    )


def days(active_params: float, tokens: float) -> float:
    return recipe(active_params, tokens).days


@dataclass(frozen=True)
class Row:
    name: str
    active_params: float
    tokens: float

    @property
    def days(self) -> float:
        return days(self.active_params, self.tokens)

    @property
    def work_fraction(self) -> float:
        """Share of the baseline's 6 * P_active * T arithmetic."""
        return (self.active_params * self.tokens) / (70e6 * 1.4e9)


OPTIONS = (
    Row("Baseline (full size)", 70e6, 1.4e9),
    Row("A  sparse activation shift", 28e6, 1.4e9),
    Row("B  curated dataset shift", 70e6, 0.56e9),
    Row("C  hybrid balance", 45e6, 0.88e9),
)


def tokens_for_deadline(active_params: float, target_days: float,
                        lo: float = 1e6, hi: float = 1.4e9) -> float:
    """Largest token budget that still finishes within target_days.

    Bisection rather than algebra: total_seconds is not a clean closed form in
    tokens (throughput depends on tokens-per-expert, and I/O overlaps compute),
    so we ask the model instead of assuming proportionality.
    """
    if days(active_params, lo) > target_days:
        return 0.0
    if days(active_params, hi) <= target_days:
        return hi
    for _ in range(60):
        mid = (lo + hi) / 2
        if days(active_params, mid) <= target_days:
            lo = mid
        else:
            hi = mid
    return lo


def sandbox_days(corpus_bytes: float, bytes_per_token: float = 4.0,
                 active_params: float = 70e6) -> tuple[float, float]:
    """Wall-clock for a single epoch over a local corpus. Returns (tokens, days)."""
    tokens = corpus_bytes / bytes_per_token
    return tokens, days(active_params, tokens)


def main() -> None:
    print(f"Frontier at MODEL_SLACK = {MODEL_SLACK}")
    print(f"{NODES} nodes, {STRATEGY}, {TOTAL_PARAMS/1e9:.0f}B capacity held fixed\n")

    print(f"{'option':<30}{'active':>10}{'tokens':>10}{'work':>8}{'days':>9}")
    for row in OPTIONS:
        print(f"{row.name:<30}{row.active_params/1e6:>9.0f}M"
              f"{row.tokens/1e9:>9.2f}B{row.work_fraction:>8.2f}"
              f"{row.days:>9.1f}")

    print("\nToken budget that actually lands on 16.0 days:")
    for active in (70e6, 45e6, 28e6):
        t = tokens_for_deadline(active, 16.0)
        text_gb = t * 4.0 / 1e9
        print(f"  {active/1e6:>3.0f}M active -> {t/1e9:.3f}B tokens "
              f"({text_gb:.2f} GB of text)")

    print("\nLocal 3.2 MB corpus, single epoch at 70M active:")
    tokens, d = sandbox_days(3.2e6)
    print(f"  {tokens/1e6:.2f}M tokens -> {d*24*3600:.0f} s "
          f"({d*24:.2f} hours)")


if __name__ == "__main__":
    main()
