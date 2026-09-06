"""Tests for the deadline frontier.

The point of these is to stop the frontier drifting away from budget.py, and to
pin the one result that actually decides the recipe: at a fixed deadline, the
low-active-parameter rows are the only ones that are also adequately trained.
"""

from __future__ import annotations

import pytest

from src.sparse_engine.budget import MODEL_SLACK
from src.sparse_engine.frontier import (
    OPTIONS,
    days,
    recipe,
    sandbox_days,
    tokens_for_deadline,
)


def test_frontier_uses_the_current_residual():
    """If MODEL_SLACK moves, these numbers must be regenerated, not trusted."""
    assert MODEL_SLACK == pytest.approx(1.92)


def test_all_options_hold_capacity_fixed():
    for row in OPTIONS:
        assert recipe(row.active_params, row.tokens).total_params == 70e9


def test_equal_work_fractions_do_not_give_equal_days():
    """0.40x the arithmetic is not 0.40x the clock.

    Throughput depends on tokens-per-expert and I/O partly overlaps compute, so
    the three 0.40 rows land on visibly different days.  This is exactly why the
    frontier bisects the model instead of scaling the baseline linearly.
    """
    equal_work = [r for r in OPTIONS if r.name != "Baseline (full size)"]
    assert all(r.work_fraction == pytest.approx(0.40, abs=0.01) for r in equal_work)

    spread = max(r.days for r in equal_work) - min(r.days for r in equal_work)
    assert spread > 0.3, "expected the equal-work rows to differ in wall-clock"


def test_none_of_the_quoted_options_actually_reach_sixteen_days():
    """The 16.9 / 18.0 / 17.5 figures were computed at MODEL_SLACK = 1.63."""
    for row in OPTIONS:
        if row.name == "Baseline (full size)":
            continue
        assert row.days > 16.0


def test_full_active_size_cannot_be_rushed_to_the_deadline():
    """70M active hits 16 days only by going under-trained -- a blocked config."""
    tokens = tokens_for_deadline(70e6, 16.0)
    problems = recipe(70e6, tokens).problems()
    assert any("under-trained" in p for p in problems)


@pytest.mark.parametrize("active", [45e6, 28e6])
def test_reducing_active_parameters_reaches_the_deadline_cleanly(active):
    """Fewer active params buys both the clock and the token ratio."""
    tokens = tokens_for_deadline(active, 16.0)
    config = recipe(active, tokens)
    assert config.days <= 16.0
    assert config.problems() == []
    assert tokens / active >= 15.0


def test_twenty_million_active_beats_the_deadline_on_the_full_token_budget():
    """The one row that needs no data-budget concession at all."""
    config = recipe(20e6, 1.4e9)
    assert config.days < 16.0
    assert config.problems() == []


def test_deadline_is_monotonic_in_work():
    assert days(28e6, 1.4e9) > days(28e6, 0.7e9)
    assert days(70e6, 1.0e9) > days(28e6, 1.0e9)


def test_local_corpus_epoch_is_tens_of_minutes_not_seconds():
    """Guards a 1000x exponent slip: 3.2 MB is a coffee break, not a heartbeat."""
    tokens, d = sandbox_days(3.2e6)
    assert tokens == pytest.approx(0.8e6)
    seconds = d * 24 * 3600
    assert 600 < seconds < 7200, f"expected minutes-scale, got {seconds:.0f} s"
