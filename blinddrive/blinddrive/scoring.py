"""Scoring: turn episode results into comparable numbers.

Per episode
    effective time  = elapsed time, plus the remaining distance charged at a crawl
                      (PENALTY_SPEED) if the run did not finish. This gives one
                      number for finished, crashed, timed-out and aborted runs.
    score           = 100 * (T_floor - T_eff) / (T_floor - T_top)
                      where T_floor is the crude baseline's time and T_top the
                      full-vision oracle's time on the same seed. 0 = as good as
                      the crude rule, 100 = as fast as the oracle anchor. The oracle
                      is a strong planner, not a proven optimum: > 100 is possible.
                      Scores are normalised per seed, so hard and easy roads count
                      the same, and floored at -100 per episode so a single
                      disastrous run cannot dominate the mean (the finish rate and
                      failure counts are reported separately).

Per run: mean score with a 95% confidence interval, plus separate safety,
latency and cost figures. Nothing is folded in silently.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

SCORING_VERSION = "score-v2"
PENALTY_SPEED = 2.0   # m/s charged for distance not driven
SCORE_FLOOR = -100.0  # per-episode floor, so one disastrous seed cannot swamp the mean


def effective_time(elapsed: float, distance: float, road_length: float, finished: bool) -> float:
    if finished:
        return elapsed
    return elapsed + max(0.0, road_length - distance) / PENALTY_SPEED


def normalized_score(t_eff: float, t_floor: float, t_top: float) -> float | None:
    span = t_floor - t_top
    if span <= 1e-6:
        return None  # degenerate seed: the anchors do not separate
    return max(SCORE_FLOOR, 100.0 * (t_floor - t_eff) / span)


def mean_ci(values: Sequence[float]) -> dict[str, Any]:
    """Mean with a normal-approximation 95% interval (n >= 2)."""
    n = len(values)
    if n == 0:
        return {"mean": None, "ci95": None, "n": 0}
    mean = sum(values) / n
    if n < 2:
        return {"mean": round(mean, 2), "ci95": None, "n": n}
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    half = 1.96 * sd / math.sqrt(n)
    return {"mean": round(mean, 2), "ci95": [round(mean - half, 2), round(mean + half, 2)], "n": n}


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    k = (len(xs) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 1)


def scorecard(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-episode rows (see bench.run_one) into a run scorecard."""
    n = len(rows)
    errors = [r for r in rows if r["status"] == "error"]
    ran = [r for r in rows if r["status"] != "error"]
    finished = [r for r in ran if r["status"] == "finished"]
    scores = [r["score"] for r in rows if r.get("score") is not None]
    latencies = [x for r in rows for x in r.get("latencies_ms", [])]
    mean = lambda xs: round(sum(xs) / len(xs), 2) if xs else None  # noqa: E731
    return {
        "episodes": n,
        "score": mean_ci(scores),
        "reference_score": mean_ci([r["reference_score"] for r in rows if r.get("reference_score") is not None]),
        "finish_rate": round(len(finished) / n, 3) if n else None,
        "crashed": sum(r["status"] == "crashed" for r in ran),
        "stalled": sum(r["status"] == "stalled" for r in ran),
        "timeouts": sum(r["status"] == "timeout" for r in ran),
        "errors": len(errors),
        "mean_finish_time_s": mean([r["time"] for r in finished]),
        "edge_hits_per_episode": mean([r["edge_hits"] for r in ran]),
        "edge_contact_s_per_episode": mean([r["edge_contact_s"] for r in ran]),
        "latency_ms": {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "max": round(max(latencies), 1) if latencies else None,
            "mean": mean(latencies),
        },
        "decisions": sum(r["decisions"] for r in rows),
        "tokens": {
            "input": sum(r.get("tokens_in", 0) for r in rows),
            "output": sum(r.get("tokens_out", 0) for r in rows),
        },
    }
