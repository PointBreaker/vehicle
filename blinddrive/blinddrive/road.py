"""Random road generation and road geometry queries.

A road is a smooth centerline sampled every ``sample_spacing`` metres plus a
constant width. The centerline is built from a random curvature profile
(straights, gentle curves, sharp curves) that is smoothed so curvature changes
gradually, then integrated into x/y points.

The full Road object lives only inside the environment. Controllers never get it.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .config import RoadConfig


class RoadGenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Projection:
    s: float          # arc length of the closest centerline point
    lateral: float    # signed distance from the centerline, + = left of travel direction
    heading: float    # road heading at s


@dataclass(frozen=True)
class Road:
    seed: int
    width: float
    length: float          # finish line (arc length)
    spacing: float
    xs: tuple[float, ...]
    ys: tuple[float, ...]
    headings: tuple[float, ...]
    curvatures: tuple[float, ...]

    @property
    def total_length(self) -> float:
        return self.spacing * (len(self.xs) - 1)

    def pose_at(self, s: float) -> tuple[float, float, float]:
        """Interpolated (x, y, heading) at arc length s (clamped to the road)."""
        s = min(max(s, 0.0), self.total_length)
        i = min(int(s / self.spacing), len(self.xs) - 2)
        t = (s - i * self.spacing) / self.spacing
        x = self.xs[i] + t * (self.xs[i + 1] - self.xs[i])
        y = self.ys[i] + t * (self.ys[i + 1] - self.ys[i])
        h = self.headings[i] + t * _wrap(self.headings[i + 1] - self.headings[i])
        return x, y, h

    def project(self, x: float, y: float, s_hint: float, window: float) -> Projection:
        """Closest point on the centerline within ``s_hint +- window``.

        The search is local so that a nearby but distant part of the road
        (e.g. the other leg of a hairpin) is never mistaken for the current one.
        The first and last segments are extended as lines.
        """
        n = len(self.xs)
        i0 = max(0, int((s_hint - window) / self.spacing))
        i1 = min(n - 2, int(math.ceil((s_hint + window) / self.spacing)))
        best_d2 = math.inf
        best = (0.0, 0.0, 0.0)
        for i in range(i0, i1 + 1):
            ax, ay = self.xs[i], self.ys[i]
            dx, dy = self.xs[i + 1] - ax, self.ys[i + 1] - ay
            seg2 = dx * dx + dy * dy
            t = ((x - ax) * dx + (y - ay) * dy) / seg2
            if i > 0:
                t = max(t, 0.0)
            if i < n - 2:
                t = min(t, 1.0)
            px, py = ax + t * dx, ay + t * dy
            d2 = (x - px) ** 2 + (y - py) ** 2
            if d2 < best_d2:
                best_d2 = d2
                cross = dx * (y - py) - dy * (x - px)
                lateral = math.copysign(math.sqrt(d2), cross) if d2 > 0 else 0.0
                best = ((i + t) * self.spacing, lateral, math.atan2(dy, dx))
        return Projection(*best)

    def sample(self, s_start: float, s_end: float, step: float) -> list[tuple[float, float, float]]:
        """Points (x, y, s) from s_start to s_end inclusive, every ``step`` metres."""
        s_start = max(0.0, s_start)
        s_end = min(self.total_length, s_end)
        out = []
        k = 0
        while True:
            s = s_start + k * step
            if s >= s_end - 1e-9:
                break
            x, y, _ = self.pose_at(s)
            out.append((x, y, s))
            k += 1
        x, y, _ = self.pose_at(s_end)
        out.append((x, y, s_end))
        return out


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


# ------------------------------------------------------------------ generation

def generate_road(seed: int, cfg: RoadConfig | None = None) -> Road:
    """Generate the full road for ``seed``. Same seed + config => identical road."""
    if cfg is None:
        cfg = RoadConfig()
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"seed must be an int, got {seed!r}")
    rng = random.Random(seed)
    n = int(round((cfg.length + cfg.runout) / cfg.sample_spacing)) + 1
    window = max(1, int(round(cfg.smoothing_window / cfg.sample_spacing)))

    for _ in range(cfg.max_attempts):
        kappa = _curvature_profile(rng, cfg, n)
        kappa = _moving_average(_moving_average(kappa, window), window)
        xs, ys, hs = _integrate(kappa, cfg.sample_spacing)
        if not _self_intersects(xs, ys, cfg):
            return Road(
                seed=seed,
                width=cfg.width,
                length=cfg.length,
                spacing=cfg.sample_spacing,
                xs=tuple(xs),
                ys=tuple(ys),
                headings=tuple(hs),
                curvatures=tuple(kappa),
            )
    raise RoadGenerationError(f"no valid road for seed {seed} after {cfg.max_attempts} attempts")


def _curvature_profile(rng: random.Random, cfg: RoadConfig, n: int) -> list[float]:
    ds = cfg.sample_spacing
    kappa = [0.0] * int(round(cfg.start_straight / ds))
    heading = 0.0  # net heading change so far, used to keep the road going "forward"
    bias_scale = math.radians(cfg.heading_bias_scale_deg)
    max_dev = math.radians(cfg.max_heading_deviation_deg)

    while len(kappa) < n:
        kind = rng.choices(("straight", "gentle", "sharp"), weights=cfg.segment_weights)[0]
        if kind == "straight":
            kappa.extend([0.0] * max(1, int(rng.uniform(*cfg.straight_length) / ds)))
            continue
        radius_range, angle_range = (
            (cfg.gentle_radius, cfg.gentle_angle_deg) if kind == "gentle"
            else (cfg.sharp_radius, cfg.sharp_angle_deg)
        )
        radius = rng.uniform(*radius_range)
        angle = math.radians(rng.uniform(*angle_range))
        # Turning further away from the start direction becomes less likely.
        p_left = 1.0 / (1.0 + math.exp(heading / bias_scale))
        sign = 1.0 if rng.random() < p_left else -1.0
        if abs(heading + sign * angle) > max_dev:
            sign = -sign
        heading += sign * angle
        kappa.extend([sign / radius] * max(1, int(angle * radius / ds)))
    return kappa[:n]


def _moving_average(values: list[float], window: int) -> list[float]:
    half = window // 2
    n = len(values)
    prefix = [0.0]
    for v in values:
        prefix.append(prefix[-1] + v)
    out = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out.append((prefix[hi] - prefix[lo]) / (hi - lo))
    return out


def _integrate(kappa: list[float], ds: float) -> tuple[list[float], list[float], list[float]]:
    xs, ys, hs = [0.0], [0.0], [0.0]
    for i in range(1, len(kappa)):
        h_prev = hs[-1]
        h = h_prev + 0.5 * (kappa[i - 1] + kappa[i]) * ds
        mid = 0.5 * (h_prev + h)
        xs.append(xs[-1] + ds * math.cos(mid))
        ys.append(ys[-1] + ds * math.sin(mid))
        hs.append(h)
    return xs, ys, [_wrap(h) for h in hs]


def _self_intersects(xs: list[float], ys: list[float], cfg: RoadConfig) -> bool:
    clearance = cfg.min_clearance_widths * cfg.width
    step = max(1, int(round(1.0 / cfg.sample_spacing)))  # check every ~1 m
    idx = list(range(0, len(xs), step))
    # Points closer than this along the road are "adjacent" and may be close in space.
    min_gap = math.pi * clearance
    c2 = clearance * clearance
    ds = cfg.sample_spacing
    for a_pos, a in enumerate(idx):
        ax, ay = xs[a], ys[a]
        for b in idx[a_pos + 1:]:
            if (b - a) * ds < min_gap:
                continue
            if (xs[b] - ax) ** 2 + (ys[b] - ay) ** 2 < c2:
                return True
    return False
