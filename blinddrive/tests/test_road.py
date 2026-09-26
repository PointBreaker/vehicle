import math

import pytest

from blinddrive.config import PRESETS, get_preset
from blinddrive.road import generate_road


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_same_seed_same_road(preset):
    cfg = get_preset(preset).road
    assert generate_road(42, cfg) == generate_road(42, cfg)


def test_different_seeds_differ():
    assert generate_road(1).xs != generate_road(2).xs


def test_seed_must_be_int():
    with pytest.raises(TypeError):
        generate_road("42")


@pytest.mark.parametrize("preset", sorted(PRESETS))
@pytest.mark.parametrize("seed", range(8))
def test_road_is_smooth_and_varied(preset, seed):
    cfg = get_preset(preset).road
    road = generate_road(seed, cfg)
    assert road.total_length >= cfg.length + cfg.runout - cfg.sample_spacing
    # Never tighter than the sharpest configured radius; no kinks between samples.
    kmax = max(abs(k) for k in road.curvatures)
    assert kmax <= 1 / cfg.sharp_radius[0] + 1e-9
    for a, b in zip(road.headings, road.headings[1:]):
        assert abs((b - a + math.pi) % (2 * math.pi) - math.pi) < math.radians(3)
    # Curvature itself changes gradually (smoothing): no step changes.
    dk = max(abs(b - a) for a, b in zip(road.curvatures, road.curvatures[1:]))
    assert dk < kmax / 5
    # Has straights and real curves.
    assert any(abs(k) < 1e-6 for k in road.curvatures)
    assert kmax > 1 / cfg.gentle_radius[1]


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_no_self_intersection(preset):
    cfg = get_preset(preset).road
    for seed in range(5):
        road = generate_road(seed, cfg)
        pts = list(zip(road.xs[::4], road.ys[::4]))
        step = road.spacing * 4
        for i, (ax, ay) in enumerate(pts):
            for j in range(i + 1, len(pts)):
                if (j - i) * step > math.pi * 2 * cfg.width:
                    bx, by = pts[j]
                    assert math.hypot(ax - bx, ay - by) >= cfg.width
