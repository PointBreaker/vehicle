import dataclasses
import math

import pytest

from blinddrive.actions import Action, Steering, Throttle
from blinddrive.config import PRESETS, get_preset
from blinddrive.env import BlindDriveEnv
from blinddrive.observation import Observation


def drive(env, action, seconds):
    for _ in range(round(seconds * env.config.sim.physics_hz)):
        if env.done:
            break
        if env.decision_due:
            env.apply_action(action)
        env.tick()


def to_world(p, st):
    c, s = math.cos(st.heading), math.sin(st.heading)
    return (st.x + c * p[0] - s * p[1], st.y + s * p[0] + c * p[1])


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_observation_hides_future_road(preset):
    cfg = get_preset(preset)
    env = BlindDriveEnv(cfg, 7)
    drive(env, Action(Steering.STRAIGHT, Throttle.ACCELERATE), 3.0)
    obs = env.observe()
    dv = env.debug_view()
    road, s_car = dv.road, dv.progress
    horizon = s_car + cfg.sim.lookahead

    assert max(obs.visible_arclengths) <= cfg.sim.lookahead + 1e-9
    assert min(obs.visible_arclengths) >= -cfg.sim.view_behind - 1e-9
    # Every observed point, mapped back to the world, lies on the road at s <= horizon.
    for p, a in zip(obs.visible_centerline, obs.visible_arclengths):
        x, y, _ = road.pose_at(s_car + a)
        wx, wy = to_world(p, dv.state)
        assert math.hypot(wx - x, wy - y) < 1e-6
    # And no observed point coincides with any road point beyond the horizon.
    future = [(x, y) for x, y, s in road.sample(horizon + 1.0, road.total_length, 0.5)]
    for p in obs.visible_centerline:
        wx, wy = to_world(p, dv.state)
        assert all(math.hypot(wx - fx, wy - fy) > 0.25 for fx, fy in future)


def test_observation_is_car_local():
    env = BlindDriveEnv(get_preset("normal"), 3)
    obs = env.observe()
    # At spawn the car sits on the centerline pointing along the road: first
    # forward point is (0, 0) and the start straight runs along +x.
    fwd = [p for p, a in zip(obs.visible_centerline, obs.visible_arclengths) if a >= 0]
    assert fwd[0] == pytest.approx((0.0, 0.0), abs=1e-9)
    assert all(abs(y) < 1e-6 and x >= 0 for x, y in fwd[:20])


def test_observation_holds_only_plain_data():
    """No Road / env reference a controller could use to look ahead."""
    obs = BlindDriveEnv(get_preset("normal"), 3).observe()
    assert dataclasses.is_dataclass(obs) and type(obs).__dataclass_params__.frozen

    def check(v):
        if isinstance(v, tuple):
            for item in v:
                check(item)
        else:
            assert v is None or isinstance(v, (float, int, Action)), type(v)

    for f in dataclasses.fields(Observation):
        check(getattr(obs, f.name))
