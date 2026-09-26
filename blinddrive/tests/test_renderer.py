import math

import pytest

from blinddrive.actions import Action, Steering, Throttle
from blinddrive.config import PRESETS, get_preset
from blinddrive.env import BlindDriveEnv, run_episode
from blinddrive.runner import interpolate
from blinddrive.vehicle import VehicleState
from blinddrive.view import Frame


class _Constant:
    def __init__(self, action):
        self.action = action

    def reset(self):
        pass

    def act(self, observation):
        return self.action


def frame(env, **kw):
    dv = env.debug_view()
    base = dict(config=env.config, obs=env.observe(), obs_state=dv.state, camera=dv.state, seed=env.seed,
                controller="test", action=env.current_action, status="running", edge_hits=env.edge_hits)
    base.update(kw)
    return Frame(**base)


@pytest.mark.parametrize("preset", sorted(PRESETS))
@pytest.mark.parametrize("debug", [False, True])
def test_view_draws_every_screen(preset, debug):
    from blinddrive.renderer import TopDownRenderer

    env = BlindDriveEnv(get_preset(preset).with_edge("crash"), 3)
    view = TopDownRenderer(debug=debug)
    dv = env.debug_view() if debug else None
    view.draw(frame(env, status="ready", message="press Enter\nsecond line", debug_view=dv))
    run_episode(env, _Constant(Action(Steering.LEFT, Throttle.FULL_ACCELERATE)))
    view.draw(frame(env, status="over", result=env.result, detail="jev · 80%"))
    view.draw(frame(env, status="error", message="controller failed\nboom"))


def test_view_does_not_draw_beyond_the_observation():
    """In normal mode the view has nothing but the Observation to draw the road from."""
    from blinddrive.renderer import TopDownRenderer

    env = BlindDriveEnv(get_preset("normal"), 3)
    f = frame(env)
    assert f.debug_view is None
    TopDownRenderer().draw(f)


def test_interpolate_wraps_heading():
    a = VehicleState(0, 0, math.pi - 0.1, 10, 0)
    b = VehicleState(1, 0, -math.pi + 0.1, 12, 0.2)
    mid = interpolate(a, b, 0.5)
    assert math.isclose(abs(mid.heading), math.pi, abs_tol=1e-9)
    assert mid.x == 0.5 and mid.speed == 11 and math.isclose(mid.steering_angle, 0.1)
    assert interpolate(a, b, 1.0).x == 1
