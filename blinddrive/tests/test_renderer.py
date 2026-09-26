import math

import pytest

from blinddrive.actions import Action, Steering, Throttle
from blinddrive.config import PRESETS, get_preset
from blinddrive.env import BlindDriveEnv, run_episode
from blinddrive.runner import interpolate
from blinddrive.vehicle import VehicleState


@pytest.mark.parametrize("preset", sorted(PRESETS))
@pytest.mark.parametrize("debug", [False, True])
def test_renderer_draws_every_screen(preset, debug):
    from blinddrive.renderer import Hud, Renderer

    env = BlindDriveEnv(get_preset(preset).with_edge("crash"), 3)
    r = Renderer(env.config, debug=debug)
    hud = Hud(seed=3, controller="test", action=None, seconds_since_decision=1.0, fps=120)
    dv = env.debug_view()
    r.draw(env.observe(), dv.state, dv.state, hud, debug_view=dv, message="hello\nworld")
    run_episode(env, _Constant(Action(Steering.LEFT, Throttle.FULL_ACCELERATE)))
    dv = env.debug_view()
    r.draw(env.observe(), dv.state, dv.state, hud, debug_view=dv, result=env.result)


class _Constant:
    def __init__(self, action):
        self.action = action

    def reset(self):
        pass

    def act(self, observation):
        return self.action


def test_interpolate_wraps_heading():
    a = VehicleState(0, 0, math.pi - 0.1, 10, 0)
    b = VehicleState(1, 0, -math.pi + 0.1, 12, 0.2)
    mid = interpolate(a, b, 0.5)
    assert math.isclose(abs(mid.heading), math.pi, abs_tol=1e-9)
    assert mid.x == 0.5 and mid.speed == 11 and math.isclose(mid.steering_angle, 0.1)
    assert interpolate(a, b, 1.0).x == 1
