import pytest

from blinddrive.actions import Action, Steering, Throttle
from blinddrive.config import get_preset, normal
from blinddrive.controllers import HumanController, ReplayController
from blinddrive.controllers.base import Controller
from blinddrive.controllers.human import keys_to_action
from blinddrive.env import BlindDriveEnv, run_episode
from blinddrive.recorder import load_log, verify_log, write_log


class Constant:
    def __init__(self, action):
        self.action = action

    def reset(self):
        pass

    def act(self, observation):
        return self.action


def test_leaving_the_road_crashes_in_crash_mode():
    env = BlindDriveEnv(normal().with_edge("crash"), 5)
    result = run_episode(env, Constant(Action(Steering.HARD_LEFT, Throttle.FULL_ACCELERATE)))
    assert result.termination == "crashed"
    assert not result.success
    assert result.crash_distance is not None and result.crash_distance < 30
    assert result.completion_time is None


def test_crash_uses_footprint_not_centre():
    """Crash happens as soon as a corner leaves the road, while the centre is still on it."""
    env = BlindDriveEnv(normal().with_edge("crash"), 5)
    run_episode(env, Constant(Action(Steering.SLIGHT_LEFT, Throttle.ACCELERATE)))
    assert env.result.termination == "crashed"
    obs = env.observe()
    half = env.config.road.width / 2
    assert abs(obs.lateral_offset) < half


def test_timeout():
    cfg = get_preset("normal")
    cfg = type(cfg)(name=cfg.name, vehicle=cfg.vehicle, road=cfg.road,
                    sim=type(cfg.sim)(timeout=5.0))
    result = run_episode(BlindDriveEnv(cfg, 1), Constant(Action(Steering.STRAIGHT, Throttle.COAST)))
    assert result.termination == "timeout"
    assert result.decision_count == 5 * cfg.sim.decision_hz


def test_same_seed_same_episode(tmp_path):
    actions = [Action(Steering.STRAIGHT, Throttle.FULL_ACCELERATE)] * 8 + \
              [Action(Steering.LEFT, Throttle.ACCELERATE)] * 400
    logs = []
    for _ in range(2):
        env = BlindDriveEnv(normal().with_edge("crash"), 11)
        run_episode(env, ReplayController(actions))
        logs.append(env.decision_log)
    assert logs[0] == logs[1]


def test_log_roundtrip_and_replay(tmp_path):
    env = BlindDriveEnv(normal().with_edge("crash"), 11)
    run_episode(env, Constant(Action(Steering.SLIGHT_RIGHT, Throttle.ACCELERATE)))
    path = write_log(tmp_path / "ep.jsonl", env, "constant")
    log = load_log(path)
    assert log.seed == 11
    assert log.header["controller"] == "constant"
    assert log.config == env.config
    assert len(log.decisions) == env.result.decision_count
    assert set(log.decisions[0]) >= {"time", "position", "heading", "speed", "steering_angle",
                                     "observation", "action"}
    assert verify_log(log) == []
    # Tampering is detected.
    log.decisions[3]["action"] = {"steering": "HARD_LEFT", "throttle": "FULL_ACCELERATE"}
    assert verify_log(log) != []


def test_controllers_share_one_interface():
    assert isinstance(HumanController(lambda: set()), Controller)
    assert isinstance(ReplayController([]), Controller)


def test_human_keys_map_to_standard_actions():
    assert keys_to_action(set()) == Action(Steering.STRAIGHT, Throttle.COAST)
    assert keys_to_action({"up"}) == Action(Steering.STRAIGHT, Throttle.ACCELERATE)
    assert keys_to_action({"up", "shift", "left"}) == Action(Steering.HARD_LEFT, Throttle.FULL_ACCELERATE)
    assert keys_to_action({"space", "up", "right"}) == Action(Steering.RIGHT, Throttle.HARD_BRAKE)
    assert keys_to_action({"left", "right"}).steering is Steering.STRAIGHT
    assert keys_to_action({"slight_right", "down"}) == Action(Steering.SLIGHT_RIGHT, Throttle.BRAKE)


def test_human_decides_only_at_decision_ticks():
    """Key changes between decision ticks do not affect the car."""
    keys = {"now": {"up"}}
    human = HumanController(lambda: keys["now"])
    env = BlindDriveEnv(normal(), 2)
    env.apply_action(human.act(env.observe()))
    keys["now"] = {"space"}  # pressed mid-interval
    for _ in range(env.ticks_per_decision - 1):
        env.tick()
    assert env.current_action.throttle is Throttle.ACCELERATE
    env.tick()
    env.apply_action(human.act(env.observe()))
    assert env.current_action.throttle is Throttle.HARD_BRAKE
