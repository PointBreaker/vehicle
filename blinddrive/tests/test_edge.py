"""Wall mode: the road edge is a barrier that costs speed, not the end of the run."""

import dataclasses

import pytest

from blinddrive.actions import Action, Steering, Throttle
from blinddrive.config import get_preset, normal
from blinddrive.env import BlindDriveEnv
from blinddrive.vehicle import footprint, step_vehicle


def drive(env, action, seconds, check=None):
    for _ in range(round(seconds * env.config.sim.physics_hz)):
        if env.done:
            return
        if env.decision_due:
            env.apply_action(action)
        before = env.debug_view().state
        env.tick()
        if check:
            check(env, before, action)


def corners_inside(env, before, action):
    dv = env.debug_view()
    half = dv.road.width / 2
    for c in footprint(dv.state, env.config.vehicle):
        assert abs(dv.road.project(*c, dv.progress, 10).lateral) <= half + 1e-6


def controls_untouched(env, before, action):
    """The wall only moves the car; heading and steering are pure physics."""
    free = step_vehicle(before, action, env.config.vehicle, env.dt)
    after = env.debug_view().state
    assert after.heading == free.heading
    assert after.steering_angle == free.steering_angle
    assert after.speed <= free.speed


def test_default_mode_is_wall():
    assert normal().sim.edge == "wall"


@pytest.mark.parametrize("preset", ["easy", "normal", "hard"])
def test_hitting_the_edge_does_not_end_the_run(preset):
    env = BlindDriveEnv(get_preset(preset), 5)
    drive(env, Action(Steering.HARD_LEFT, Throttle.FULL_ACCELERATE), 8.0,
          check=lambda e, b, a: (corners_inside(e, b, a), controls_untouched(e, b, a)))
    assert not env.done
    assert env.edge_hits >= 1


def test_impact_costs_speed_and_scraping_is_capped():
    env = BlindDriveEnv(normal(), 5)
    drive(env, Action(Steering.STRAIGHT, Throttle.FULL_ACCELERATE), 4.0)
    fast = env.debug_view().state.speed
    assert fast > 12
    sim = env.config.sim
    speeds = []

    def record(e, before, action):
        if e.observe().touching_edge:
            speeds.append((before.speed, e.debug_view().state.speed))

    drive(env, Action(Steering.LEFT, Throttle.FULL_ACCELERATE), 3.0, check=record)
    assert speeds, "never touched the edge"
    first_before, first_after = speeds[0]
    assert first_after <= min(first_before * sim.edge_impact_speed_factor, sim.edge_max_speed) + 1e-9
    assert all(after <= sim.edge_max_speed for _, after in speeds)


def test_can_drive_away_from_the_wall():
    env = BlindDriveEnv(normal(), 5)
    drive(env, Action(Steering.STRAIGHT, Throttle.ACCELERATE), 3.0)
    for _ in range(40):  # drift left until the car scrapes the edge
        drive(env, Action(Steering.SLIGHT_LEFT, Throttle.COAST), 0.25)
        if env.observe().touching_edge:
            break
    assert env.observe().touching_edge
    drive(env, Action(Steering.RIGHT, Throttle.ACCELERATE), 1.5)
    drive(env, Action(Steering.STRAIGHT, Throttle.ACCELERATE), 0.5)
    assert not env.observe().touching_edge
    assert env.edge_hits == 1


def test_cannot_drive_back_past_the_start():
    env = BlindDriveEnv(normal(), 5)
    drive(env, Action(Steering.HARD_LEFT, Throttle.ACCELERATE), 20.0,
          check=lambda e, b, a: corners_inside(e, b, a))
    dv = env.debug_view()
    start_wall = -(env.config.vehicle.length / 2 + 0.5)
    for c in footprint(dv.state, env.config.vehicle):
        assert dv.road.project(*c, dv.progress, 10).s >= start_wall - 1e-6


def test_result_reports_edge_hits():
    cfg = normal()
    cfg = dataclasses.replace(cfg, sim=dataclasses.replace(cfg.sim, timeout=10.0))
    env = BlindDriveEnv(cfg, 5)
    drive(env, Action(Steering.HARD_LEFT, Throttle.FULL_ACCELERATE), 11.0)
    r = env.result
    assert r.termination == "timeout"
    assert r.edge_hits >= 1 and r.edge_contact_time > 0
    assert r.crash_distance is None


def test_invalid_edge_mode_rejected():
    with pytest.raises(ValueError):
        normal().with_edge("bouncy")


def test_wall_episode_replays_exactly(tmp_path):
    from blinddrive.recorder import load_log, verify_log, write_log

    cfg = normal()
    cfg = dataclasses.replace(cfg, sim=dataclasses.replace(cfg.sim, timeout=12.0))
    env = BlindDriveEnv(cfg, 8)
    drive(env, Action(Steering.SLIGHT_LEFT, Throttle.FULL_ACCELERATE), 13.0)
    assert env.result.edge_hits >= 1
    log = load_log(write_log(tmp_path / "wall.jsonl", env, "constant"))
    assert verify_log(log) == []


def test_stuck_car_ends_as_stalled():
    """A car that stops making progress ends the run instead of burning decisions until timeout."""
    env = BlindDriveEnv(normal(), 5)
    drive(env, Action(Steering.HARD_LEFT, Throttle.FULL_ACCELERATE), 60.0)
    assert env.done and env.result.termination == "stalled"
    assert env.elapsed_time < 60
    assert env.result.decision_count < 60 * 4
