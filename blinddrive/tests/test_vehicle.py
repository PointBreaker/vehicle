import math

from blinddrive.actions import Action, Steering, Throttle
from blinddrive.config import VehicleConfig
from blinddrive.vehicle import VehicleState, step_vehicle, steering_target

CFG = VehicleConfig()
DT = 1 / 60


def still(speed=0.0, steer=0.0):
    return VehicleState(0.0, 0.0, 0.0, speed, steer)


def run(state, action, seconds):
    for _ in range(round(seconds / DT)):
        state = step_vehicle(state, action, CFG, DT)
    return state


def test_steering_is_rate_limited():
    action = Action(Steering.HARD_LEFT, Throttle.COAST)
    target = steering_target(Steering.HARD_LEFT, CFG)
    s1 = step_vehicle(still(10), action, CFG, DT)
    assert 0 < s1.steering_angle < target
    assert math.isclose(s1.steering_angle, math.radians(CFG.max_steering_rate_deg) * DT)
    # Full lock to full lock takes 2 * max_angle / rate seconds.
    s = run(still(10, target), Action(Steering.HARD_RIGHT, Throttle.COAST), 0.5)
    assert s.steering_angle > -target + 1e-6
    s = run(s, Action(Steering.HARD_RIGHT, Throttle.COAST), 0.6)
    assert math.isclose(s.steering_angle, -target)


def test_acceleration_is_gradual():
    s1 = step_vehicle(still(), Action(Steering.STRAIGHT, Throttle.FULL_ACCELERATE), CFG, DT)
    assert 0 < s1.speed <= CFG.max_acceleration * DT
    s = run(still(), Action(Steering.STRAIGHT, Throttle.FULL_ACCELERATE), 3.0)
    assert s.speed < CFG.max_speed / 2


def test_braking_takes_distance():
    v0 = 30.0
    s = still(v0)
    brake = Action(Steering.STRAIGHT, Throttle.HARD_BRAKE)
    while s.speed > 0:
        s = step_vehicle(s, brake, CFG, DT)
    # Can never stop shorter than the physical limit v^2 / (2 * (brake + resistance)).
    max_decel = CFG.max_braking + CFG.rolling_resistance + CFG.drag_coefficient * v0 * v0
    assert s.x >= v0 * v0 / (2 * max_decel) - 0.5
    assert s.speed == 0.0  # no reverse


def test_speed_capped():
    s = run(still(), Action(Steering.STRAIGHT, Throttle.FULL_ACCELERATE), 60)
    assert s.speed <= CFG.max_speed


def test_grip_limits_turning_at_speed():
    """Fast cars cannot follow a tight steering angle (understeer)."""
    hard_left = Action(Steering.HARD_LEFT, Throttle.COAST)
    target = steering_target(Steering.HARD_LEFT, CFG)
    fast = step_vehicle(still(25, target), hard_left, CFG, DT)
    yaw_rate = fast.heading / DT
    assert fast.speed * yaw_rate <= CFG.grip + 1e-6
    assert yaw_rate < 25 * math.tan(target) / CFG.wheelbase


def test_braking_reduces_available_grip():
    target = steering_target(Steering.HARD_LEFT, CFG)
    coast = step_vehicle(still(20, target), Action(Steering.HARD_LEFT, Throttle.COAST), CFG, DT)
    brake = step_vehicle(still(20, target), Action(Steering.HARD_LEFT, Throttle.HARD_BRAKE), CFG, DT)
    assert brake.heading < coast.heading
