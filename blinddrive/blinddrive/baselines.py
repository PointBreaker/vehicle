"""Reference controllers that anchor the score scale.

These are measuring sticks, never stand-ins: they are not used as a fallback for
any other controller. Each one uses only the Observation plus public rules.

    crude      "look 10 m ahead and hold 12 m/s". Defines score 0 (the floor).
    reference  pure pursuit + a speed limit that assumes the tightest legal curve
               may start right past the visible range. Finishing with it proves a
               seed is solvable with exactly the information a controller gets.
    oracle     the reference planner run with (almost) unlimited visibility on the
               same road. Its time defines score 100. It is a *score anchor* only:
               it needs a config with a longer lookahead, i.e. privileged vision.
"""

from __future__ import annotations

import dataclasses
import math

from .actions import Action, Steering, Throttle
from .config import GameConfig
from .observation import Observation
from .vehicle import steering_target

ORACLE_LOOKAHEAD = 1000.0     # metres: effectively the whole road


def _curvatures(points: list[tuple[float, float]]) -> list[float]:
    ks = [0.0] * len(points)
    for i in range(1, len(points) - 1):
        (ax, ay), (bx, by), (cx, cy) = points[i - 1], points[i], points[i + 1]
        dh = (math.atan2(cy - by, cx - bx) - math.atan2(by - ay, bx - ax) + math.pi) % (2 * math.pi) - math.pi
        ds = 0.5 * (math.hypot(bx - ax, by - ay) + math.hypot(cx - bx, cy - by))
        ks[i] = dh / ds if ds > 0 else 0.0
    if len(ks) > 1:
        ks[0], ks[-1] = ks[1], ks[-2]
    return ks


def _nearest(options, value, key):
    return min(options, key=lambda o: abs(key(o) - value))


class ReferenceController:
    """Observation-only planner: pure pursuit steering + look-ahead speed limit."""

    name = "reference"

    def __init__(self, config: GameConfig, grip_use: float = 0.8, brake_use: float = 0.8,
                 assume_worst_beyond: bool = True):
        self.v = config.vehicle
        self.grip = grip_use * config.vehicle.grip
        self.brake = brake_use * config.vehicle.max_braking
        self.worst_radius = config.road.sharp_radius[0] if assume_worst_beyond else None
        self.delay = 1.0 / config.sim.decision_hz  # the action is held this long

    def reset(self) -> None:
        pass

    def act(self, obs: Observation) -> Action:
        fwd = [(p, a) for p, a in zip(obs.visible_centerline, obs.visible_arclengths) if a >= 0]
        pts = [p for p, _ in fwd]
        arcs = [a for _, a in fwd]
        v = obs.speed

        # Speed: be able to slow down to each visible curve's grip limit in time.
        limit = self.v.max_speed
        reaction = v * self.delay
        for k, a in zip(_curvatures(pts), arcs):
            v_curve = math.sqrt(self.grip / max(abs(k), 1e-6))
            limit = min(limit, math.sqrt(v_curve ** 2 + 2 * self.brake * max(0.0, a - reaction)))
        if self.worst_radius:
            v_curve = math.sqrt(self.grip * self.worst_radius)
            limit = min(limit, math.sqrt(v_curve ** 2 + 2 * self.brake * max(0.0, arcs[-1] - reaction)))

        # Steering: pure pursuit toward a point ahead, compensating for the hold time.
        lookahead = max(4.0, 0.6 * v + 3.0)
        target = next((p for p, a in fwd if a >= lookahead), pts[-1])
        x, y = target[0] - reaction, target[1]
        kappa = 2 * y / max(x * x + y * y, 1e-6)
        delta = math.atan(kappa * self.v.wheelbase)
        steering = _nearest(list(Steering), delta, lambda s: steering_target(s, self.v))

        dv = limit - v
        if dv > 3:
            throttle = Throttle.FULL_ACCELERATE
        elif dv > 0.5:
            throttle = Throttle.ACCELERATE
        elif dv > -0.5:
            throttle = Throttle.COAST
        elif dv > -3:
            throttle = Throttle.BRAKE
        else:
            throttle = Throttle.HARD_BRAKE
        return Action(steering, throttle)


class CrudeController:
    """Deliberately simple rule. Its time is score 0."""

    name = "crude"

    def __init__(self, config: GameConfig, target_speed: float = 12.0, aim_distance: float = 10.0):
        self.target_speed = min(target_speed, config.vehicle.max_speed)
        self.aim = aim_distance

    def reset(self) -> None:
        pass

    def act(self, obs: Observation) -> Action:
        ahead = [p for p, a in zip(obs.visible_centerline, obs.visible_arclengths) if a >= self.aim]
        x, y = ahead[0] if ahead else obs.visible_centerline[-1]
        angle = math.degrees(math.atan2(y, x))
        if angle > 25:
            steering = Steering.HARD_LEFT
        elif angle > 10:
            steering = Steering.LEFT
        elif angle > 3:
            steering = Steering.SLIGHT_LEFT
        elif angle < -25:
            steering = Steering.HARD_RIGHT
        elif angle < -10:
            steering = Steering.RIGHT
        elif angle < -3:
            steering = Steering.SLIGHT_RIGHT
        else:
            steering = Steering.STRAIGHT
        if obs.speed < self.target_speed - 1:
            throttle = Throttle.ACCELERATE
        elif obs.speed > self.target_speed + 1:
            throttle = Throttle.BRAKE
        else:
            throttle = Throttle.COAST
        return Action(steering, throttle)


def oracle_config(config: GameConfig) -> GameConfig:
    """Same game, same road, but the whole road is visible (score anchor only)."""
    return dataclasses.replace(config, sim=dataclasses.replace(config.sim, lookahead=ORACLE_LOOKAHEAD))


def oracle_controller(config: GameConfig) -> ReferenceController:
    ctl = ReferenceController(oracle_config(config), grip_use=0.9, brake_use=0.9, assume_worst_beyond=False)
    ctl.name = "oracle"
    return ctl


BASELINES = {
    "reference": ReferenceController,
    "crude": CrudeController,
}
