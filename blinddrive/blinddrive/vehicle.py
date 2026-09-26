"""Simplified kinematic bicycle model with inertia.

State: x, y, heading, speed, steering_angle.
Control: an Action (steering target level + throttle level).

Inertia comes from three places:
  * speed changes at a bounded rate (acceleration / braking limits);
  * the steering angle moves toward its target at a bounded rate;
  * tyre grip caps lateral acceleration, so a fast car cannot follow a tight
    curvature (it understeers). Braking hard uses up grip that would otherwise
    be available for turning (friction circle).

This is physics, not a safety system: nothing here ever changes the action.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .actions import Action, Steering, Throttle
from .config import VehicleConfig


@dataclass(frozen=True)
class VehicleState:
    x: float
    y: float
    heading: float         # radians, world frame, CCW positive
    speed: float           # m/s, never negative (no reverse gear)
    steering_angle: float  # radians, + = left


def steering_target(steering: Steering, cfg: VehicleConfig) -> float:
    full = math.radians(cfg.max_steering_angle_deg)
    fraction = {
        Steering.HARD_LEFT: 1.0,
        Steering.LEFT: cfg.steer_fraction,
        Steering.SLIGHT_LEFT: cfg.slight_steer_fraction,
        Steering.STRAIGHT: 0.0,
        Steering.SLIGHT_RIGHT: -cfg.slight_steer_fraction,
        Steering.RIGHT: -cfg.steer_fraction,
        Steering.HARD_RIGHT: -1.0,
    }[steering]
    return fraction * full


def tyre_acceleration(throttle: Throttle, cfg: VehicleConfig) -> float:
    """Longitudinal acceleration requested from the tyres (before resistance)."""
    return {
        Throttle.HARD_BRAKE: -cfg.max_braking,
        Throttle.BRAKE: -cfg.partial_brake * cfg.max_braking,
        Throttle.COAST: 0.0,
        Throttle.ACCELERATE: cfg.partial_throttle * cfg.max_acceleration,
        Throttle.FULL_ACCELERATE: cfg.max_acceleration,
    }[throttle]


def step_vehicle(state: VehicleState, action: Action, cfg: VehicleConfig, dt: float) -> VehicleState:
    # Steering: move toward the target at a limited rate.
    target = steering_target(action.steering, cfg)
    max_delta = math.radians(cfg.max_steering_rate_deg) * dt
    delta = state.steering_angle + max(-max_delta, min(max_delta, target - state.steering_angle))

    # Longitudinal: tyre force minus resistance. Brakes cannot push backwards.
    a_tyre = tyre_acceleration(action.throttle, cfg)
    v = state.speed
    resistance = (cfg.rolling_resistance + cfg.drag_coefficient * v * v) if v > 0 else 0.0
    v_new = min(cfg.max_speed, max(0.0, v + (a_tyre - resistance) * dt))

    # Lateral: commanded curvature, limited by remaining grip (friction circle).
    v_mid = 0.5 * (v + v_new)
    kappa = math.tan(delta) / cfg.wheelbase
    a_long = min(abs(a_tyre), cfg.grip)
    lat_limit = math.sqrt(cfg.grip * cfg.grip - a_long * a_long)
    if v_mid > 0 and v_mid * v_mid * abs(kappa) > lat_limit:
        kappa = math.copysign(lat_limit / (v_mid * v_mid), kappa)

    heading_new = state.heading + v_mid * kappa * dt
    h_mid = 0.5 * (state.heading + heading_new)
    return VehicleState(
        x=state.x + v_mid * math.cos(h_mid) * dt,
        y=state.y + v_mid * math.sin(h_mid) * dt,
        heading=(heading_new + math.pi) % (2 * math.pi) - math.pi,
        speed=v_new,
        steering_angle=delta,
    )


def footprint(state: VehicleState, cfg: VehicleConfig) -> tuple[tuple[float, float], ...]:
    """The four corners of the car rectangle: front-left, front-right, rear-right, rear-left."""
    c, s = math.cos(state.heading), math.sin(state.heading)
    hl, hw = cfg.length / 2, cfg.width / 2
    corners = ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))
    return tuple((state.x + fx * c - fy * s, state.y + fx * s + fy * c) for fx, fy in corners)
