"""All tunable numbers live here. No magic numbers elsewhere.

Units: metres, seconds, radians unless the field name says ``_deg``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VehicleConfig:
    # Geometry (footprint used for crash detection).
    length: float = 4.2
    width: float = 1.8
    wheelbase: float = 2.6

    # Longitudinal dynamics.
    max_speed: float = 30.0            # m/s, hard cap
    max_acceleration: float = 4.0      # m/s^2 at FULL_ACCELERATE
    max_braking: float = 8.0           # m/s^2 at HARD_BRAKE
    partial_throttle: float = 0.5      # ACCELERATE = partial_throttle * max_acceleration
    partial_brake: float = 0.5         # BRAKE = partial_brake * max_braking
    rolling_resistance: float = 0.3    # m/s^2, always opposing motion
    drag_coefficient: float = 0.0012   # m/s^2 per (m/s)^2

    # Tyre grip: total acceleration budget (friction circle).
    # Lateral acceleration available = sqrt(grip^2 - a_long^2).
    # If the commanded curvature needs more, the car understeers.
    grip: float = 9.0                  # m/s^2

    # Steering.
    max_steering_angle_deg: float = 30.0
    max_steering_rate_deg: float = 60.0   # deg/s the wheels can turn
    slight_steer_fraction: float = 1 / 6  # SLIGHT_* = fraction * max angle
    steer_fraction: float = 0.5           # LEFT / RIGHT = fraction * max angle


@dataclass(frozen=True)
class RoadConfig:
    length: float = 500.0          # finish line distance along the centerline
    width: float = 8.0
    runout: float = 60.0           # extra road generated after the finish line
    sample_spacing: float = 0.5    # centerline sample spacing
    start_straight: float = 30.0   # calm straight at the start

    # Segment generators. Curves are defined by (radius, turn angle).
    straight_length: tuple[float, float] = (15.0, 60.0)
    gentle_radius: tuple[float, float] = (50.0, 120.0)
    gentle_angle_deg: tuple[float, float] = (20.0, 60.0)
    sharp_radius: tuple[float, float] = (18.0, 30.0)
    sharp_angle_deg: tuple[float, float] = (60.0, 150.0)
    # Relative probability of (straight, gentle curve, sharp curve).
    segment_weights: tuple[float, float, float] = (0.3, 0.35, 0.35)

    # Moving-average window applied (twice) to the curvature profile.
    # This turns curvature steps into smooth ramps (no instant kinks).
    smoothing_window: float = 10.0

    # Keeps the road heading roughly "forward" so it does not spiral.
    heading_bias_scale_deg: float = 45.0
    max_heading_deviation_deg: float = 150.0

    # Non-adjacent parts of the road must stay at least this far apart
    # (centerline to centerline), in multiples of road width.
    min_clearance_widths: float = 2.0
    max_attempts: int = 100


@dataclass(frozen=True)
class SimConfig:
    physics_hz: int = 60
    decision_hz: int = 4
    timeout: float = 180.0          # seconds of simulated time

    # Partial observation.
    lookahead: float = 30.0         # visible road ahead (arc length)
    view_behind: float = 8.0        # visible road behind (arc length)
    observation_spacing: float = 1.0


@dataclass(frozen=True)
class GameConfig:
    name: str = "normal"
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    road: RoadConfig = field(default_factory=RoadConfig)
    sim: SimConfig = field(default_factory=SimConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "GameConfig":
        return GameConfig(
            name=data["name"],
            vehicle=_build(VehicleConfig, data["vehicle"]),
            road=_build(RoadConfig, data["road"]),
            sim=_build(SimConfig, data["sim"]),
        )


def _build(cls, data: dict[str, Any]):
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in data.items()})


# ---------------------------------------------------------------- presets

def easy() -> GameConfig:
    return GameConfig(
        name="easy",
        vehicle=VehicleConfig(max_speed=22.0),
        road=RoadConfig(
            width=10.0,
            gentle_radius=(70.0, 150.0),
            gentle_angle_deg=(20.0, 50.0),
            sharp_radius=(30.0, 45.0),
            sharp_angle_deg=(45.0, 110.0),
            segment_weights=(0.35, 0.4, 0.25),
        ),
        sim=SimConfig(lookahead=40.0),
    )


def normal() -> GameConfig:
    return GameConfig(name="normal")


def hard() -> GameConfig:
    return GameConfig(
        name="hard",
        vehicle=VehicleConfig(max_speed=40.0),
        road=RoadConfig(
            width=6.5,
            gentle_radius=(40.0, 90.0),
            gentle_angle_deg=(25.0, 70.0),
            sharp_radius=(12.0, 20.0),
            sharp_angle_deg=(70.0, 160.0),
            segment_weights=(0.25, 0.3, 0.45),
        ),
        sim=SimConfig(lookahead=20.0),
    )


PRESETS = {"easy": easy, "normal": normal, "hard": hard}


def get_preset(name: str) -> GameConfig:
    try:
        return PRESETS[name]()
    except KeyError:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}") from None
