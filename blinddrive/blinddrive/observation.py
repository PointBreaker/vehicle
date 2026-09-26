"""What a controller is allowed to see.

The Observation is built only from the road inside the visible window
``[s - view_behind, s + lookahead]`` (arc length), expressed in the car's
local frame. It holds plain floats and tuples only: no reference to the
Road, the environment, or anything else a controller could use to peek ahead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .actions import Action
from .config import SimConfig
from .road import Road
from .vehicle import VehicleState


@dataclass(frozen=True)
class Observation:
    speed: float                    # m/s
    steering_angle: float           # rad, actual wheel angle (+ = left)
    heading_error: float            # rad, car heading - road heading (+ = car points left of road)
    lateral_offset: float           # m, car centre from centerline (+ = left)
    # Visible centerline in car-local coordinates: x = forward, y = left (metres).
    visible_centerline: tuple[tuple[float, float], ...]
    # Arc length of each visible point relative to the car's position on the
    # road (negative = behind). Always <= lookahead.
    visible_arclengths: tuple[float, ...]
    road_width: float
    lookahead: float
    previous_action: Action | None
    elapsed_time: float
    distance_travelled: float       # progress along the centerline
    distance_to_finish: float

    def to_dict(self, precision: int = 4) -> dict[str, Any]:
        r = lambda v: round(v, precision)  # noqa: E731
        return {
            "speed": r(self.speed),
            "steering_angle": r(self.steering_angle),
            "heading_error": r(self.heading_error),
            "lateral_offset": r(self.lateral_offset),
            "visible_centerline": [[r(x), r(y)] for x, y in self.visible_centerline],
            "visible_arclengths": [r(a) for a in self.visible_arclengths],
            "road_width": self.road_width,
            "lookahead": self.lookahead,
            "previous_action": self.previous_action.to_dict() if self.previous_action else None,
            "elapsed_time": r(self.elapsed_time),
            "distance_travelled": r(self.distance_travelled),
            "distance_to_finish": r(self.distance_to_finish),
        }


def to_local(x: float, y: float, state: VehicleState) -> tuple[float, float]:
    """World point -> car frame (x forward, y left)."""
    dx, dy = x - state.x, y - state.y
    c, s = math.cos(state.heading), math.sin(state.heading)
    return (c * dx + s * dy, -s * dx + c * dy)


def build_observation(
    road: Road,
    state: VehicleState,
    progress: float,
    lateral_offset: float,
    road_heading: float,
    cfg: SimConfig,
    previous_action: Action | None,
    elapsed_time: float,
) -> Observation:
    pts = road.sample(progress - cfg.view_behind, progress + cfg.lookahead, cfg.observation_spacing)
    local = tuple(to_local(x, y, state) for x, y, _ in pts)
    arcs = tuple(s - progress for _, _, s in pts)
    heading_error = (state.heading - road_heading + math.pi) % (2 * math.pi) - math.pi
    return Observation(
        speed=state.speed,
        steering_angle=state.steering_angle,
        heading_error=heading_error,
        lateral_offset=lateral_offset,
        visible_centerline=local,
        visible_arclengths=arcs,
        road_width=road.width,
        lookahead=cfg.lookahead,
        previous_action=previous_action,
        elapsed_time=elapsed_time,
        distance_travelled=progress,
        distance_to_finish=max(0.0, road.length - progress),
    )
