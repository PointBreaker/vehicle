"""Jev (TypeSafe AI System One) as a BlindDrive controller.

Each decision tick, the Observation is turned into a JSON ``state`` and Jev is
asked two typed *choice* questions:

    steering: one of the 7 Steering labels
    throttle: one of the 5 Throttle labels

Jev's chosen labels become the Action. Nothing else decides anything: there is
no fallback policy, no heuristic override and no default action. If the API
fails (after the client's retries) or answers with an unknown label, the
controller raises and the episode is aborted and reported as an error.

The state contains only (a) the Observation, reformatted and summarised in
plain geometric terms, and (b) the public rules of the game (limits, timing).
It never contains the seed, the road, or anything beyond the lookahead.
"""

from __future__ import annotations

import math
from typing import Any

from ..actions import Action, Steering, Throttle
from ..config import GameConfig
from ..observation import Observation
from ..typesafe import SystemOneResult, TypeSafeClient, TypeSafeConfig, TypeSafeError
from ..vehicle import steering_target, tyre_acceleration

PREVIEW_STEP = 5.0          # metres between "road_ahead" samples
STRAIGHT_RADIUS = 300.0     # curves gentler than this are reported as straight


class JevDecisionError(TypeSafeError):
    """Jev could not produce a legal decision; the episode must stop."""


def public_rules(config: GameConfig) -> dict[str, Any]:
    """Game rules every controller may know. No road, no seed."""
    v, sim = config.vehicle, config.sim
    rules = {
        "goal": f"Reach the finish {config.road.length:.0f} m along the road as fast as possible.",
        "decision_interval_ms": round(1000 / sim.decision_hz),
        "visible_road_ahead_m": sim.lookahead,
        "road_width_m": config.road.width,
        "car_length_m": v.length,
        "car_width_m": v.width,
        "max_speed_mps": v.max_speed,
        "max_braking_mps2": v.max_braking,
        "max_acceleration_mps2": v.max_acceleration,
        "tyre_grip_mps2": v.grip,
        "max_steering_angle_deg": v.max_steering_angle_deg,
        "steering_rate_deg_per_s": v.max_steering_rate_deg,
    }
    if sim.edge == "wall":
        rules["road_edge"] = (f"The edges are walls. Hitting one keeps only {sim.edge_impact_speed_factor:.0%} of "
                              f"your speed, and while scraping it you cannot exceed {sim.edge_max_speed:g} m/s.")
    else:
        rules["road_edge"] = "Touching the road edge ends the run as a crash."
    return rules


def build_questions(config: GameConfig) -> dict[str, dict[str, Any]]:
    v, sim = config.vehicle, config.sim
    interval = round(1000 / sim.decision_hz)

    def steer_desc(s: Steering, text: str) -> str:
        deg = math.degrees(steering_target(s, v))
        radius = v.wheelbase / math.tan(abs(math.radians(deg))) if deg else math.inf
        turn = "no turn" if not deg else f"turning circle radius about {radius:.0f} m at low speed"
        return f"{text}: wheel angle target {deg:+.0f} deg ({turn})"

    steering = {
        Steering.HARD_LEFT: "full lock to the left",
        Steering.LEFT: "medium turn to the left",
        Steering.SLIGHT_LEFT: "small correction to the left",
        Steering.STRAIGHT: "wheels straight",
        Steering.SLIGHT_RIGHT: "small correction to the right",
        Steering.RIGHT: "medium turn to the right",
        Steering.HARD_RIGHT: "full lock to the right",
    }
    throttle = {
        Throttle.HARD_BRAKE: "maximum braking",
        Throttle.BRAKE: "moderate braking",
        Throttle.COAST: "no throttle, no brake (slow drift down)",
        Throttle.ACCELERATE: "moderate acceleration",
        Throttle.FULL_ACCELERATE: "full acceleration",
    }
    return {
        "steering": {
            "type": "choice",
            "instructions": (
                f"You are driving a car on a road you can only partly see (the next {sim.lookahead:g} m). "
                f"Choose the steering target for the next {interval} ms. The wheels move toward the target at "
                f"only {v.max_steering_rate_deg:g} deg/s, so steering must start early. Positive y / left means "
                f"to the car's left. Follow the road and keep the whole car away from both edges."
            ),
            "criteria": {s.value: steer_desc(s, t) for s, t in steering.items()},
        },
        "throttle": {
            "type": "choice",
            "instructions": (
                f"Choose throttle or brake for the next {interval} ms. Finish as fast as possible without hitting "
                f"the road edges. The road beyond {sim.lookahead:g} m is unknown and may turn sharply, and braking "
                f"needs distance. Tyre grip ({v.grip:g} m/s^2 in total) limits cornering speed, and braking hard "
                f"while turning reduces how sharply the car can turn."
            ),
            "criteria": {t.value: f"{text} ({tyre_acceleration(t, v):+.0f} m/s^2)" for t, text in throttle.items()},
        },
    }


def _turn_at(points: list[tuple[float, float]], i: int, span: int = 2) -> float:
    """Signed curvature (1/m, + = left) around point i of a ~1 m spaced polyline."""
    a, b = max(0, i - span), min(len(points) - 1, i + span)
    if b - a < 2:
        return 0.0
    m = (a + b) // 2
    (x0, y0), (x1, y1), (x2, y2) = points[a], points[m], points[b]
    h1 = math.atan2(y1 - y0, x1 - x0)
    h2 = math.atan2(y2 - y1, x2 - x1)
    dh = (h2 - h1 + math.pi) % (2 * math.pi) - math.pi
    ds = math.hypot(x1 - x0, y1 - y0) + math.hypot(x2 - x1, y2 - y1)
    return 2 * dh / ds if ds > 0 else 0.0


def observation_to_state(obs: Observation, rules: dict[str, Any]) -> dict[str, Any]:
    """Describe the Observation for Jev. Pure function of the Observation + public rules."""
    half = obs.road_width / 2
    pts = list(obs.visible_centerline)
    arcs = list(obs.visible_arclengths)

    road_ahead = []
    d = 0.0
    while d <= arcs[-1] + 1e-6:
        i = min(range(len(arcs)), key=lambda k: abs(arcs[k] - d))
        x, y = pts[i]
        j = min(len(pts) - 1, i + 1)
        k = max(0, i - 1)
        direction = math.degrees(math.atan2(pts[j][1] - pts[k][1], pts[j][0] - pts[k][0]))
        kappa = _turn_at(pts, i)
        if abs(kappa) < 1 / STRAIGHT_RADIUS:
            curve = "straight"
        else:
            curve = f"curving {'left' if kappa > 0 else 'right'}, radius {1 / abs(kappa):.0f} m"
        road_ahead.append({
            "along_road_m": round(arcs[i], 1),
            "centerline_forward_m": round(x, 1),
            "centerline_left_m": round(y, 1),
            "road_direction_deg": round(direction, 1),
            "shape": curve,
        })
        d += PREVIEW_STEP

    prev = obs.previous_action
    return {
        "car": {
            "speed_mps": round(obs.speed, 2),
            "steering_angle_deg": round(math.degrees(obs.steering_angle), 1),
            "heading_vs_road_deg": round(math.degrees(obs.heading_error), 1),
            "offset_from_centerline_m": round(obs.lateral_offset, 2),
            "space_to_left_edge_m": round(half - obs.lateral_offset, 2),
            "space_to_right_edge_m": round(half + obs.lateral_offset, 2),
            "touching_edge": obs.touching_edge,
        },
        "road_ahead": road_ahead,
        "road_beyond_visible": "unknown",
        "progress": {
            "distance_m": round(obs.distance_travelled, 1),
            "remaining_m": round(obs.distance_to_finish, 1),
            "time_s": round(obs.elapsed_time, 2),
        },
        "previous_action": prev.to_dict() if prev else None,
        "coordinates": "Car frame: forward = ahead of the car, left = to the car's left (negative = right). "
                       "Angles: positive = to the left.",
        "rules": rules,
    }


def _pick(result: SystemOneResult, name: str, enum):
    answer = result.answers.get(name)
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        raise JevDecisionError(f"Jev returned no choice answer for {name!r}: {answer!r}")
    label = answer.get("choice")
    try:
        value = enum(label)
    except ValueError:
        raise JevDecisionError(f"Jev returned an illegal {name} label {label!r}") from None
    return value, {
        "choice": label,
        "confidence": answer.get("confidence"),
        "probabilities": answer.get("probabilities"),
    }


class JevController:
    name = "jev"
    remote = True  # act() does network I/O; GUI runners call it off the render thread

    def __init__(self, config: GameConfig, client: TypeSafeClient, model: str | None = None):
        self.client = client
        self.model = model or client.config.model
        self.rules = public_rules(config)
        self.questions = build_questions(config)
        self.last_info: dict[str, Any] | None = None
        self.calls = 0

    @classmethod
    def from_env(cls, config: GameConfig, model: str | None = None) -> "JevController":
        return cls(config, TypeSafeClient(TypeSafeConfig.from_env(model=model)))

    def reset(self) -> None:
        self.last_info = None
        self.calls = 0

    def act(self, observation: Observation) -> Action:
        state = observation_to_state(observation, self.rules)
        result = self.client.system_one(state, self.questions, model=self.model)
        self.calls += 1
        steering, steer_info = _pick(result, "steering", Steering)
        throttle, throttle_info = _pick(result, "throttle", Throttle)
        self.last_info = {
            "model": result.model,
            "request_id": result.request_id,
            "latency_ms": round(result.latency_s * 1000, 1),
            "total_ms": round(result.total_s * 1000, 1),
            "attempts": result.attempts,
            "usage": result.usage,
            "steering": steer_info,
            "throttle": throttle_info,
            "state": state,
        }
        return Action(steering, throttle)
