"""The BlindDrive environment.

The environment owns physics. The controller owns decisions.

The environment only simulates, observes, and judges (finish / crash / timeout).
It never modifies, overrides or "corrects" an action.

Road edge (``sim.edge``):
  * "wall"  (default): the edge is a barrier. When the car footprint reaches it,
    the car's position is pushed back just inside the road (collision), it loses
    speed on impact and is speed-capped while scraping. Heading, steering and
    throttle are never touched: getting away from the wall is the controller's job.
  * "crash": touching outside the road ends the episode immediately.

Timing: physics runs at ``physics_hz``; a new decision is required every
``physics_hz / decision_hz`` ticks. Between decisions the last action is held.
The environment enforces this schedule itself, so every runner (GUI,
headless, future AI harness) gives every controller the same decision rate.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any

from .actions import DEFAULT_ACTION as NO_INPUT
from .actions import Action, validate_action
from .config import GameConfig
from .controllers.base import Controller
from .observation import Observation, build_observation
from .road import Road, generate_road
from .vehicle import VehicleState, footprint, step_vehicle


EDGE_CONTACT_MARGIN = 0.05   # m; within this distance of a wall counts as still touching
START_WALL_MARGIN = 0.5      # m behind the car's rear at spawn


class ProtocolError(RuntimeError):
    """Raised when a runner breaks the decision schedule."""


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    preset: str
    success: bool
    termination: str                   # "finished" | "crashed" | "stalled" | "timeout"
    completion_time: float | None      # only when success
    elapsed_time: float
    distance_travelled: float
    crash_distance: float | None       # only when crashed
    average_speed: float
    max_speed: float
    decision_count: int
    edge_hits: int = 0                 # separate impacts with the edge (wall mode)
    edge_contact_time: float = 0.0     # seconds spent scraping the edge (wall mode)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DebugView:
    """Privileged environment state. For --debug rendering only, never for controllers."""
    road: Road
    state: VehicleState
    progress: float
    footprint: tuple[tuple[float, float], ...]


class BlindDriveEnv:
    def __init__(self, config: GameConfig, seed: int):
        sim = config.sim
        if sim.physics_hz % sim.decision_hz != 0:
            raise ValueError("physics_hz must be a multiple of decision_hz")
        self.config = config
        self.seed = seed
        self.dt = 1.0 / sim.physics_hz
        self.ticks_per_decision = sim.physics_hz // sim.decision_hz
        if sim.action_delay_ms < 0:
            raise ValueError("action_delay_ms must be >= 0")
        self.delay_ticks = round(sim.action_delay_ms / 1000 * sim.physics_hz)
        self._road = generate_road(seed, config.road)
        self.reset()

    # ------------------------------------------------------------ lifecycle

    def reset(self) -> Observation:
        x, y, h = self._road.pose_at(0.0)
        self._state = VehicleState(x=x, y=y, heading=h, speed=0.0, steering_angle=0.0)
        self._progress = 0.0
        self._tick = 0
        self._action: Action | None = None       # latest decision
        self._applied: Action | None = None      # action the physics is executing
        self._queue: list[tuple[int, Action]] = []   # (tick it takes effect, action)
        self._pending_decision = True
        self._max_speed = 0.0
        self._touching_edge = False
        self._edge_hits = 0
        self._edge_contact_ticks = 0
        self._stall_mark = (0.0, 0)              # (progress, tick) of the last real progress
        self._result: EpisodeResult | None = None
        self.decision_log: list[dict[str, Any]] = []
        return self.observe()

    @property
    def elapsed_time(self) -> float:
        return self._tick * self.dt

    @property
    def decision_due(self) -> bool:
        return self._pending_decision and self._result is None

    @property
    def done(self) -> bool:
        return self._result is not None

    @property
    def result(self) -> EpisodeResult | None:
        return self._result

    @property
    def current_action(self) -> Action | None:
        """The action the car is executing right now (differs from the latest
        decision while ``action_delay_ms`` has not elapsed)."""
        return self._applied

    @property
    def decision_count(self) -> int:
        return len(self.decision_log)

    @property
    def edge_hits(self) -> int:
        return self._edge_hits

    # ------------------------------------------------------------ interface

    def observe(self) -> Observation:
        proj = self._road.project(self._state.x, self._state.y, self._progress, self._search_window())
        return build_observation(
            road=self._road,
            state=self._state,
            progress=self._progress,
            lateral_offset=proj.lateral,
            road_heading=proj.heading,
            cfg=self.config.sim,
            previous_action=self._action,
            elapsed_time=self.elapsed_time,
            touching_edge=self._touching_edge,
        )

    def apply_action(self, action: Action) -> None:
        """Submit the decision for the current decision tick."""
        if self._result is not None:
            raise ProtocolError("episode is over")
        if not self._pending_decision:
            raise ProtocolError("no decision is due now; the previous action is still being held")
        action = validate_action(action)
        obs = self.observe()
        st = self._state
        self.decision_log.append({
            "index": len(self.decision_log),
            "tick": self._tick,
            "time": round(self.elapsed_time, 6),
            "position": [st.x, st.y],
            "heading": st.heading,
            "speed": st.speed,
            "steering_angle": st.steering_angle,
            "distance_travelled": self._progress,
            "touching_edge": self._touching_edge,
            "observation": obs.to_dict(),
            "action": action.to_dict(),
            "applies_at_tick": self._tick + self.delay_ticks,
        })
        self._action = action
        if self.delay_ticks == 0:
            self._applied = action
        else:
            self._queue.append((self._tick + self.delay_ticks, action))
        self._pending_decision = False

    def tick(self) -> None:
        """Advance physics by one step, holding the current action."""
        if self._result is not None:
            raise ProtocolError("episode is over")
        if self._pending_decision:
            raise ProtocolError("a decision is due; call apply_action first")
        sim = self.config.sim
        while self._queue and self._queue[0][0] <= self._tick:
            self._applied = self._queue.pop(0)[1]
        # Before the first decision takes effect (only with a delay), the car,
        # which starts at rest, simply rolls with no input.
        state = step_vehicle(self._state, self._applied or NO_INPUT, self.config.vehicle, self.dt)
        crashed = False
        if sim.edge == "wall":
            state, depth = self._resolve_edge(state)
            # A small margin keeps a car sliding along the wall "in contact"
            # instead of registering a new impact every tick.
            touching = depth > -EDGE_CONTACT_MARGIN
            if touching:
                speed = state.speed
                if not self._touching_edge:
                    self._edge_hits += 1
                    speed *= sim.edge_impact_speed_factor
                state = replace(state, speed=min(speed, sim.edge_max_speed))
                self._edge_contact_ticks += 1
            self._touching_edge = touching
        else:
            crashed = self._edge_violation(state)[0] > 0
        self._state = state
        self._tick += 1
        self._max_speed = max(self._max_speed, state.speed)
        proj = self._road.project(state.x, state.y, self._progress, self._search_window())
        self._progress = proj.s

        if self._progress >= self._stall_mark[0] + sim.stall_distance:
            self._stall_mark = (self._progress, self._tick)

        if crashed:
            self._finish("crashed")
        elif self._progress >= self._road.length:
            self._finish("finished")
        elif (self._tick - self._stall_mark[1]) * self.dt >= sim.stall_timeout - 1e-9:
            self._finish("stalled")
        elif self.elapsed_time >= self.config.sim.timeout - 1e-9:
            self._finish("timeout")
        elif self._tick % self.ticks_per_decision == 0:
            self._pending_decision = True

    def debug_view(self) -> DebugView:
        return DebugView(
            road=self._road,
            state=self._state,
            progress=self._progress,
            footprint=footprint(self._state, self.config.vehicle),
        )

    # ------------------------------------------------------------ internals

    def _search_window(self) -> float:
        return self.config.vehicle.length + self._state.speed * self.dt * 2 + 2.0

    def _edge_violation(self, state: VehicleState) -> tuple[float, float, float]:
        """Deepest penetration of the footprint into a wall.

        Returns (depth, push_x, push_y): depth in metres (<= 0 means every corner is
        inside), and the unit direction that moves the car back onto the road.
        Walls are the two road edges plus a wall across the road just behind the start.
        """
        half = self._road.width / 2
        start_wall = -(self.config.vehicle.length / 2 + START_WALL_MARGIN)
        window = self._search_window()
        worst = (-math.inf, 0.0, 0.0)
        for cx, cy in footprint(state, self.config.vehicle):
            proj = self._road.project(cx, cy, self._progress, window)
            c, s = math.cos(proj.heading), math.sin(proj.heading)
            side = abs(proj.lateral) - half
            if side > worst[0]:
                sign = -math.copysign(1.0, proj.lateral)   # back toward the centerline
                worst = (side, -s * sign, c * sign)
            behind = start_wall - proj.s
            if behind > worst[0]:
                worst = (behind, c, s)                     # forward along the road
        return worst

    def _resolve_edge(self, state: VehicleState) -> tuple[VehicleState, float]:
        """Wall collision: translate the car back onto the road.

        Only x/y change; heading, steering and speed are left exactly as the
        physics produced them. Returns the new state and the initial penetration depth.
        """
        first_depth = None
        for _ in range(6):
            depth, px, py = self._edge_violation(state)
            if first_depth is None:
                first_depth = depth
            if depth <= 0:
                break
            state = replace(state, x=state.x + px * (depth + 1e-6), y=state.y + py * (depth + 1e-6))
        return state, first_depth

    def _finish(self, termination: str) -> None:
        t = self.elapsed_time
        dist = min(self._progress, self._road.length)
        self._result = EpisodeResult(
            seed=self.seed,
            preset=self.config.name,
            success=termination == "finished",
            termination=termination,
            completion_time=t if termination == "finished" else None,
            elapsed_time=t,
            distance_travelled=dist,
            crash_distance=self._progress if termination == "crashed" else None,
            average_speed=dist / t if t > 0 else 0.0,
            max_speed=self._max_speed,
            decision_count=len(self.decision_log),
            edge_hits=self._edge_hits,
            edge_contact_time=self._edge_contact_ticks * self.dt,
        )


def run_episode(env: BlindDriveEnv, controller: Controller) -> EpisodeResult:
    """Headless loop. The GUI runner does exactly the same thing, plus rendering.

    Simulated time only advances in ``env.tick()``, so however long ``act`` takes
    (e.g. a network call), the car waits: latency never affects the outcome.
    """
    env.reset()
    controller.reset()
    while not env.done:
        if env.decision_due:
            decide(env, controller)
        env.tick()
    assert env.result is not None
    return env.result


def decide(env: BlindDriveEnv, controller: Controller) -> None:
    """Ask the controller for one decision and submit it."""
    apply_decision(env, controller, controller.act(env.observe()))


def apply_decision(env: BlindDriveEnv, controller: Controller, action: Action) -> None:
    """Submit an action; attach the controller's own notes (if any) to the log entry.

    ``last_info`` is recorded for auditing only (e.g. model answers, latency);
    the environment never reads it.
    """
    env.apply_action(action)
    info = getattr(controller, "last_info", None)
    if info is not None:
        env.decision_log[-1]["controller_info"] = info
