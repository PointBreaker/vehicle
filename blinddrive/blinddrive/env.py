"""The BlindDrive environment.

The environment owns physics. The controller owns decisions.

The environment only simulates, observes, and judges (finish / crash / timeout).
It never modifies, overrides or "corrects" an action.

Timing: physics runs at ``physics_hz``; a new decision is required every
``physics_hz / decision_hz`` ticks. Between decisions the last action is held.
The environment enforces this schedule itself, so every runner (GUI,
headless, future AI harness) gives every controller the same decision rate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .actions import Action, validate_action
from .config import GameConfig
from .controllers.base import Controller
from .observation import Observation, build_observation
from .road import Road, generate_road
from .vehicle import VehicleState, footprint, step_vehicle


class ProtocolError(RuntimeError):
    """Raised when a runner breaks the decision schedule."""


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    preset: str
    success: bool
    termination: str                   # "finished" | "crashed" | "timeout"
    completion_time: float | None      # only when success
    elapsed_time: float
    distance_travelled: float
    crash_distance: float | None       # only when crashed
    average_speed: float
    max_speed: float
    decision_count: int

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
        self._road = generate_road(seed, config.road)
        self.reset()

    # ------------------------------------------------------------ lifecycle

    def reset(self) -> Observation:
        x, y, h = self._road.pose_at(0.0)
        self._state = VehicleState(x=x, y=y, heading=h, speed=0.0, steering_angle=0.0)
        self._progress = 0.0
        self._tick = 0
        self._action: Action | None = None
        self._pending_decision = True
        self._max_speed = 0.0
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
        return self._action

    @property
    def decision_count(self) -> int:
        return len(self.decision_log)

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
            "observation": obs.to_dict(),
            "action": action.to_dict(),
        })
        self._action = action
        self._pending_decision = False

    def tick(self) -> None:
        """Advance physics by one step, holding the current action."""
        if self._result is not None:
            raise ProtocolError("episode is over")
        if self._pending_decision:
            raise ProtocolError("a decision is due; call apply_action first")
        vcfg = self.config.vehicle
        self._state = step_vehicle(self._state, self._action, vcfg, self.dt)
        self._tick += 1
        self._max_speed = max(self._max_speed, self._state.speed)
        proj = self._road.project(self._state.x, self._state.y, self._progress, self._search_window())
        self._progress = proj.s

        if self._is_off_road():
            self._finish("crashed")
        elif self._progress >= self._road.length:
            self._finish("finished")
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

    def _is_off_road(self) -> bool:
        half = self._road.width / 2
        window = self._search_window()
        for cx, cy in footprint(self._state, self.config.vehicle):
            if abs(self._road.project(cx, cy, self._progress, window).lateral) > half:
                return True
        return False

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
        )


def run_episode(env: BlindDriveEnv, controller: Controller) -> EpisodeResult:
    """Headless loop. The GUI runner does exactly the same thing, plus rendering."""
    env.reset()
    controller.reset()
    while not env.done:
        if env.decision_due:
            env.apply_action(controller.act(env.observe()))
        env.tick()
    assert env.result is not None
    return env.result
