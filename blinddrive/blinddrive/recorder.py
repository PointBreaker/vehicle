"""Episode logs (JSONL) and deterministic replay verification.

File layout, one JSON object per line:
    {"type": "header", ...}      seed, full config, controller name
    {"type": "decision", ...}    one per decision tick: state, observation, action
    {"type": "result", ...}      EpisodeResult

Because the simulation is deterministic, the seed + config + action sequence
fully determine the episode. ``verify_log`` re-simulates it and checks every
recorded state.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import __version__
from .actions import Action
from .config import GameConfig
from .controllers.replay import ReplayController, ReplayExhausted
from .env import BlindDriveEnv, run_episode

FORMAT = "blinddrive-log-v1"


def default_log_path(log_dir: str | Path, seed: int, preset: str, controller: str) -> Path:
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return Path(log_dir) / f"{stamp}_{preset}_seed{seed}_{controller}.jsonl"


def write_log(path: str | Path, env: BlindDriveEnv, controller_name: str,
              error: str | None = None, controller_meta: dict[str, Any] | None = None) -> Path:
    """Write the episode log. ``error`` marks an episode aborted by a controller failure."""
    if env.result is None and error is None:
        raise ValueError("episode not finished")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "type": "header",
        "format": FORMAT,
        "blinddrive_version": __version__,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "seed": env.seed,
        "preset": env.config.name,
        "controller": controller_name,
        "config": env.config.to_dict(),
    }
    if controller_meta:
        header["controller_meta"] = controller_meta
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(header) + "\n")
        for rec in env.decision_log:
            f.write(json.dumps({"type": "decision", **rec}) + "\n")
        if env.result is not None:
            f.write(json.dumps({"type": "result", **env.result.to_dict()}) + "\n")
        if error is not None:
            f.write(json.dumps({"type": "error", "time": env.elapsed_time,
                                "decision_count": env.decision_count, "message": error}) + "\n")
    return path


@dataclass
class EpisodeLog:
    header: dict[str, Any]
    decisions: list[dict[str, Any]]
    result: dict[str, Any] | None
    error: dict[str, Any] | None = None

    @property
    def config(self) -> GameConfig:
        return GameConfig.from_dict(self.header["config"])

    @property
    def seed(self) -> int:
        return self.header["seed"]

    @property
    def actions(self) -> list[Action]:
        return [Action.from_dict(d["action"]) for d in self.decisions]


def load_log(path: str | Path) -> EpisodeLog:
    header, decisions, result, error = None, [], None, None
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            kind = rec.pop("type")
            if kind == "header":
                header = rec
            elif kind == "decision":
                decisions.append(rec)
            elif kind == "result":
                result = rec
            elif kind == "error":
                error = rec
    if header is None or header.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} file")
    return EpisodeLog(header, decisions, result, error)


def verify_log(log: EpisodeLog) -> list[str]:
    """Re-simulate the logged actions. Returns a list of mismatches (empty = OK)."""
    env = BlindDriveEnv(log.config, log.seed)
    problems = []
    if log.error is not None:
        return verify_partial(log, env)
    try:
        result = run_episode(env, ReplayController(log.actions))
    except ReplayExhausted:
        result = None
        problems.append(f"episode did not end after the {len(log.decisions)} logged decisions")
    if len(env.decision_log) != len(log.decisions):
        problems.append(f"decision count {len(env.decision_log)} != logged {len(log.decisions)}")
    for got, want in zip(env.decision_log, log.decisions):
        for key in ("tick", "position", "heading", "speed", "steering_angle", "distance_travelled"):
            if got[key] != want[key]:
                problems.append(f"decision {want['index']}: {key} {got[key]} != logged {want[key]}")
                break
    if result is not None and log.result is not None and result.to_dict() != log.result:
        problems.append(f"result {result.to_dict()} != logged {log.result}")
    return problems


def verify_partial(log: EpisodeLog, env: BlindDriveEnv) -> list[str]:
    """Check the decisions of an aborted episode up to where it stopped."""
    problems = []
    for i, want in enumerate(log.decisions):
        if env.done:
            problems.append(f"episode ended before logged decision {i}")
            break
        while not env.decision_due:
            env.tick()
        got = {"position": [env.debug_view().state.x, env.debug_view().state.y]}
        if got["position"] != want["position"]:
            problems.append(f"decision {i}: position {got['position']} != logged {want['position']}")
            break
        env.apply_action(Action.from_dict(want["action"]))
        env.tick()
    return problems
