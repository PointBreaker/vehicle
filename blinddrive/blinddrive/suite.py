"""Evaluation suites: certified-solvable seeds with their reference times.

    uv run python -m blinddrive.suite build --preset normal --count 20 --start 1000 \\
        --name normal-dev --out suites/normal-dev.json
    uv run python -m blinddrive.suite build --preset hard --count 30 --random \\
        --name hard-test --out private/hard-test.json      # secret seeds: do not publish
    uv run python -m blinddrive.suite show suites/normal-dev.json

A seed is accepted only if the observation-only reference controller finishes it
without touching the edge, i.e. it is solvable with exactly the information any
controller gets. For each accepted seed the suite stores the crude (score 0),
reference and oracle (score 100) results, so runs can be scored without
recomputing them. The config fingerprint ties a suite to its exact rules.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import secrets
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from . import __version__
from .baselines import CrudeController, ReferenceController, oracle_config, oracle_controller
from .config import EDGE_MODES, PRESETS, GameConfig, get_preset
from .env import BlindDriveEnv, run_episode
from .scoring import SCORING_VERSION, effective_time

FORMAT = "blinddrive-suite-v1"


def anchor_config(config: GameConfig) -> GameConfig:
    """Anchors are always measured without reaction delay (see bench --delay-ms)."""
    return replace(config, sim=replace(config.sim, action_delay_ms=0.0))


def fingerprint(config: GameConfig) -> str:
    blob = json.dumps({"config": anchor_config(config).to_dict(), "scoring": SCORING_VERSION}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _summary(result, road_length: float) -> dict[str, Any]:
    return {
        "time": round(effective_time(result.elapsed_time, result.distance_travelled, road_length, result.success), 4),
        "finished": result.success,
        "edge_hits": result.edge_hits,
    }


def certify(config: GameConfig, seed: int) -> dict[str, Any]:
    """Run the three anchors on one seed."""
    cfg = anchor_config(config)
    length = cfg.road.length
    ref = run_episode(BlindDriveEnv(cfg, seed), ReferenceController(cfg))
    floor = run_episode(BlindDriveEnv(cfg, seed), CrudeController(cfg))
    top = run_episode(BlindDriveEnv(oracle_config(cfg), seed), oracle_controller(cfg))
    return {
        "seed": seed,
        "solvable": ref.success and ref.edge_hits == 0,
        "reference": _summary(ref, length),
        "floor": _summary(floor, length),
        "oracle": _summary(top, length),
    }


def build(config: GameConfig, count: int, start: int | None, name: str, max_tries: int | None = None) -> dict[str, Any]:
    seeds, rejected = [], []
    candidate = start if start is not None else secrets.randbelow(2 ** 31)
    tries = 0
    limit = max_tries or count * 5
    while len(seeds) < count and tries < limit:
        cert = certify(config, candidate)
        ok = cert["solvable"] and cert["floor"]["time"] > cert["oracle"]["time"]
        (seeds if ok else rejected).append(cert if ok else {"seed": candidate, "reason": "not certified"})
        tries += 1
        candidate = candidate + 1 if start is not None else secrets.randbelow(2 ** 31)
    if len(seeds) < count:
        raise RuntimeError(f"only {len(seeds)} of {count} seeds certified after {tries} tries")
    return {
        "format": FORMAT,
        "name": name,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "blinddrive_version": __version__,
        "scoring": SCORING_VERSION,
        "preset": config.name,
        "fingerprint": fingerprint(config),
        "config": anchor_config(config).to_dict(),
        "seeds": seeds,
        "rejected": rejected,
    }


class SuiteError(ValueError):
    pass


def load_suite(path: str | Path, check_stale: bool = True) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format") != FORMAT:
        raise SuiteError(f"{path}: not a {FORMAT} file")
    config = GameConfig.from_dict(data["config"])
    if fingerprint(config) != data["fingerprint"]:
        raise SuiteError(f"{path}: config does not match its fingerprint (edited or built by another version)")
    if check_stale and data["seeds"]:
        # Physics or baseline code may have changed since the suite was built.
        first = data["seeds"][0]
        now = certify(config, first["seed"])
        if now["reference"] != first["reference"] or now["oracle"] != first["oracle"]:
            raise SuiteError(f"{path}: reference times no longer reproduce; rebuild the suite")
    data["game_config"] = config
    return data


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="blinddrive.suite", description="Build and inspect evaluation suites.")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="certify seeds and write a suite file")
    b.add_argument("--preset", choices=sorted(PRESETS), default="normal")
    b.add_argument("--edge", choices=EDGE_MODES, default=None)
    b.add_argument("--count", type=int, default=20)
    g = b.add_mutually_exclusive_group()
    g.add_argument("--start", type=int, default=1000, help="first candidate seed (consecutive search)")
    g.add_argument("--random", action="store_true", help="draw secret random seeds (for private test suites)")
    b.add_argument("--name", default=None)
    b.add_argument("--out", required=True)
    s = sub.add_parser("show", help="summarise a suite file")
    s.add_argument("path")
    args = p.parse_args(argv)

    if args.cmd == "build":
        config = get_preset(args.preset)
        if args.edge:
            config = config.with_edge(args.edge)
        name = args.name or Path(args.out).stem
        suite = build(config, args.count, None if args.random else args.start, name)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(suite, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {out}: {len(suite['seeds'])} seeds, {len(suite['rejected'])} rejected, "
              f"fingerprint {suite['fingerprint']}")
        return 0

    data = load_suite(args.path, check_stale=False)
    seeds = data["seeds"]
    mean = lambda k: sum(s[k]["time"] for s in seeds) / len(seeds)  # noqa: E731
    print(f"{data['name']}: preset={data['preset']} edge={data['game_config'].sim.edge} seeds={len(seeds)} "
          f"fingerprint={data['fingerprint']} built={data['created']}")
    print(f"mean times: crude {mean('floor'):.1f}s   reference {mean('reference'):.1f}s   oracle {mean('oracle'):.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
