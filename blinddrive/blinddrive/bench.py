"""Headless evaluation runs with scoring.

    uv run python -m blinddrive.bench --check                           # verify key + list models
    uv run python -m blinddrive.bench --dry-run --seed 42               # print one request, send nothing
    uv run python -m blinddrive.bench --suite suites/normal-dev.json    # score Jev on a certified suite
    uv run python -m blinddrive.bench --suite suites/normal-dev.json --controller reference
    uv run python -m blinddrive.bench --suite suites/hard-dev.json --delay-ms 250 --workers 4
    uv run python -m blinddrive.bench --seeds 1-5 --preset normal       # ad-hoc seeds (anchors computed now)

Each run writes one JSONL log per episode plus ``bench-*.json`` (scorecard +
per-episode rows) and ``bench-*.html`` (readable report). Compare several runs
with ``python -m blinddrive.report runs/bench-*.json``.

Simulated time is paused while waiting for a remote controller, so network
latency is measured and reported but never changes the driving. To measure how
much a reaction delay costs, use ``--delay-ms`` (a fixed, reproducible delay);
scores under delay are still measured against the zero-delay anchors.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

from .baselines import BASELINES
from .config import EDGE_MODES, PRESETS, GameConfig, get_preset
from .controllers.base import controller_name
from .env import BlindDriveEnv, run_episode
from .recorder import default_log_path, write_log
from .scoring import SCORING_VERSION, effective_time, normalized_score, scorecard
from .suite import SuiteError, certify, fingerprint, load_suite
from .typesafe import TypeSafeError, load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
CONTROLLERS = ["jev", *BASELINES]


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        elif part:
            seeds.append(int(part))
    if not seeds:
        raise argparse.ArgumentTypeError("no seeds given")
    return seeds


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="blinddrive.bench", description="Evaluate a controller on BlindDrive.")
    p.add_argument("--controller", choices=CONTROLLERS, default="jev",
                   help="jev, or a reference controller (to calibrate / sanity-check the pipeline)")
    p.add_argument("--suite", default=None, help="certified suite file (recommended)")
    p.add_argument("--seeds", type=parse_seeds, default=None, help="ad-hoc seeds, e.g. 1-10 or 3,5,8")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--preset", choices=sorted(PRESETS), default="normal", help="for ad-hoc seeds")
    p.add_argument("--edge", choices=EDGE_MODES, default=None, help="for ad-hoc seeds")
    p.add_argument("--delay-ms", type=float, default=0.0, help="fixed reaction delay applied to every action")
    p.add_argument("--model", default=None, help="Jev model (default: TYPESAFE_DEFAULT_MODEL or jev-latest)")
    p.add_argument("--workers", type=int, default=1, help="episodes in parallel (API calls are the bottleneck)")
    p.add_argument("--log-dir", default="runs")
    p.add_argument("--check", action="store_true", help="check the API key and list models, then exit")
    p.add_argument("--dry-run", action="store_true", help="print the first request body, send nothing")
    return p.parse_args(argv)


def make_controller(name: str, config: GameConfig, model: str | None):
    if name == "jev":
        from .controllers.jev import JevController

        return JevController.from_env(config, model=model)
    return BASELINES[name](config)


def run_one(config: GameConfig, cert: dict[str, Any], controller_kind: str, model: str | None,
            log_dir: str) -> dict[str, Any]:
    seed = cert["seed"]
    env = BlindDriveEnv(config, seed)
    controller = make_controller(controller_kind, config, model)
    name = controller_name(controller)
    meta = None
    if hasattr(controller, "questions"):
        meta = {"model": controller.model, "harness": getattr(controller, "harness", None),
                "questions": controller.questions, "rules": controller.rules}
    path = default_log_path(log_dir, seed, config.name, name)
    started = time.monotonic()
    error = None
    try:
        run_episode(env, controller)
    except TypeSafeError as err:
        error = str(err)
    wall = time.monotonic() - started
    write_log(path, env, name, error=error, controller_meta=meta)

    length = config.road.length
    progress = env.debug_view().progress
    result = env.result
    finished = bool(result and result.success)
    elapsed = env.elapsed_time
    t_eff = effective_time(elapsed, min(progress, length), length, finished)
    infos = [d["controller_info"] for d in env.decision_log if "controller_info" in d]
    row = {
        "seed": seed,
        "status": "error" if error else result.termination,
        "time": round(elapsed, 2),
        "effective_time": round(t_eff, 2),
        "distance": round(min(progress, length), 1),
        "score": _round(normalized_score(t_eff, cert["floor"]["time"], cert["oracle"]["time"])),
        "reference_score": _round(normalized_score(cert["reference"]["time"], cert["floor"]["time"],
                                                   cert["oracle"]["time"])),
        "edge_hits": env.edge_hits,
        "edge_contact_s": round(result.edge_contact_time, 2) if result else None,
        "max_speed": round(result.max_speed, 2) if result else None,
        "decisions": env.decision_count,
        "latencies_ms": [i["latency_ms"] for i in infos if "latency_ms" in i],
        "tokens_in": sum((i.get("usage") or {}).get("input_tokens") or 0 for i in infos),
        "tokens_out": sum((i.get("usage") or {}).get("output_tokens") or 0 for i in infos),
        "wall_s": round(wall, 1),
        "log": str(path),
    }
    if error:
        row["error"] = error
    return row


def _round(x: float | None) -> float | None:
    return None if x is None else round(x, 2)


def resolve_seeds(args: argparse.Namespace) -> tuple[GameConfig, list[dict[str, Any]], dict[str, Any]]:
    """Seeds with their anchor results, from a suite or computed now."""
    if args.suite:
        suite = load_suite(args.suite)
        config = suite["game_config"]
        certs = suite["seeds"]
        info = {"suite": suite["name"], "suite_file": str(args.suite), "fingerprint": suite["fingerprint"]}
    else:
        config = get_preset(args.preset)
        if args.edge:
            config = config.with_edge(args.edge)
        print("no --suite: computing anchors for ad-hoc seeds (uncertified seeds are scored but flagged)")
        certs = [certify(config, s) for s in (args.seeds or [args.seed])]
        info = {"suite": None, "fingerprint": fingerprint(config),
                "uncertified_seeds": [c["seed"] for c in certs if not c["solvable"]]}
    if args.delay_ms:
        config = replace(config, sim=replace(config.sim, action_delay_ms=args.delay_ms))
    return config, certs, info


def main(argv: list[str] | None = None) -> int:
    load_dotenv(Path.cwd() / ".env", PROJECT_DIR / ".env")
    args = parse_args(argv)

    if args.dry_run:
        from .controllers.jev import build_questions, observation_to_state, public_rules

        config = get_preset(args.preset)
        if args.edge:
            config = config.with_edge(args.edge)
        env = BlindDriveEnv(config, args.seed)
        body = {
            "state": observation_to_state(env.observe(), public_rules(config)),
            "model": args.model or "<TYPESAFE_DEFAULT_MODEL or jev-latest>",
            "questions": build_questions(config),
        }
        print("POST /v1/systemone  (first decision of seed %d, not sent)" % args.seed)
        print(json.dumps(body, indent=2, ensure_ascii=False))
        return 0

    if args.check:
        from .typesafe import TypeSafeClient, TypeSafeConfig

        try:
            client = TypeSafeClient(TypeSafeConfig.from_env(model=args.model))
            models = client.list_models()
        except TypeSafeError as err:
            print(f"ERROR: {err}", file=sys.stderr)
            return 2
        print(f"OK: {client.config.base_url} accepted the key. Models:")
        for m in models:
            print(f"  {m.get('name')}  ({m.get('release_date', '?')})  {m.get('description', '')}")
        print(f"Using: {client.config.model}")
        return 0

    try:
        config, certs, suite_info = resolve_seeds(args)
        probe = make_controller(args.controller, config, args.model)  # fail fast on a missing key
    except (SuiteError, TypeSafeError) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2
    name = controller_name(probe)
    model = getattr(probe, "model", None)
    print(f"BlindDrive bench  controller={name}{' model=' + model if model else ''}  preset={config.name}  "
          f"edge={config.sim.edge}  delay={config.sim.action_delay_ms:g}ms  "
          f"{'suite=' + suite_info['suite'] if suite_info['suite'] else 'ad-hoc seeds'}  episodes={len(certs)}")

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(run_one, config, c, args.controller, args.model, args.log_dir) for c in certs]
        rows = []
        for fut in futures:
            row = fut.result()
            rows.append(row)
            print(format_row(row), flush=True)

    card = scorecard(rows)
    run = {
        "format": "blinddrive-bench-v1",
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "controller": name,
        "model": model,
        "harness": getattr(probe, "harness", None),
        "preset": config.name,
        "edge": config.sim.edge,
        "delay_ms": config.sim.action_delay_ms,
        "scoring": SCORING_VERSION,
        **suite_info,
        "wall_s": round(time.monotonic() - started, 1),
        "config": config.to_dict(),
        "scorecard": card,
        "episodes": rows,
    }
    print("\n" + format_scorecard(run))
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    base = Path(args.log_dir) / f"bench-{stamp}_{name}{'-' + model if model else ''}_{config.name}"
    base.parent.mkdir(parents=True, exist_ok=True)
    json_path = base.with_suffix(".json")
    json_path.write_text(json.dumps(run, indent=1), encoding="utf-8")
    from .report import write_report

    html_path = write_report([run], base.with_suffix(".html"))
    print(f"results: {json_path}\nreport:  {html_path}")
    return 0 if card["errors"] == 0 else 1


def format_row(r: dict[str, Any]) -> str:
    score = "  n/a " if r["score"] is None else f"{r['score']:6.1f}"
    lat = f"  api p50 {sorted(r['latencies_ms'])[len(r['latencies_ms']) // 2]:.0f} ms" if r["latencies_ms"] else ""
    tail = f"  ERROR: {r['error']}" if r["status"] == "error" else ""
    return (f"  seed {r['seed']:>10}  {r['status']:<8}  score {score}  time {r['time']:6.2f}s  "
            f"dist {r['distance']:5.1f} m  bumps {r['edge_hits']:>2}  decisions {r['decisions']:>4}{lat}{tail}")


def format_scorecard(run: dict[str, Any]) -> str:
    c = run["scorecard"]
    s, ref, lat = c["score"], c["reference_score"], c["latency_ms"]
    ci = f" [{s['ci95'][0]}, {s['ci95'][1]}]" if s.get("ci95") else ""
    lines = [
        f"SCORE {s['mean']}{ci}   (crude = 0, oracle = 100; reference controller = {ref['mean']})",
        f"finished {c['finish_rate']:.0%}  crashed {c['crashed']}  stalled {c['stalled']}  timeouts {c['timeouts']}  "
        f"errors {c['errors']}   "
        f"bumps/episode {c['edge_hits_per_episode']}  wall contact/episode {c['edge_contact_s_per_episode']} s",
        f"mean finish time {c['mean_finish_time_s']} s   decisions {c['decisions']}",
    ]
    if lat["p50"] is not None:
        lines.append(f"latency p50 {lat['p50']} ms  p95 {lat['p95']} ms  max {lat['max']} ms   "
                     f"tokens in {c['tokens']['input']} out {c['tokens']['output']}")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
