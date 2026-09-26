"""Headless benchmark runs for automatic controllers (currently: Jev).

    uv run python -m blinddrive.bench --check                     # verify key + list models
    uv run python -m blinddrive.bench --dry-run --seed 42         # print the first request, send nothing
    uv run python -m blinddrive.bench --seeds 1-5 --preset normal # run 5 episodes
    uv run python -m blinddrive.bench --seeds 1,2,7 --workers 3 --edge crash

Every episode writes a JSONL log (with Jev's answers, probabilities and latency
per decision) and the run writes a summary JSON. Simulated time is paused while
waiting for the API, so network latency never changes the result.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .config import EDGE_MODES, PRESETS, GameConfig, get_preset
from .controllers.jev import JevController, build_questions, observation_to_state, public_rules
from .env import BlindDriveEnv, run_episode
from .recorder import default_log_path, write_log
from .typesafe import TypeSafeClient, TypeSafeConfig, TypeSafeError, load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent


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
    p = argparse.ArgumentParser(prog="blinddrive.bench", description="Run BlindDrive episodes with Jev.")
    p.add_argument("--controller", choices=["jev"], default="jev")
    p.add_argument("--seeds", type=parse_seeds, default=None, help="e.g. 1-10 or 3,5,8 (default: --seed)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--preset", choices=sorted(PRESETS), default="normal")
    p.add_argument("--edge", choices=EDGE_MODES, default=None, help="override the preset's edge rule")
    p.add_argument("--model", default=None, help="Jev model (default: TYPESAFE_DEFAULT_MODEL or jev-latest)")
    p.add_argument("--workers", type=int, default=1, help="episodes run in parallel (API calls are the bottleneck)")
    p.add_argument("--log-dir", default="runs")
    p.add_argument("--check", action="store_true", help="check the API key and list models, then exit")
    p.add_argument("--dry-run", action="store_true", help="print the first request body, send nothing")
    return p.parse_args(argv)


def run_one(config: GameConfig, seed: int, model: str | None, log_dir: str) -> dict[str, Any]:
    env = BlindDriveEnv(config, seed)
    controller = JevController.from_env(config, model=model)
    meta = {"model": controller.model, "questions": controller.questions, "rules": controller.rules}
    path = default_log_path(log_dir, seed, config.name, controller.name)
    try:
        result = run_episode(env, controller)
    except TypeSafeError as err:
        write_log(path, env, controller.name, error=str(err), controller_meta=meta)
        return {"seed": seed, "status": "error", "error": str(err), "decisions": env.decision_count,
                "distance": round(env.debug_view().progress, 1), "log": str(path)}
    write_log(path, env, controller.name, controller_meta=meta)
    latencies = [d["controller_info"]["latency_ms"] for d in env.decision_log if "controller_info" in d]
    return {
        "seed": seed,
        "status": result.termination,
        "time": round(result.elapsed_time, 2),
        "distance": round(result.distance_travelled, 1),
        "avg_speed": round(result.average_speed, 2),
        "max_speed": round(result.max_speed, 2),
        "edge_hits": result.edge_hits,
        "decisions": result.decision_count,
        "api_latency_ms_mean": round(sum(latencies) / len(latencies), 1) if latencies else None,
        "log": str(path),
    }


def main(argv: list[str] | None = None) -> int:
    load_dotenv(Path.cwd() / ".env", PROJECT_DIR / ".env")
    args = parse_args(argv)
    config = get_preset(args.preset)
    if args.edge:
        config = config.with_edge(args.edge)
    seeds = args.seeds or [args.seed]

    if args.dry_run:
        env = BlindDriveEnv(config, seeds[0])
        body = {
            "state": observation_to_state(env.observe(), public_rules(config)),
            "model": args.model or "<TYPESAFE_DEFAULT_MODEL or jev-latest>",
            "questions": build_questions(config),
        }
        print("POST /v1/systemone  (first decision of seed %d, not sent)" % seeds[0])
        print(json.dumps(body, indent=2, ensure_ascii=False))
        return 0

    try:
        client = TypeSafeClient(TypeSafeConfig.from_env(model=args.model))
        if args.check:
            models = client.list_models()
            print(f"OK: {client.config.base_url} accepted the key. Models:")
            for m in models:
                print(f"  {m.get('name')}  ({m.get('release_date', '?')})  {m.get('description', '')}")
            print(f"Using: {client.config.model}")
            return 0
    except TypeSafeError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2

    print(f"BlindDrive x Jev  preset={config.name} edge={config.sim.edge} model={client.config.model} "
          f"seeds={seeds[0]}..{seeds[-1]} ({len(seeds)})")
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(run_one, config, s, args.model, args.log_dir) for s in seeds]
        rows = []
        for fut in futures:
            row = fut.result()
            rows.append(row)
            print(format_row(row), flush=True)

    summary = summarize(rows)
    print("\n" + format_summary(summary))
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.log_dir) / f"bench-{stamp}_{config.name}_{client.config.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"preset": config.name, "edge": config.sim.edge, "model": client.config.model,
                               "config": config.to_dict(), "summary": summary, "episodes": rows}, indent=2))
    print(f"summary: {out}")
    return 0 if summary["errors"] == 0 else 1


def format_row(r: dict[str, Any]) -> str:
    if r["status"] == "error":
        return f"  seed {r['seed']:>6}  ERROR after {r['decisions']} decisions ({r['distance']} m): {r['error']}"
    return (f"  seed {r['seed']:>6}  {r['status']:<8} time {r['time']:>6.2f}s  dist {r['distance']:>6.1f} m  "
            f"avg {r['avg_speed']:>5.2f} m/s  bumps {r['edge_hits']:>2}  decisions {r['decisions']:>4}  "
            f"api {r['api_latency_ms_mean']} ms")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    done = [r for r in rows if r["status"] != "error"]
    finished = [r for r in done if r["status"] == "finished"]
    mean = lambda xs: round(sum(xs) / len(xs), 2) if xs else None  # noqa: E731
    return {
        "episodes": len(rows),
        "finished": len(finished),
        "crashed": sum(r["status"] == "crashed" for r in done),
        "timeouts": sum(r["status"] == "timeout" for r in done),
        "errors": len(rows) - len(done),
        "mean_finish_time": mean([r["time"] for r in finished]),
        "mean_avg_speed": mean([r["avg_speed"] for r in done]),
        "mean_edge_hits": mean([r["edge_hits"] for r in done]),
        "total_decisions": sum(r["decisions"] for r in rows),
    }


def format_summary(s: dict[str, Any]) -> str:
    return (f"finished {s['finished']}/{s['episodes']}  crashed {s['crashed']}  timeouts {s['timeouts']}  "
            f"errors {s['errors']}\nmean finish time {s['mean_finish_time']} s  mean avg speed "
            f"{s['mean_avg_speed']} m/s  mean bumps {s['mean_edge_hits']}  API calls {s['total_decisions']}")


if __name__ == "__main__":
    sys.exit(main())
