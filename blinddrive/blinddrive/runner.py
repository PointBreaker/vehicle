"""Command-line entry point.

    uv run python -m blinddrive.runner                     # random seed, normal preset
    uv run python -m blinddrive.runner --seed 42 --preset hard
    uv run python -m blinddrive.runner --seed 42 --debug   # shows the full road (dev only)
    uv run python -m blinddrive.runner --replay runs/<file>.jsonl
    uv run python -m blinddrive.runner --fps 144           # render frame-rate cap

Physics always runs at the fixed ``physics_hz`` (and decisions at ``decision_hz``);
the render frame rate only affects how smoothly it is drawn. The simulation is
identical to the headless one.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import textwrap
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from .config import EDGE_MODES, PRESETS, get_preset
from .controllers import HumanController, ReplayController, controller_name
from .controllers.base import Controller
from .env import BlindDriveEnv, apply_decision
from .recorder import default_log_path, load_log, write_log
from .vehicle import VehicleState
from .view import Frame, View

MAX_FRAME_TIME = 0.1  # seconds; longer frames are clipped (simulation slows down)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="blinddrive", description="BlindDrive: drive with limited vision.")
    p.add_argument("--seed", type=int, default=None, help="road seed (default: random)")
    p.add_argument("--preset", choices=sorted(PRESETS), default="normal")
    p.add_argument("--edge", choices=EDGE_MODES, default=None,
                   help="road edge: 'wall' slows you down (default), 'crash' ends the run")
    p.add_argument("--debug", action="store_true", help="show the FULL road and internals (development only)")
    p.add_argument("--log-dir", default="runs", help="directory for episode JSONL logs")
    p.add_argument("--no-log", action="store_true", help="do not write episode logs")
    p.add_argument("--replay", metavar="FILE", help="visually replay a recorded episode log")
    p.add_argument("--controller", choices=["human", "jev"], default="human",
                   help="who drives: you (keyboard) or Jev (needs TYPESAFE_API_KEY, e.g. in .env)")
    p.add_argument("--model", default=None, help="Jev model (default: TYPESAFE_DEFAULT_MODEL or jev-latest)")
    p.add_argument("--fps", type=int, default=120, help="render frame-rate cap (0 = uncapped); physics stays fixed")
    return p.parse_args(argv)


def play(env: BlindDriveEnv, controller: Controller, *, debug: bool, log_dir: str | None,
         fps: int = 120, autostart: bool = False, allow_new_seed: bool = True,
         view: View | None = None) -> None:
    """Interactive loop: fixed-step physics, a view drawn at up to ``fps``.

    ``view`` is anything implementing ``draw(Frame)`` / ``close()`` (default: the
    minimal top-down pygame view). Keyboard input is read through pygame.
    """
    import pygame

    if view is None:
        from .renderer import TopDownRenderer

        view = TopDownRenderer(debug=debug)
    clock = pygame.time.Clock()
    name = controller_name(controller)
    physics_dt = 1.0 / env.config.sim.physics_hz
    # Remote controllers (network calls) are asked on a worker thread so the window
    # stays responsive. Simulated time is paused until the answer arrives.
    remote = bool(getattr(controller, "remote", False))
    executor = ThreadPoolExecutor(max_workers=1) if remote else None
    pending: Future | None = None
    thinking_since = 0.0
    error_text: str | None = None

    def start_episode() -> None:
        env.reset()
        controller.reset()

    def quit_() -> None:
        view.close()
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    start_episode()
    phase = "running" if autostart else "ready"
    accumulator = 0.0
    prev_state = env.debug_view().state

    while True:
        frame_dt = clock.tick(fps) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                quit_()
                return
            if event.type != pygame.KEYDOWN:
                continue
            if phase == "ready" and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                phase, accumulator = "running", 0.0
            elif phase in ("over", "error") and event.key in (pygame.K_r, pygame.K_RETURN, pygame.K_KP_ENTER):
                start_episode()
                phase, prev_state, error_text = "ready", env.debug_view().state, None
            elif phase in ("over", "error") and event.key == pygame.K_n and allow_new_seed:
                env = BlindDriveEnv(env.config, random.randrange(1_000_000))
                start_episode()
                phase, prev_state, error_text = "ready", env.debug_view().state, None

        # Fixed-step physics driven by wall-clock time. The view runs at its own
        # rate and interpolates between the last two physics states. If the
        # machine falls far behind, simulated time slows down instead of skipping.
        if phase == "running":
            accumulator += min(frame_dt, MAX_FRAME_TIME)
            while accumulator >= physics_dt and not env.done:
                if env.decision_due:
                    if remote:
                        if pending is None:
                            pending = executor.submit(controller.act, env.observe())
                            thinking_since = time.monotonic()
                        if not pending.done():
                            accumulator = 0.0      # simulation paused while the controller thinks
                            prev_state = env.debug_view().state
                            break
                        future, pending = pending, None
                        try:
                            action = future.result()
                        except Exception as err:  # noqa: BLE001 - shown to the user, episode aborted
                            error_text = f"{type(err).__name__}: {err}"
                            phase = "error"
                            report_error(env, controller, name, log_dir, error_text)
                            break
                    else:
                        action = controller.act(env.observe())
                    apply_decision(env, controller, action)
                prev_state = env.debug_view().state
                env.tick()
                accumulator -= physics_dt
            if env.done and phase == "running":
                phase, accumulator = "over", 0.0
                report(env, controller, name, log_dir)

        dv = env.debug_view()  # privileged: used for the camera pose and --debug only
        alpha = min(1.0, accumulator / physics_dt) if phase == "running" else 1.0
        status, message = phase, None
        if phase == "ready":
            message = (f"press Enter to start\n"
                       f"{env.config.sim.lookahead:.0f} m visible · decision every "
                       f"{1000 / env.config.sim.decision_hz:.0f} ms")
        elif phase == "error":
            message = "controller failed\n" + _wrap(error_text or "", 80) + "\nR retry · N new seed · Esc quit"
        elif pending is not None and time.monotonic() - thinking_since > 0.15:
            status, message = "thinking", f"thinking… {time.monotonic() - thinking_since:.1f} s"
        view.draw(Frame(
            config=env.config,
            obs=env.observe(),
            obs_state=dv.state,
            camera=interpolate(prev_state, dv.state, alpha),
            seed=env.seed,
            controller=name,
            action=env.current_action,
            status=status,
            edge_hits=env.edge_hits,
            message=message,
            result=env.result if phase == "over" else None,
            detail=controller_detail(controller),
            fps=clock.get_fps(),
            debug_view=dv if debug else None,
        ))


def interpolate(a: VehicleState, b: VehicleState, t: float) -> VehicleState:
    """Pose between two physics states, for drawing only."""
    dh = (b.heading - a.heading + math.pi) % (2 * math.pi) - math.pi
    return VehicleState(
        x=a.x + (b.x - a.x) * t,
        y=a.y + (b.y - a.y) * t,
        heading=a.heading + dh * t,
        speed=a.speed + (b.speed - a.speed) * t,
        steering_angle=a.steering_angle + (b.steering_angle - a.steering_angle) * t,
    )


def _wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width)[:4])


def controller_detail(controller: Controller) -> str | None:
    """One HUD line from a controller's last_info (e.g. Jev's confidence and latency)."""
    info = getattr(controller, "last_info", None)
    if not info:
        return None
    parts = [controller_name(controller)]
    conf = (info.get("answer") or {}).get("confidence")
    if isinstance(conf, (int, float)):
        parts.append(f"confidence {conf:.0%}")
    if "latency_ms" in info:
        parts.append(f"{info['latency_ms']:.0f} ms")
    return "  ·  ".join(parts)


def controller_meta(controller: Controller) -> dict | None:
    if not hasattr(controller, "questions"):
        return None
    return {"model": getattr(controller, "model", None), "questions": controller.questions,
            "rules": getattr(controller, "rules", None)}


def report_error(env: BlindDriveEnv, controller: Controller, name: str, log_dir: str | None, error: str) -> None:
    print(f"[error] seed={env.seed} after {env.decision_count} decisions: {error}")
    if log_dir is not None:
        path = write_log(default_log_path(log_dir, env.seed, env.config.name, name), env, name,
                         error=error, controller_meta=controller_meta(controller))
        print(f"  log: {path}")


def report(env: BlindDriveEnv, controller: Controller, name: str, log_dir: str | None) -> None:
    r = env.result
    print(f"[{r.termination}] seed={r.seed} preset={r.preset} distance={r.distance_travelled:.1f}m "
          f"time={r.elapsed_time:.2f}s avg={r.average_speed:.2f}m/s max={r.max_speed:.2f}m/s "
          f"decisions={r.decision_count}")
    if log_dir is not None:
        path = write_log(default_log_path(log_dir, env.seed, env.config.name, name), env, name,
                         controller_meta=controller_meta(controller))
        print(f"  log: {path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.replay:
        log = load_log(args.replay)
        env = BlindDriveEnv(log.config, log.seed)
        print(f"Replaying {Path(args.replay).name}: seed={log.seed} preset={log.header['preset']} "
              f"controller={log.header['controller']}")
        play(env, ReplayController(log.actions), debug=args.debug, log_dir=None, fps=args.fps,
             autostart=True, allow_new_seed=False)
        return 0

    seed = args.seed if args.seed is not None else random.randrange(1_000_000)
    config = get_preset(args.preset)
    if args.edge:
        config = config.with_edge(args.edge)
    if args.controller == "jev":
        from .controllers.jev import JevController
        from .typesafe import TypeSafeError, load_dotenv

        load_dotenv(Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env")
        try:
            controller = JevController.from_env(config, model=args.model)
        except TypeSafeError as err:
            print(f"ERROR: {err}", file=sys.stderr)
            return 2
    else:
        controller = HumanController()
    print(f"BlindDrive  seed={seed}  preset={config.name}  controller={controller_name(controller)}")
    env = BlindDriveEnv(config, seed)
    play(env, controller, debug=args.debug, log_dir=None if args.no_log else args.log_dir, fps=args.fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
