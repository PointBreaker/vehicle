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
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from .config import EDGE_MODES, PRESETS, get_preset
from .controllers import HumanController, ReplayController, controller_name
from .controllers.base import Controller
from .env import BlindDriveEnv
from .recorder import default_log_path, load_log, write_log
from .vehicle import VehicleState

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
    p.add_argument("--fps", type=int, default=120, help="render frame-rate cap (0 = uncapped); physics stays fixed")
    return p.parse_args(argv)


def play(env: BlindDriveEnv, controller: Controller, *, debug: bool, log_dir: str | None,
         fps: int = 120, autostart: bool = False, allow_new_seed: bool = True) -> None:
    import pygame

    from .renderer import Hud, Renderer

    renderer = Renderer(env.config, debug=debug)
    clock = pygame.time.Clock()
    name = controller_name(controller)
    physics_dt = 1.0 / env.config.sim.physics_hz

    def start_episode() -> None:
        nonlocal last_hits
        env.reset()
        controller.reset()
        last_hits = 0

    last_hits = 0
    start_episode()
    phase = "running" if autostart else "ready"
    accumulator = 0.0
    since_decision = 999.0
    since_bump = 999.0
    prev_state = env.debug_view().state

    while True:
        frame_dt = clock.tick(fps) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            start = (event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER)) or (
                event.type == pygame.MOUSEBUTTONDOWN and renderer.flag_rect.collidepoint(event.pos))
            if event.type == pygame.MOUSEBUTTONDOWN and renderer.stop_rect.collidepoint(event.pos):
                pygame.quit()
                return
            if phase == "ready" and start:
                phase, accumulator = "running", 0.0
            elif phase == "over" and (start or (event.type == pygame.KEYDOWN and event.key == pygame.K_r)):
                start_episode()
                phase, prev_state = "ready", env.debug_view().state
            elif phase == "over" and event.type == pygame.KEYDOWN and event.key == pygame.K_n and allow_new_seed:
                env = BlindDriveEnv(env.config, random.randrange(1_000_000))
                start_episode()
                phase, prev_state = "ready", env.debug_view().state

        # Fixed-step physics driven by wall-clock time. Rendering runs at its own
        # rate and interpolates between the last two physics states. If the
        # machine falls far behind, simulated time slows down instead of skipping.
        if phase == "running":
            accumulator += min(frame_dt, MAX_FRAME_TIME)
            since_decision += frame_dt
            since_bump += frame_dt
            while accumulator >= physics_dt and not env.done:
                prev_state = env.debug_view().state
                if env.decision_due:
                    env.apply_action(controller.act(env.observe()))
                    since_decision = 0.0
                env.tick()
                accumulator -= physics_dt
                if env.edge_hits != last_hits:
                    since_bump, last_hits = 0.0, env.edge_hits
            if env.done:
                phase, accumulator = "over", 0.0
                report(env, name, log_dir)

        dv = env.debug_view()  # privileged: used for camera motion and --debug only
        alpha = min(1.0, accumulator / physics_dt) if phase == "running" else 1.0
        message = None
        if phase == "ready":
            message = (f"Ready? Press ENTER or click the green flag!\n"
                       f"I can only see {env.config.sim.lookahead:.0f} m ahead, and I decide every "
                       f"{1000 / env.config.sim.decision_hz:.0f} ms.\n"
                       + ("Bumping the edge slows us down a lot." if env.config.sim.edge == "wall"
                          else "If we leave the road, the run is over."))
        renderer.draw(
            env.observe(),
            dv.state,
            interpolate(prev_state, dv.state, alpha),
            Hud(seed=env.seed, controller=name, action=env.current_action,
                seconds_since_decision=since_decision, fps=clock.get_fps(),
                edge_hits=env.edge_hits, seconds_since_bump=since_bump),
            debug_view=dv if debug else None,
            message=message,
            result=env.result if phase == "over" else None,
        )


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


def report(env: BlindDriveEnv, name: str, log_dir: str | None) -> None:
    r = env.result
    print(f"[{r.termination}] seed={r.seed} preset={r.preset} distance={r.distance_travelled:.1f}m "
          f"time={r.elapsed_time:.2f}s avg={r.average_speed:.2f}m/s max={r.max_speed:.2f}m/s "
          f"decisions={r.decision_count}")
    if log_dir is not None:
        path = write_log(default_log_path(log_dir, env.seed, env.config.name, name), env, name)
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
    print(f"BlindDrive  seed={seed}  preset={config.name}")
    env = BlindDriveEnv(config, seed)
    play(env, HumanController(), debug=args.debug, log_dir=None if args.no_log else args.log_dir, fps=args.fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
