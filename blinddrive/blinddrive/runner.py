"""Command-line entry point.

    uv run python -m blinddrive.runner                     # random seed, normal preset
    uv run python -m blinddrive.runner --seed 42 --preset hard
    uv run python -m blinddrive.runner --seed 42 --debug   # shows the full road (dev only)
    uv run python -m blinddrive.runner --replay runs/<file>.jsonl

One physics tick is simulated per rendered frame (60 fps), so simulated time
runs at wall-clock speed when the machine keeps up, and slows down (never
skips) when it does not. The simulation itself is identical to the headless one.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from .config import PRESETS, get_preset
from .controllers import HumanController, ReplayController, controller_name
from .controllers.base import Controller
from .env import BlindDriveEnv
from .recorder import default_log_path, load_log, write_log


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="blinddrive", description="BlindDrive: drive with limited vision.")
    p.add_argument("--seed", type=int, default=None, help="road seed (default: random)")
    p.add_argument("--preset", choices=sorted(PRESETS), default="normal")
    p.add_argument("--debug", action="store_true", help="show the FULL road and internals (development only)")
    p.add_argument("--log-dir", default="runs", help="directory for episode JSONL logs")
    p.add_argument("--no-log", action="store_true", help="do not write episode logs")
    p.add_argument("--replay", metavar="FILE", help="visually replay a recorded episode log")
    return p.parse_args(argv)


def play(env: BlindDriveEnv, controller: Controller, *, debug: bool, log_dir: str | None,
         autostart: bool = False, allow_new_seed: bool = True) -> None:
    import pygame

    from .renderer import Hud, Renderer, result_lines

    renderer = Renderer(env.config, debug=debug)
    clock = pygame.time.Clock()
    name = controller_name(controller)

    def start_episode() -> None:
        env.reset()
        controller.reset()

    start_episode()
    phase = "running" if autostart else "ready"
    ticks_since_decision = 999

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if event.type != pygame.KEYDOWN:
                continue
            if phase == "ready" and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                phase = "running"
            elif phase == "over" and event.key == pygame.K_r:
                start_episode()
                phase = "ready"
            elif phase == "over" and event.key == pygame.K_n and allow_new_seed:
                env = BlindDriveEnv(env.config, random.randrange(1_000_000))
                renderer.config = env.config
                start_episode()
                phase = "ready"

        if phase == "running":
            if env.decision_due:
                env.apply_action(controller.act(env.observe()))
                ticks_since_decision = 0
            env.tick()
            ticks_since_decision += 1
            if env.done:
                phase = "over"
                report(env, name, log_dir)

        overlay, color = None, (230, 234, 240)
        if phase == "ready":
            overlay = [
                "Press ENTER to start",
                f"seed {env.seed}  ·  preset {env.config.name}  ·  {env.config.road.length:.0f} m",
                f"You see {env.config.sim.lookahead:.0f} m ahead. You decide every "
                f"{1000 / env.config.sim.decision_hz:.0f} ms.",
                "Leave the road and the run is over. Nobody will save you.",
            ]
        elif phase == "over":
            overlay, color = result_lines(env.result)

        dv = env.debug_view()  # privileged: used for the motion grid and --debug only
        renderer.draw(
            env.observe(),
            dv.state,
            Hud(seed=env.seed, controller=name, action=env.current_action,
                ticks_since_decision=ticks_since_decision),
            debug_view=dv if debug else None,
            overlay=overlay,
            overlay_color=color,
        )
        clock.tick(env.config.sim.physics_hz)


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
        play(env, ReplayController(log.actions), debug=args.debug, log_dir=None,
             autostart=True, allow_new_seed=False)
        return 0

    seed = args.seed if args.seed is not None else random.randrange(1_000_000)
    config = get_preset(args.preset)
    print(f"BlindDrive  seed={seed}  preset={config.name}")
    env = BlindDriveEnv(config, seed)
    play(env, HumanController(), debug=args.debug, log_dir=None if args.no_log else args.log_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
