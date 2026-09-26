"""Minimal top-down pygame view: the visible road, the car, one line of text.

Implements the ``View`` interface from view.py. The road is drawn only from
``frame.obs``; beyond the lookahead there is nothing. ``--debug`` additionally
draws the full road and the car footprint from the privileged DebugView.
"""

from __future__ import annotations

import math

import pygame

from .env import EpisodeResult
from .view import Frame

BG = (24, 26, 31)
ROAD = (66, 70, 78)
EDGE = (205, 208, 214)
DASH = (105, 110, 120)
CAR = (240, 240, 240)
CAR_FRONT = (60, 64, 72)
CAR_HIT = (235, 85, 70)
TEXT = (200, 204, 212)
TEXT_DIM = (120, 125, 135)
GOOD = (120, 210, 140)
BAD = (235, 100, 90)
DEBUG = (170, 90, 230)

DASH_PERIOD = 3.0      # metres per centerline dash
MAX_PX_PER_M = 22.0


class TopDownRenderer:
    def __init__(self, size: tuple[int, int] = (1000, 800), debug: bool = False):
        pygame.display.init()
        pygame.font.init()
        pygame.display.set_caption("BlindDrive" + (" [debug]" if debug else ""))
        self.screen = pygame.display.set_mode(size)
        self.w, self.h = size
        self.debug = debug
        self.font = pygame.font.Font(None, 22)
        self.font_big = pygame.font.Font(None, 32)
        self.car_px = (self.w / 2, self.h * 0.75)
        self._cam = (0.0, 0.0, 1.0, 0.0)
        self.ppm = MAX_PX_PER_M

    def close(self) -> None:
        pygame.quit()

    # ------------------------------------------------------------ transforms

    def to_screen(self, x: float, y: float) -> tuple[float, float]:
        """Camera-local (x forward, y left) -> screen. The car always points up."""
        return (self.car_px[0] - y * self.ppm, self.car_px[1] - x * self.ppm)

    def world_to_screen(self, x: float, y: float) -> tuple[float, float]:
        cx, cy, c, s = self._cam
        dx, dy = x - cx, y - cy
        return self.to_screen(c * dx + s * dy, -s * dx + c * dy)

    # ------------------------------------------------------------ frame

    def draw(self, frame: Frame) -> None:
        cam = frame.camera
        self._cam = (cam.x, cam.y, math.cos(cam.heading), math.sin(cam.heading))
        lookahead = frame.config.sim.lookahead
        self.ppm = min(MAX_PX_PER_M, (self.car_px[1] - 40) / lookahead, (self.w / 2 - 30) / (0.8 * lookahead))

        self.screen.fill(BG)
        if frame.debug_view is not None:
            self._draw_full_road(frame)
        self._draw_road(frame)
        self._draw_car(frame)
        self._draw_text(frame)
        pygame.display.flip()

    # ------------------------------------------------------------ world

    def _draw_road(self, frame: Frame) -> None:
        obs, st = frame.obs, frame.obs_state
        if len(obs.visible_centerline) < 2:
            return
        c, s = math.cos(st.heading), math.sin(st.heading)
        pts = [self.world_to_screen(st.x + c * x - s * y, st.y + s * x + c * y) for x, y in obs.visible_centerline]
        arcs = obs.visible_arclengths
        half = obs.road_width / 2 * self.ppm

        left, right = [], []
        for i, (x, y) in enumerate(pts):
            a, b = pts[max(0, i - 1)], pts[min(len(pts) - 1, i + 1)]
            dx, dy = b[0] - a[0], b[1] - a[1]
            n = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / n * half, dx / n * half
            left.append((x + nx, y + ny))
            right.append((x - nx, y - ny))
        for i in range(len(pts) - 1):
            pygame.draw.polygon(self.screen, ROAD, (left[i], left[i + 1], right[i + 1], right[i]))
        pygame.draw.aalines(self.screen, EDGE, False, left)
        pygame.draw.aalines(self.screen, EDGE, False, right)
        for i in range(len(pts) - 1):
            if int((obs.distance_travelled + arcs[i]) // DASH_PERIOD) % 2 == 0:
                pygame.draw.line(self.screen, DASH, pts[i], pts[i + 1], 2)

        # Finish line, if visible.
        d = obs.distance_to_finish
        for i in range(len(arcs) - 1):
            if arcs[i] <= d <= arcs[i + 1]:
                t = (d - arcs[i]) / max(1e-9, arcs[i + 1] - arcs[i])
                lerp = lambda p, q: (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))  # noqa: E731
                pygame.draw.line(self.screen, GOOD, lerp(left[i], left[i + 1]), lerp(right[i], right[i + 1]), 4)
                break

    def _draw_car(self, frame: Frame) -> None:
        v = frame.config.vehicle
        hl, hw = v.length / 2, v.width / 2
        color = CAR_HIT if frame.obs.touching_edge else CAR
        body = [self.to_screen(x, y) for x, y in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]
        pygame.draw.polygon(self.screen, color, body)
        # A bar at the front shows which way the car points; it tilts with the wheels.
        a = frame.obs.steering_angle
        fx = hl - 0.35
        dx, dy = math.sin(a) * hw * 0.8, math.cos(a) * hw * 0.8
        pygame.draw.line(self.screen, CAR_FRONT, self.to_screen(fx + dx, dy), self.to_screen(fx - dx, -dy), 4)

    def _draw_full_road(self, frame: Frame) -> None:
        dv = frame.debug_view
        road = dv.road
        pts = [self.world_to_screen(x, y) for x, y in zip(road.xs[::4], road.ys[::4])]
        pygame.draw.lines(self.screen, DEBUG, False, pts, 1)
        foot = [self.world_to_screen(x, y) for x, y in dv.footprint]
        pygame.draw.polygon(self.screen, DEBUG, foot, 1)

    # ------------------------------------------------------------ text

    def _draw_text(self, frame: Frame) -> None:
        obs, cfg = frame.obs, frame.config
        parts = [
            f"seed {frame.seed} · {cfg.name} · {frame.controller}",
            f"{obs.distance_travelled:.0f} / {cfg.road.length:.0f} m",
            f"{obs.speed:.1f} m/s",
            f"{obs.elapsed_time:.1f} s",
        ]
        if cfg.sim.edge == "wall":
            parts.append(f"bumps {frame.edge_hits}")
        if frame.action:
            parts.append(frame.action.label().replace("_", " ").lower())
        self.screen.blit(self.font.render("    ".join(parts), True, TEXT), (16, 14))
        if frame.detail:
            self.screen.blit(self.font.render(frame.detail, True, TEXT_DIM), (16, 36))
        if frame.debug_view is not None:
            fps = self.font.render(f"{frame.fps:.0f} fps  [debug]", True, DEBUG)
            self.screen.blit(fps, (self.w - fps.get_width() - 16, 14))

        lines, color = [], TEXT
        if frame.result is not None:
            lines, color = result_lines(frame.result)
        elif frame.message:
            lines = frame.message.split("\n")
            color = BAD if frame.status == "error" else TEXT
        y = self.h * 0.22
        for i, line in enumerate(lines):
            font = self.font_big if i == 0 else self.font
            surf = font.render(line, True, color if i == 0 else TEXT_DIM)
            self.screen.blit(surf, (self.w / 2 - surf.get_width() / 2, y))
            y += surf.get_height() + 8


def result_lines(result: EpisodeResult) -> tuple[list[str], tuple[int, int, int]]:
    if result.success:
        head, color = f"finished in {result.completion_time:.2f} s", GOOD
    elif result.termination == "crashed":
        head, color = f"crashed at {result.crash_distance:.1f} m", BAD
    else:
        head, color = "time's up", BAD
    stats = f"avg {result.average_speed:.1f} m/s · max {result.max_speed:.1f} m/s · bumps {result.edge_hits}"
    return [head, stats, "R retry · N new seed · Esc quit"], color
