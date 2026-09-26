"""Top-down pygame renderer.

Normal mode draws the road *only* from the Observation, i.e. the human sees
exactly the road a controller would see. Nothing beyond the lookahead is drawn.

Debug mode (``--debug``) additionally uses the privileged DebugView to draw the
full road, the car footprint, heading and the lookahead cutoff. It is a tool
for developing the environment, not a way to play.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pygame

from .actions import Action
from .config import GameConfig
from .env import DebugView, EpisodeResult
from .observation import Observation, to_local
from .vehicle import VehicleState, steering_target

# Colours.
BG = (14, 17, 22)
GRID = (44, 54, 66)
ASPHALT = (58, 62, 70)
EDGE = (225, 228, 232)
CURB_RED = (196, 52, 52)
CENTER = (230, 200, 90)
CUTOFF = (110, 120, 135)
CAR = (60, 150, 240)
CAR_DARK = (25, 70, 125)
WHEEL = (20, 20, 20)
TEXT = (230, 234, 240)
TEXT_DIM = (140, 150, 165)
PANEL = (8, 10, 14, 200)
DEBUG_ROAD = (150, 90, 210)
DEBUG_FOOT = (80, 255, 140)
GOOD = (90, 220, 120)
BAD = (240, 90, 80)

GRID_SPACING = 5.0      # metres between background grid dots (motion cue only)
CURB_PERIOD = 2.0       # metres per curb colour block
DASH_PERIOD = 3.0       # metres per centerline dash
MAX_PX_PER_M = 22.0


@dataclass(frozen=True)
class Hud:
    seed: int
    controller: str
    action: Action | None
    ticks_since_decision: int


class Renderer:
    def __init__(self, config: GameConfig, debug: bool = False, size: tuple[int, int] = (1000, 800)):
        pygame.display.init()
        pygame.font.init()
        pygame.display.set_caption("BlindDrive" + (" [DEBUG]" if debug else ""))
        self.screen = pygame.display.set_mode(size)
        self.w, self.h = size
        self.config = config
        self.debug = debug
        self.car_px = (self.w / 2, self.h * 0.76)
        lookahead = config.sim.lookahead
        self.ppm = min(MAX_PX_PER_M, (self.car_px[1] - 40) / lookahead, (self.w / 2 - 30) / (0.8 * lookahead))
        self.font = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 20)
        self.font_big = pygame.font.Font(None, 44)
        self.font_title = pygame.font.Font(None, 34)
        self.overlay_surface = pygame.Surface(size, pygame.SRCALPHA)

    # ------------------------------------------------------------ transforms

    def to_screen(self, x: float, y: float) -> tuple[float, float]:
        """Car-local (x forward, y left) -> screen pixels. The car always points up."""
        return (self.car_px[0] - y * self.ppm, self.car_px[1] - x * self.ppm)

    # ------------------------------------------------------------ frame

    def draw(
        self,
        obs: Observation,
        car: VehicleState,
        hud: Hud,
        debug_view: DebugView | None = None,
        overlay: list[str] | None = None,
        overlay_color: tuple[int, int, int] = TEXT,
    ) -> None:
        self.screen.fill(BG)
        self._draw_grid(car)
        if self.debug and debug_view is not None:
            self._draw_full_road(debug_view)
        self._draw_visible_road(obs)
        self._draw_car(obs.steering_angle)
        if self.debug and debug_view is not None:
            self._draw_debug_overlays(debug_view, obs)
        self._draw_hud(obs, hud)
        if overlay:
            self._draw_overlay(overlay, overlay_color)
        pygame.display.flip()

    # ------------------------------------------------------------ world

    def _draw_grid(self, car: VehicleState) -> None:
        # World-fixed dots: they only show your own motion, never the road.
        reach = max(self.w, self.h) / self.ppm
        gx0 = math.floor((car.x - reach) / GRID_SPACING)
        gy0 = math.floor((car.y - reach) / GRID_SPACING)
        n = int(2 * reach / GRID_SPACING) + 2
        for i in range(n):
            for j in range(n):
                lx, ly = to_local((gx0 + i) * GRID_SPACING, (gy0 + j) * GRID_SPACING, car)
                sx, sy = self.to_screen(lx, ly)
                if 0 <= sx < self.w and 0 <= sy < self.h:
                    self.screen.fill(GRID, (sx, sy, 2, 2))

    def _edges(self, pts: list[tuple[float, float]], half: float):
        left, right = [], []
        for i, (x, y) in enumerate(pts):
            a = pts[max(0, i - 1)]
            b = pts[min(len(pts) - 1, i + 1)]
            dx, dy = b[0] - a[0], b[1] - a[1]
            norm = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / norm, dx / norm
            left.append((x + nx * half, y + ny * half))
            right.append((x - nx * half, y - ny * half))
        return left, right

    def _draw_visible_road(self, obs: Observation) -> None:
        pts = list(obs.visible_centerline)
        if len(pts) < 2:
            return
        half = obs.road_width / 2
        left, right = self._edges(pts, half)
        L = [self.to_screen(*p) for p in left]
        R = [self.to_screen(*p) for p in right]
        C = [self.to_screen(*p) for p in pts]
        abs_s = [obs.distance_travelled + a for a in obs.visible_arclengths]

        for i in range(len(pts) - 1):
            pygame.draw.polygon(self.screen, ASPHALT, (L[i], L[i + 1], R[i + 1], R[i]))
        for i in range(len(pts) - 1):
            curb = CURB_RED if int(abs_s[i] // CURB_PERIOD) % 2 else EDGE
            pygame.draw.line(self.screen, curb, L[i], L[i + 1], 4)
            pygame.draw.line(self.screen, curb, R[i], R[i + 1], 4)
            if int(abs_s[i] // DASH_PERIOD) % 2 == 0:
                pygame.draw.line(self.screen, CENTER, C[i], C[i + 1], 2)

        # Finish line, if it is inside the visible window.
        d = obs.distance_to_finish
        arcs = obs.visible_arclengths
        if arcs[0] <= d <= arcs[-1]:
            for i in range(len(arcs) - 1):
                if arcs[i] <= d <= arcs[i + 1]:
                    t = (d - arcs[i]) / max(1e-9, arcs[i + 1] - arcs[i])
                    lerp = lambda p, q: (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))  # noqa: E731
                    self._draw_checker(lerp(left[i], left[i + 1]), lerp(right[i], right[i + 1]))
                    break

        # Visibility cutoff: the road simply ends here.
        pygame.draw.line(self.screen, CUTOFF, L[-1], R[-1], 1)
        lx, ly = L[-1]
        self.screen.blit(self.font_small.render(f"{obs.lookahead:.0f} m", True, CUTOFF), (lx + 6, ly - 8))

    def _draw_checker(self, a: tuple[float, float], b: tuple[float, float]) -> None:
        n = 10
        depth = 1.0
        dx, dy = (b[0] - a[0]) / n, (b[1] - a[1]) / n
        norm = math.hypot(dx, dy) or 1.0
        fx, fy = -dy / norm * depth, dx / norm * depth  # along the road (roughly)
        for row in range(2):
            for k in range(n):
                p0 = (a[0] + k * dx + row * fx, a[1] + k * dy + row * fy)
                quad = (p0, (p0[0] + dx, p0[1] + dy), (p0[0] + dx + fx, p0[1] + dy + fy), (p0[0] + fx, p0[1] + fy))
                color = (245, 245, 245) if (k + row) % 2 else (15, 15, 15)
                pygame.draw.polygon(self.screen, color, [self.to_screen(*q) for q in quad])

    def _draw_car(self, steering_angle: float) -> None:
        v = self.config.vehicle
        hl, hw = v.length / 2, v.width / 2
        body = [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]
        pygame.draw.polygon(self.screen, CAR, [self.to_screen(*p) for p in body])
        pygame.draw.polygon(self.screen, CAR_DARK, [self.to_screen(*p) for p in body], 2)
        # Windscreen.
        wind = [(hl * 0.45, hw * 0.8), (hl * 0.45, -hw * 0.8), (hl * 0.05, -hw * 0.8), (hl * 0.05, hw * 0.8)]
        pygame.draw.polygon(self.screen, CAR_DARK, [self.to_screen(*p) for p in wind])
        # Wheels: front wheels show the *actual* steering angle.
        axle = v.wheelbase / 2
        wl, ww = 0.35, 0.12
        for ax, ang in ((axle, steering_angle), (-axle, 0.0)):
            for side in (hw, -hw):
                c, s = math.cos(ang), math.sin(ang)
                quad = [(ax + dx * c - dy * s, side + dx * s + dy * c)
                        for dx, dy in ((wl, ww), (wl, -ww), (-wl, -ww), (-wl, ww))]
                pygame.draw.polygon(self.screen, WHEEL, [self.to_screen(*p) for p in quad])

    # ------------------------------------------------------------ debug

    def _draw_full_road(self, dv: DebugView) -> None:
        road = dv.road
        pts = [to_local(x, y, dv.state) for x, y in zip(road.xs[::4], road.ys[::4])]
        left, right = self._edges(pts, road.width / 2)
        for edge in (left, right):
            pygame.draw.lines(self.screen, DEBUG_ROAD, False, [self.to_screen(*p) for p in edge], 1)
        pygame.draw.lines(self.screen, (90, 60, 130), False, [self.to_screen(*p) for p in pts], 1)
        fx, fy, fh = road.pose_at(road.length)
        a = to_local(fx - math.sin(fh) * road.width / 2, fy + math.cos(fh) * road.width / 2, dv.state)
        b = to_local(fx + math.sin(fh) * road.width / 2, fy - math.cos(fh) * road.width / 2, dv.state)
        pygame.draw.line(self.screen, GOOD, self.to_screen(*a), self.to_screen(*b), 2)

    def _draw_debug_overlays(self, dv: DebugView, obs: Observation) -> None:
        foot = [self.to_screen(*to_local(x, y, dv.state)) for x, y in dv.footprint]
        pygame.draw.polygon(self.screen, DEBUG_FOOT, foot, 2)
        # Heading vector (always straight up in the car frame) and velocity length.
        tip = self.to_screen(max(2.0, dv.state.speed * 0.5), 0.0)
        pygame.draw.line(self.screen, DEBUG_FOOT, self.to_screen(0, 0), tip, 2)
        for x, y in obs.visible_centerline:
            self.screen.fill(CENTER, (*self.to_screen(x, y), 3, 3))
        # Lookahead cutoff and visible boundary in bright colour.
        x, y = obs.visible_centerline[-1]
        pygame.draw.circle(self.screen, BAD, self.to_screen(x, y), 6, 2)
        # Mini-map of the full road.
        road = dv.road
        box = pygame.Rect(self.w - 230, self.h - 230, 210, 210)
        pygame.draw.rect(self.screen, (0, 0, 0), box)
        pygame.draw.rect(self.screen, DEBUG_ROAD, box, 1)
        minx, maxx, miny, maxy = min(road.xs), max(road.xs), min(road.ys), max(road.ys)
        k = 190 / max(maxx - minx, maxy - miny, 1.0)
        mm = lambda x, y: (box.x + 10 + (x - minx) * k, box.bottom - 10 - (y - miny) * k)  # noqa: E731
        pygame.draw.lines(self.screen, DEBUG_ROAD, False, [mm(x, y) for x, y in zip(road.xs[::8], road.ys[::8])], 1)
        pygame.draw.circle(self.screen, DEBUG_FOOT, mm(dv.state.x, dv.state.y), 4)
        self.screen.blit(self.font_small.render("DEBUG: full road visible", True, BAD), (box.x, box.y - 20))

    # ------------------------------------------------------------ HUD

    def _panel(self, rect: pygame.Rect) -> None:
        surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        surf.fill(PANEL)
        self.screen.blit(surf, rect.topleft)

    def _draw_hud(self, obs: Observation, hud: Hud) -> None:
        cfg = self.config
        interval_ms = 1000 / cfg.sim.decision_hz
        action = hud.action.label() if hud.action else "-"
        lines = [
            f"Seed: {hud.seed}   Preset: {cfg.name}",
            f"Distance: {obs.distance_travelled:.0f} / {cfg.road.length:.0f} m",
            f"Speed: {obs.speed:.1f} m/s  ({obs.speed * 3.6:.0f} km/h)",
            f"Steering: {math.degrees(obs.steering_angle):+.0f}°",
            f"Action: {action}",
            f"Time: {obs.elapsed_time:.1f} s",
            "",
            f"Visible: {cfg.sim.lookahead:.0f} m",
            f"Decision interval: {interval_ms:.0f} ms",
            f"Controller: {hud.controller}",
        ]
        self._panel(pygame.Rect(12, 12, 300, 38 + 22 * len(lines)))
        self.screen.blit(self.font_title.render("BlindDrive", True, TEXT), (24, 20))
        for i, line in enumerate(lines):
            self.screen.blit(self.font.render(line, True, TEXT if i < 6 else TEXT_DIM), (24, 52 + 22 * i))

        # Decision pulse: lights up for a moment at every decision tick.
        pulse = max(0.0, 1.0 - hud.ticks_since_decision / 8)
        color = tuple(int(TEXT_DIM[k] + (GOOD[k] - TEXT_DIM[k]) * pulse) for k in range(3))
        pygame.draw.circle(self.screen, color, (290, 34), 8)

        self._draw_gauges(obs, hud)
        help_text = "W/Up accel (+Shift full)   S/Down brake   Space hard brake   A/D or Left/Right steer (+Shift hard)   Q/E slight"
        surf = self.font_small.render(help_text, True, TEXT_DIM)
        self.screen.blit(surf, (self.w / 2 - surf.get_width() / 2, self.h - 24))

    def _draw_gauges(self, obs: Observation, hud: Hud) -> None:
        v = self.config.vehicle
        rect = pygame.Rect(self.w - 232, 12, 220, 112)
        self._panel(rect)
        x0, width = rect.x + 12, rect.w - 24

        # Speed bar.
        self.screen.blit(self.font_small.render("speed", True, TEXT_DIM), (x0, rect.y + 8))
        bar = pygame.Rect(x0, rect.y + 26, width, 12)
        pygame.draw.rect(self.screen, (40, 45, 55), bar)
        frac = min(1.0, obs.speed / v.max_speed)
        pygame.draw.rect(self.screen, CAR, (bar.x, bar.y, bar.w * frac, bar.h))

        # Steering: actual angle (bar) and target (tick).
        self.screen.blit(self.font_small.render("steering  (bar = actual, | = target)", True, TEXT_DIM),
                         (x0, rect.y + 48))
        bar = pygame.Rect(x0, rect.y + 66, width, 12)
        pygame.draw.rect(self.screen, (40, 45, 55), bar)
        full = math.radians(v.max_steering_angle_deg)
        mid = bar.centerx
        cur = -obs.steering_angle / full * bar.w / 2  # left = left on screen
        pygame.draw.rect(self.screen, CENTER, (min(mid, mid + cur), bar.y, abs(cur), bar.h))
        pygame.draw.line(self.screen, TEXT_DIM, (mid, bar.y - 2), (mid, bar.bottom + 2), 1)
        if hud.action:
            tgt = mid - steering_target(hud.action.steering, v) / full * bar.w / 2
            pygame.draw.line(self.screen, TEXT, (tgt, bar.y - 4), (tgt, bar.bottom + 4), 3)

        # Throttle state.
        if hud.action:
            name = hud.action.throttle.value
            color = BAD if "BRAKE" in name else GOOD if "ACCEL" in name else TEXT_DIM
            self.screen.blit(self.font_small.render(name, True, color), (x0, rect.y + 88))

    def _draw_overlay(self, lines: list[str], color: tuple[int, int, int]) -> None:
        heights = [44 if i == 0 else 26 for i in range(len(lines))]
        box_h = sum(heights) + 40
        rect = pygame.Rect(self.w / 2 - 260, self.h / 2 - box_h / 2 - 60, 520, box_h)
        self._panel(rect)
        y = rect.y + 20
        for i, line in enumerate(lines):
            font = self.font_big if i == 0 else self.font
            surf = font.render(line, True, color if i == 0 else TEXT)
            self.screen.blit(surf, (self.w / 2 - surf.get_width() / 2, y))
            y += heights[i]


def result_lines(result: EpisodeResult) -> tuple[list[str], tuple[int, int, int]]:
    if result.success:
        head, color = f"FINISHED in {result.completion_time:.2f} s", GOOD
    elif result.termination == "crashed":
        head, color = f"CRASHED at {result.crash_distance:.1f} m", BAD
    else:
        head, color = "TIMEOUT", BAD
    return [
        head,
        f"distance {result.distance_travelled:.1f} m   avg {result.average_speed:.1f} m/s   "
        f"max {result.max_speed:.1f} m/s",
        f"decisions: {result.decision_count}",
        "",
        "R: retry same seed    N: new seed    Esc: quit",
    ], color
