"""Top-down pygame renderer, in a Scratch-like cartoon style.

Normal mode draws the road *only* from the Observation, i.e. the human sees
exactly the road a controller would see. Nothing beyond the lookahead is drawn;
the grass decorations are placed from a fixed hash of world position and are
independent of the road, so they carry no information about it.

Debug mode (``--debug``) additionally uses the privileged DebugView to draw the
full road, the car footprint and a mini-map. It is a development tool only.

Rendering is decoupled from physics: the runner may draw several frames per
physics tick and passes an interpolated ``camera`` pose for smooth motion.
The observation itself always comes from the latest physics tick.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pygame
import pygame.gfxdraw

from .actions import Action, Throttle
from .config import GameConfig
from .env import DebugView, EpisodeResult
from .observation import Observation
from .vehicle import VehicleState, steering_target

# ---------------------------------------------------------------- palette
# Scratch 3 block / UI colours.
PURPLE = (133, 92, 214)
PURPLE_DARK = (113, 78, 182)
MOTION = (76, 151, 255)
MOTION_DARK = (51, 115, 204)
CONTROL = (255, 171, 25)
CONTROL_DARK = (207, 139, 23)
EVENTS = (255, 191, 0)
EVENTS_DARK = (204, 153, 0)
OPERATORS = (89, 192, 89)
OPERATORS_DARK = (56, 148, 56)
PINK = (255, 102, 128)
ORANGE_PILL = (255, 140, 26)
INK = (87, 94, 117)          # Scratch text colour
WHITE = (255, 255, 255)
GLOW = (255, 230, 0)

# Stage.
GRASS = (156, 216, 118)
TUFT = (120, 190, 90)
FLOWERS = ((255, 255, 255), (255, 214, 64), (255, 140, 170))
ROAD = (128, 134, 160)
ROAD_OUTLINE = (78, 82, 108)
CLOUD = (255, 255, 255)
CLOUD_EDGE = (205, 214, 232)
MONITOR_BG = (233, 241, 252)
MONITOR_EDGE = (196, 207, 224)
DEBUG_ROAD = (170, 60, 220)
DEBUG_FOOT = (0, 150, 70)

# Cartoon car and its driver.
CAR = MOTION
CAR_DARK = MOTION_DARK
GLASS = (214, 238, 255)
CAT = CONTROL
CAT_DARK = CONTROL_DARK
TYRE = (70, 72, 90)

TOP_BAR = 48
DECO_CELL = 4.0        # metres; one possible grass decoration per cell
CURB_PERIOD = 2.0      # metres per curb colour block
DASH_PERIOD = 3.0      # metres per centerline dash
CURB_WIDTH = 0.45      # metres, painted inside the road edge
MAX_PX_PER_M = 22.0


@dataclass(frozen=True)
class Hud:
    seed: int
    controller: str
    action: Action | None
    seconds_since_decision: float
    fps: float = 0.0
    edge_hits: int = 0
    seconds_since_bump: float = 999.0
    detail: str | None = None   # e.g. Jev's confidence and latency for the last decision


def _hash(i: int, j: int) -> int:
    h = (i * 73856093) ^ (j * 19349663) ^ 0x5BD1E995
    h = (h ^ (h >> 13)) * 1274126177
    return (h ^ (h >> 16)) & 0xFFFFFFFF


class Renderer:
    def __init__(self, config: GameConfig, debug: bool = False, size: tuple[int, int] = (1000, 800)):
        pygame.display.init()
        pygame.font.init()
        pygame.display.set_caption("BlindDrive" + (" [DEBUG]" if debug else ""))
        self.screen = pygame.display.set_mode(size)
        self.w, self.h = size
        self.debug = debug
        self.fonts = {k: pygame.font.Font(None, k) for k in (18, 20, 22, 24, 28, 34, 44)}
        self._text_cache: dict[tuple, pygame.Surface] = {}
        self.flag_rect = pygame.Rect(self.w - 104, 8, 40, 32)
        self.stop_rect = pygame.Rect(self.w - 56, 8, 40, 32)
        self.set_config(config)
        self._cam = (0.0, 0.0, 1.0, 0.0)

    def set_config(self, config: GameConfig) -> None:
        self.config = config
        self.car_px = (self.w / 2, TOP_BAR + (self.h - TOP_BAR) * 0.74)
        lookahead = config.sim.lookahead
        self.ppm = min(MAX_PX_PER_M, (self.car_px[1] - TOP_BAR - 50) / lookahead,
                       (self.w / 2 - 30) / (0.8 * lookahead))

    # ------------------------------------------------------------ helpers

    def text(self, s: str, size: int, color=INK) -> pygame.Surface:
        key = (s, size, color)
        surf = self._text_cache.get(key)
        if surf is None:
            if len(self._text_cache) > 600:
                self._text_cache.clear()
            surf = self._text_cache[key] = self.fonts[size].render(s, True, color)
        return surf

    def to_screen(self, x: float, y: float) -> tuple[float, float]:
        """Camera-local (x forward, y left) -> screen. The camera car always points up."""
        return (self.car_px[0] - y * self.ppm, self.car_px[1] - x * self.ppm)

    def world_to_screen(self, x: float, y: float) -> tuple[float, float]:
        cx, cy, c, s = self._cam
        dx, dy = x - cx, y - cy
        return self.to_screen(c * dx + s * dy, -s * dx + c * dy)

    def _set_camera(self, camera: VehicleState) -> None:
        self._cam = (camera.x, camera.y, math.cos(camera.heading), math.sin(camera.heading))

    def _obs_to_world(self, obs_state: VehicleState):
        c, s = math.cos(obs_state.heading), math.sin(obs_state.heading)
        return lambda p: (obs_state.x + c * p[0] - s * p[1], obs_state.y + s * p[0] + c * p[1])

    def _poly(self, color, pts, outline=None, width=2) -> None:
        pts = [(round(x), round(y)) for x, y in pts]
        pygame.gfxdraw.filled_polygon(self.screen, pts, color)
        pygame.gfxdraw.aapolygon(self.screen, pts, outline or color)
        if outline and width > 1:
            pygame.draw.polygon(self.screen, outline, pts, width)

    def _circle(self, color, center, r, outline=None) -> None:
        x, y, r = round(center[0]), round(center[1]), max(1, round(r))
        if outline:
            pygame.gfxdraw.filled_circle(self.screen, x, y, r + 2, outline)
            pygame.gfxdraw.aacircle(self.screen, x, y, r + 2, outline)
        pygame.gfxdraw.filled_circle(self.screen, x, y, r, color)
        pygame.gfxdraw.aacircle(self.screen, x, y, r, color)

    def _round_rect(self, rect, color, outline=None, radius=8, width=2) -> None:
        pygame.draw.rect(self.screen, color, rect, border_radius=radius)
        if outline:
            pygame.draw.rect(self.screen, outline, rect, width, border_radius=radius)

    # ------------------------------------------------------------ frame

    def draw(
        self,
        obs: Observation,
        obs_state: VehicleState,
        camera: VehicleState,
        hud: Hud,
        debug_view: DebugView | None = None,
        message: str | None = None,
        result: EpisodeResult | None = None,
    ) -> None:
        """``obs_state`` is the car pose the observation was taken at;
        ``camera`` is the (possibly interpolated) pose to draw from."""
        self._set_camera(camera)
        self.screen.fill(GRASS)
        self._draw_decorations()
        if self.debug and debug_view is not None:
            self._draw_full_road(debug_view)
        self._draw_visible_road(obs, obs_state)
        self._draw_car(obs.steering_angle)
        if obs.touching_edge:
            self._draw_sparks(obs.lateral_offset)
        if self.debug and debug_view is not None:
            self._draw_debug(debug_view)
        self._draw_monitors(obs, hud)
        self._draw_sliders(obs, hud)
        self._draw_script(hud)
        self._draw_top_bar(hud, running=message is None and result is None)
        if hud.controller == "human":
            self._draw_help()
        if message is None and result is None and hud.seconds_since_bump < 0.7:
            message = "Bump!"
        if message:
            self._draw_speech(message)
        if result:
            self._draw_result(result)
        pygame.display.flip()

    # ------------------------------------------------------------ stage

    def _draw_decorations(self) -> None:
        # World-fixed grass tufts and flowers: a pure function of world position,
        # independent of the road. They only show your own motion.
        cx, cy, _, _ = self._cam
        reach = math.hypot(self.w, self.h) / self.ppm * 0.62
        i0, i1 = math.floor((cx - reach) / DECO_CELL), math.ceil((cx + reach) / DECO_CELL)
        j0, j1 = math.floor((cy - reach) / DECO_CELL), math.ceil((cy + reach) / DECO_CELL)
        r = max(2, self.ppm * 0.18)
        for i in range(i0, i1):
            for j in range(j0, j1):
                h = _hash(i, j)
                kind = h % 7
                if kind > 2:
                    continue
                wx = (i + ((h >> 3) & 255) / 256) * DECO_CELL
                wy = (j + ((h >> 11) & 255) / 256) * DECO_CELL
                sx, sy = self.world_to_screen(wx, wy)
                if not (-10 < sx < self.w + 10 and TOP_BAR - 10 < sy < self.h + 10):
                    continue
                if kind == 0:  # flower
                    color = FLOWERS[(h >> 20) % 3]
                    for k in range(4):
                        a = k * math.pi / 2
                        self._circle(color, (sx + math.cos(a) * r, sy + math.sin(a) * r), r * 0.8)
                    self._circle((255, 196, 0), (sx, sy), r * 0.6)
                else:  # grass tuft
                    for k in (-1, 0, 1):
                        pygame.draw.line(self.screen, TUFT, (sx + k * r, sy),
                                         (sx + k * r * 1.8, sy - r * (2.6 - abs(k))), 2)

    def _thick_path(self, pts, width_px, color_of_segment, join) -> None:
        """Polyline with round joins where ``join[i]`` is set (none near the ends,
        so the visible road stops flat instead of bulging past its last point)."""
        w = max(1, round(width_px))
        n = len(pts)
        for i in range(n - 1):
            a, b = pts[i], pts[i + 1]
            dx, dy = b[0] - a[0], b[1] - a[1]
            norm = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / norm * w / 2, dx / norm * w / 2
            color = color_of_segment(i)
            quad = [(a[0] + nx, a[1] + ny), (b[0] + nx, b[1] + ny), (b[0] - nx, b[1] - ny), (a[0] - nx, a[1] - ny)]
            pygame.draw.polygon(self.screen, color, quad)
            if join[i]:
                pygame.draw.circle(self.screen, color, a, w / 2)

    def _draw_visible_road(self, obs: Observation, obs_state: VehicleState) -> None:
        if len(obs.visible_centerline) < 2:
            return
        to_world = self._obs_to_world(obs_state)
        pts = [self.world_to_screen(*to_world(p)) for p in obs.visible_centerline]
        abs_s = [obs.distance_travelled + a for a in obs.visible_arclengths]
        ppm = self.ppm
        road_px = obs.road_width * ppm
        arcs = obs.visible_arclengths
        margin = obs.road_width / 2 + 0.5
        join = [arcs[0] + margin <= a <= arcs[-1] - margin for a in arcs]

        self._thick_path(pts, road_px + 6, lambda i: ROAD_OUTLINE, join)
        self._thick_path(pts, road_px, lambda i: PINK if int(abs_s[i] // CURB_PERIOD) % 2 else WHITE, join)
        self._thick_path(pts, road_px - 2 * CURB_WIDTH * ppm, lambda i: ROAD, join)
        for i in range(len(pts) - 1):
            if int(abs_s[i] // DASH_PERIOD) % 2 == 0:
                pygame.draw.line(self.screen, WHITE, pts[i], pts[i + 1], max(2, round(ppm * 0.25)))

        # Finish line, if it is inside the visible window.
        d = obs.distance_to_finish
        if arcs[0] <= d <= arcs[-1]:
            for i in range(len(arcs) - 1):
                if arcs[i] <= d <= arcs[i + 1]:
                    t = (d - arcs[i]) / max(1e-9, arcs[i + 1] - arcs[i])
                    a, b = pts[i], pts[i + 1]
                    self._draw_checker((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])),
                                       (b[0] - a[0], b[1] - a[1]), road_px)
                    break

        # Clouds sit on the visibility limit: the road beyond is unknown.
        end, before = pts[-1], pts[-2]
        dx, dy = end[0] - before[0], end[1] - before[1]
        norm = math.hypot(dx, dy) or 1.0
        fx, fy = dx / norm, dy / norm           # along the road (screen)
        nx, ny = -fy, fx                        # across the road
        puff = road_px * 0.22 + 8
        for k, (across, along, scale) in enumerate(
                ((-0.62, 0.35, 0.9), (-0.3, 0.2, 1.1), (0.0, 0.45, 1.25), (0.32, 0.15, 1.05), (0.64, 0.4, 0.9))):
            c = (end[0] + nx * across * road_px + fx * along * puff,
                 end[1] + ny * across * road_px + fy * along * puff)
            self._circle(CLOUD, c, puff * scale, outline=CLOUD_EDGE)
        q = self.text("?", 44, (150, 160, 190))
        self.screen.blit(q, (end[0] + fx * puff * 0.5 - q.get_width() / 2, end[1] + fy * puff * 0.5 - q.get_height() / 2))

    def _draw_checker(self, center, direction, road_px) -> None:
        norm = math.hypot(*direction) or 1.0
        fx, fy = direction[0] / norm, direction[1] / norm
        nx, ny = -fy, fx
        n = 8
        cell = road_px / n
        for row in range(2):
            for k in range(n):
                off = -road_px / 2 + k * cell
                base = (center[0] + nx * off + fx * row * cell, center[1] + ny * off + fy * row * cell)
                quad = [base, (base[0] + nx * cell, base[1] + ny * cell),
                        (base[0] + nx * cell + fx * cell, base[1] + ny * cell + fy * cell),
                        (base[0] + fx * cell, base[1] + fy * cell)]
                pygame.draw.polygon(self.screen, WHITE if (k + row) % 2 else INK, quad)

    def _draw_car(self, steering_angle: float) -> None:
        v = self.config.vehicle
        hl, hw = v.length / 2, v.width / 2
        P = lambda pts: [self.to_screen(x, y) for x, y in pts]  # noqa: E731

        # Tyres (front ones show the actual steering angle).
        axle = v.wheelbase / 2
        wl, ww = 0.42, 0.2
        for ax, ang in ((axle, steering_angle), (-axle, 0.0)):
            for side in (hw - 0.05, -hw + 0.05):
                c, s = math.cos(ang), math.sin(ang)
                quad = [(ax + dx * c - dy * s, side + dx * s + dy * c)
                        for dx, dy in ((wl, ww), (wl, -ww), (-wl, -ww), (-wl, ww))]
                self._poly(TYRE, P(quad))

        # Body: rounded rectangle via a polygon with chamfered corners.
        r = 0.45
        body = [(hl, hw - r), (hl - r * 0.3, hw - r * 0.3), (hl - r, hw), (-hl + r, hw), (-hl + r * 0.3, hw - r * 0.3),
                (-hl, hw - r), (-hl, -hw + r), (-hl + r * 0.3, -hw + r * 0.3), (-hl + r, -hw), (hl - r, -hw),
                (hl - r * 0.3, -hw + r * 0.3), (hl, -hw + r)]
        self._poly(CAR, P(body), outline=CAR_DARK, width=3)
        # Windscreen and rear window.
        self._poly(GLASS, P([(hl * 0.62, hw * 0.72), (hl * 0.62, -hw * 0.72), (hl * 0.2, -hw * 0.8), (hl * 0.2, hw * 0.8)]),
                   outline=CAR_DARK)
        self._poly(GLASS, P([(-hl * 0.62, hw * 0.7), (-hl * 0.62, -hw * 0.7), (-hl * 0.8, -hw * 0.6), (-hl * 0.8, hw * 0.6)]),
                   outline=CAR_DARK)
        # Headlights.
        for side in (hw * 0.6, -hw * 0.6):
            self._circle((255, 245, 180), self.to_screen(hl - 0.12, side), self.ppm * 0.16)

        # The driver: a Scratch-style cat seen from above.
        head_x, rad = -0.25, 0.5
        for side in (1, -1):
            ear = [(head_x + rad * 1.45, side * rad * 0.35), (head_x + rad * 0.55, side * rad * 0.95),
                   (head_x + rad * 0.35, side * rad * 0.15)]
            self._poly(CAT, P(ear), outline=CAT_DARK)
        hx, hy = self.to_screen(head_x, 0.0)
        self._circle(CAT, (hx, hy), rad * self.ppm, outline=CAT_DARK)
        # Stripes on the back of the head.
        for k in (-1, 0, 1):
            a = self.to_screen(head_x - rad * 0.2, k * rad * 0.3)
            b = self.to_screen(head_x - rad * 0.8, k * rad * 0.35)
            pygame.draw.line(self.screen, CAT_DARK, a, b, 2)

    def _draw_sparks(self, lateral_offset: float) -> None:
        """Little star bursts on the side of the car that is scraping the edge."""
        v = self.config.vehicle
        side = 1.0 if lateral_offset > 0 else -1.0
        t = pygame.time.get_ticks() / 1000.0
        for k, along in enumerate((v.length * 0.4, -v.length * 0.35)):
            cx, cy = self.to_screen(along, side * (v.width / 2 + 0.15))
            r = self.ppm * (0.45 + 0.2 * math.sin(t * 40 + k * 2.1))
            rot = t * 9 + k
            star = []
            for i in range(10):
                rr = r if i % 2 == 0 else r * 0.42
                a = rot + i * math.pi / 5
                star.append((cx + math.cos(a) * rr, cy + math.sin(a) * rr))
            self._poly((255, 214, 0), star, outline=(255, 120, 0), width=2)

    # ------------------------------------------------------------ debug

    def _draw_full_road(self, dv: DebugView) -> None:
        road = dv.road
        pts = [self.world_to_screen(x, y) for x, y in zip(road.xs[::4], road.ys[::4])]
        half = road.width / 2 * self.ppm
        for side in (1, -1):
            edge = []
            for i, (x, y) in enumerate(pts):
                a, b = pts[max(0, i - 1)], pts[min(len(pts) - 1, i + 1)]
                dx, dy = b[0] - a[0], b[1] - a[1]
                n = math.hypot(dx, dy) or 1.0
                edge.append((x - dy / n * half * side, y + dx / n * half * side))
            pygame.draw.lines(self.screen, DEBUG_ROAD, False, edge, 2)
        pygame.draw.lines(self.screen, DEBUG_ROAD, False, pts, 1)
        fx, fy, fh = road.pose_at(road.length)
        hw = road.width / 2
        a = self.world_to_screen(fx - math.sin(fh) * hw, fy + math.cos(fh) * hw)
        b = self.world_to_screen(fx + math.sin(fh) * hw, fy - math.cos(fh) * hw)
        pygame.draw.line(self.screen, OPERATORS_DARK, a, b, 3)

    def _draw_debug(self, dv: DebugView) -> None:
        foot = [self.world_to_screen(x, y) for x, y in dv.footprint]
        pygame.draw.polygon(self.screen, DEBUG_FOOT, foot, 2)
        tip = (dv.state.x + math.cos(dv.state.heading) * max(2.0, dv.state.speed * 0.5),
               dv.state.y + math.sin(dv.state.heading) * max(2.0, dv.state.speed * 0.5))
        pygame.draw.line(self.screen, DEBUG_FOOT, self.world_to_screen(dv.state.x, dv.state.y),
                         self.world_to_screen(*tip), 3)
        road = dv.road
        box = pygame.Rect(self.w - 232, self.h - 262, 216, 216)
        self._round_rect(box, WHITE, DEBUG_ROAD)
        minx, maxx, miny, maxy = min(road.xs), max(road.xs), min(road.ys), max(road.ys)
        k = 190 / max(maxx - minx, maxy - miny, 1.0)
        mm = lambda x, y: (box.x + 13 + (x - minx) * k, box.bottom - 13 - (y - miny) * k)  # noqa: E731
        pygame.draw.lines(self.screen, DEBUG_ROAD, False, [mm(x, y) for x, y in zip(road.xs[::8], road.ys[::8])], 2)
        self._circle(PINK, mm(dv.state.x, dv.state.y), 5)
        self.screen.blit(self.text("DEBUG: full road visible", 20, DEBUG_ROAD), (box.x, box.y - 20))

    # ------------------------------------------------------------ UI

    def _draw_top_bar(self, hud: Hud, running: bool) -> None:
        pygame.draw.rect(self.screen, PURPLE, (0, 0, self.w, TOP_BAR))
        pygame.draw.line(self.screen, PURPLE_DARK, (0, TOP_BAR - 1), (self.w, TOP_BAR - 1), 2)
        self.screen.blit(self.text("BlindDrive", 34, WHITE), (16, 13))
        tag = f"seed {hud.seed}  ·  {self.config.name}  ·  {hud.controller}"
        pill = self.text(tag, 22, WHITE)
        rect = pygame.Rect(170, 12, pill.get_width() + 24, 26)
        self._round_rect(rect, PURPLE_DARK, radius=13)
        self.screen.blit(pill, (rect.x + 12, rect.y + 5))
        if hud.fps:
            fps = self.text(f"{hud.fps:.0f} fps", 20, (220, 208, 250))
            self.screen.blit(fps, (self.flag_rect.x - fps.get_width() - 14, 17))

        # Green flag and stop sign, like the Scratch stage header.
        f = self.flag_rect
        self._round_rect(f, (230, 240, 255) if running else WHITE, radius=6)
        pole = f.x + 13
        pygame.draw.line(self.screen, OPERATORS_DARK, (pole, f.y + 6), (pole, f.bottom - 5), 3)
        self._poly(OPERATORS, [(pole, f.y + 6), (f.right - 8, f.y + 9), (pole + 4, f.y + 13),
                               (f.right - 8, f.y + 18), (pole, f.y + 20)], outline=OPERATORS_DARK)
        s = self.stop_rect
        self._round_rect(s, WHITE, radius=6)
        cx, cy, r = s.centerx, s.centery, 11
        octa = [(cx + r * math.cos(math.pi / 8 + k * math.pi / 4), cy + r * math.sin(math.pi / 8 + k * math.pi / 4))
                for k in range(8)]
        self._poly((236, 83, 83), octa, outline=(184, 50, 50))

    def _monitor(self, x: int, y: int, label: str, value: str) -> int:
        lab = self.text(label, 22, INK)
        val = self.text(value, 22, WHITE)
        pill_w = max(44, val.get_width() + 16)
        rect = pygame.Rect(x, y, lab.get_width() + pill_w + 24, 30)
        self._round_rect(rect, MONITOR_BG, MONITOR_EDGE, radius=6)
        self.screen.blit(lab, (x + 8, y + 8))
        pill = pygame.Rect(rect.right - pill_w - 8, y + 5, pill_w, 20)
        self._round_rect(pill, ORANGE_PILL, radius=10)
        self.screen.blit(val, (pill.centerx - val.get_width() / 2, pill.y + 3))
        return rect.bottom + 6

    def _draw_monitors(self, obs: Observation, hud: Hud) -> None:
        cfg = self.config
        y = TOP_BAR + 12
        y = self._monitor(12, y, "distance", f"{obs.distance_travelled:.0f} / {cfg.road.length:.0f} m")
        y = self._monitor(12, y, "speed", f"{obs.speed:.1f} m/s")
        y = self._monitor(12, y, "steering", f"{math.degrees(obs.steering_angle):+.0f}°")
        y = self._monitor(12, y, "time", f"{obs.elapsed_time:.1f} s")
        if cfg.sim.edge == "wall":
            y = self._monitor(12, y, "bumps", f"{hud.edge_hits}")
        y = self._monitor(12, y, "visible", f"{cfg.sim.lookahead:.0f} m")
        self._monitor(12, y, "decision every", f"{1000 / cfg.sim.decision_hz:.0f} ms")

    def _draw_sliders(self, obs: Observation, hud: Hud) -> None:
        v = self.config.vehicle
        rect = pygame.Rect(self.w - 232, TOP_BAR + 12, 220, 124)
        self._round_rect(rect, MONITOR_BG, MONITOR_EDGE, radius=8)
        x0, width = rect.x + 12, rect.w - 24

        self.screen.blit(self.text("speed", 22), (x0, rect.y + 10))
        track = pygame.Rect(x0, rect.y + 34, width, 10)
        self._round_rect(track, (210, 218, 232), radius=5)
        frac = min(1.0, obs.speed / v.max_speed)
        if frac > 0.02:
            self._round_rect(pygame.Rect(track.x, track.y, track.w * frac, track.h), MOTION, radius=5)
        self._circle(WHITE, (track.x + track.w * frac, track.centery), 8, outline=MOTION_DARK)

        self.screen.blit(self.text("steering", 22), (x0, rect.y + 60))
        self.screen.blit(self.text("knob = wheels, | = target", 18, (140, 148, 170)), (x0 + 72, rect.y + 63))
        track = pygame.Rect(x0, rect.y + 86, width, 10)
        self._round_rect(track, (210, 218, 232), radius=5)
        full = math.radians(v.max_steering_angle_deg)
        mid = track.centerx
        pygame.draw.line(self.screen, MONITOR_EDGE, (mid, track.y - 3), (mid, track.bottom + 3), 2)
        if hud.action:
            tgt = mid - steering_target(hud.action.steering, v) / full * track.w / 2
            pygame.draw.line(self.screen, INK, (tgt, track.y - 6), (tgt, track.bottom + 6), 3)
        cur = mid - obs.steering_angle / full * track.w / 2
        self._circle(WHITE, (cur, track.centery), 8, outline=CONTROL_DARK)

    def _block(self, x, y, w, h, color, dark, hat=False, glow=0.0) -> None:
        """A Scratch stack block (notch on top, tab below); ``hat`` adds the rounded cap."""
        n0, n1, d = 12, 30, 5
        pts = [(x, y + 4), (x + 4, y), (x + n0, y), (x + n0 + 5, y + d), (x + n1 - 5, y + d), (x + n1, y),
               (x + w - 4, y), (x + w, y + 4), (x + w, y + h - 4), (x + w - 4, y + h),
               (x + n1, y + h), (x + n1 - 5, y + h + d), (x + n0 + 5, y + h + d), (x + n0, y + h),
               (x + 4, y + h), (x, y + h - 4)]
        if hat:
            pts = [(x, y + 4), (x + 4, y)] + [(x + w - 4, y), (x + w, y + 4)] + pts[8:]
        if glow > 0:
            g = tuple(int(GLOW[k] * glow + color[k] * (1 - glow)) for k in range(3))
            pygame.draw.polygon(self.screen, g, pts, 7)
            if hat:
                pygame.draw.ellipse(self.screen, g, (x - 3, y - 17, 86, 40), 7)
        if hat:
            pygame.draw.ellipse(self.screen, dark, (x - 1, y - 15, 82, 36))
            pygame.draw.ellipse(self.screen, color, (x + 1, y - 13, 78, 32))
        self._poly(color, pts, outline=dark, width=2)
        if hat:
            pygame.draw.ellipse(self.screen, color, (x + 3, y - 11, 74, 28))
            pygame.draw.rect(self.screen, color, (x + 2, y + 2, w - 4, 12))

    def _dropdown(self, x, y, label, fill, edge) -> int:
        t = self.text(label, 20, WHITE)
        rect = pygame.Rect(x, y, t.get_width() + 30, 22)
        self._round_rect(rect, fill, edge, radius=11, width=1)
        self.screen.blit(t, (x + 9, y + 5))
        ax, ay = rect.right - 14, rect.centery - 2
        pygame.draw.polygon(self.screen, WHITE, [(ax - 4, ay), (ax + 4, ay), (ax, ay + 4)])
        return rect.right

    def _draw_script(self, hud: Hud) -> None:
        # The current decision, shown as a tiny Scratch script.
        x, y = 16, self.h - 170
        glow = max(0.0, 1.0 - hud.seconds_since_decision / 0.15)
        interval = f"{1000 / self.config.sim.decision_hz:.0f}"
        hat_w = 250
        self._block(x, y, hat_w, 34, EVENTS, EVENTS_DARK, hat=True, glow=glow)
        self.screen.blit(self.text(f"when decision tick  (every {interval} ms)", 20, WHITE), (x + 10, y + 11))
        a = hud.action
        y += 34
        steer = a.steering.value.replace("_", " ").lower() if a else "-"
        self._block(x, y, 200, 36, MOTION, MOTION_DARK)
        self.screen.blit(self.text("steer", 22, WHITE), (x + 10, y + 11))
        self._dropdown(x + 62, y + 7, steer, MOTION_DARK, MOTION_DARK)
        y += 36
        throttle = a.throttle.value.replace("_", " ").lower() if a else "-"
        brake = a is not None and a.throttle in (Throttle.BRAKE, Throttle.HARD_BRAKE)
        color, dark = (PINK, (230, 70, 100)) if brake else (CONTROL, CONTROL_DARK)
        self._block(x, y, 220, 36, color, dark)
        self.screen.blit(self.text("throttle", 22, WHITE), (x + 10, y + 11))
        self._dropdown(x + 82, y + 7, throttle, dark, dark)
        if hud.detail:
            t = self.text(hud.detail, 18, INK)
            rect = pygame.Rect(x, y + 48, t.get_width() + 16, 20)
            self._round_rect(rect, MONITOR_BG, MONITOR_EDGE, radius=10, width=1)
            self.screen.blit(t, (x + 8, y + 52))

    def _draw_help(self) -> None:
        msg = "W/Up gas (+Shift full)   S/Down brake   Space hard brake   A D / Left Right steer (+Shift hard)   Q E slight"
        t = self.text(msg, 18, INK)
        rect = pygame.Rect(0, 0, t.get_width() + 20, 22)
        rect.midbottom = (self.w / 2, self.h - 6)
        rect.x = max(rect.x, 290)
        self._round_rect(rect, (255, 255, 255), MONITOR_EDGE, radius=11, width=1)
        self.screen.blit(t, (rect.x + 10, rect.y + 5))

    def _draw_speech(self, message: str) -> None:
        lines = message.split("\n")
        surfs = [self.text(line, 24 if i == 0 else 20) for i, line in enumerate(lines)]
        w = max(s.get_width() for s in surfs) + 32
        h = sum(s.get_height() + 6 for s in surfs) + 22
        anchor = self.to_screen(0.4, -self.config.vehicle.width / 2)
        rect = pygame.Rect(anchor[0] + 36, anchor[1] - h - 40, w, h)
        rect.right = min(rect.right, self.w - 12)
        tail = [(rect.x + 22, rect.bottom - 2), (rect.x + 48, rect.bottom - 2), (anchor[0] + 8, anchor[1] - 6)]
        self._poly(WHITE, tail, outline=(200, 205, 215))
        self._round_rect(rect, WHITE, (200, 205, 215), radius=16)
        pygame.draw.polygon(self.screen, WHITE, [(tail[0][0] + 2, tail[0][1] - 3), (tail[1][0] - 2, tail[1][1] - 3),
                                                 (tail[2][0] + 3, tail[2][1] - 5)])
        yy = rect.y + 13
        for s in surfs:
            self.screen.blit(s, (rect.x + 16, yy))
            yy += s.get_height() + 6

    def _draw_result(self, result: EpisodeResult) -> None:
        if result.success:
            title, color, dark = f"Finished in {result.completion_time:.2f} s!", OPERATORS, OPERATORS_DARK
        elif result.termination == "crashed":
            title, color, dark = f"Crashed at {result.crash_distance:.1f} m", PINK, (230, 70, 100)
        else:
            title, color, dark = "Time's up", CONTROL, CONTROL_DARK
        card = pygame.Rect(0, 0, 440, 276)
        card.center = (self.w / 2, TOP_BAR + (self.h - TOP_BAR) * 0.38)
        shadow = card.move(0, 6)
        pygame.draw.rect(self.screen, (120, 170, 95), shadow, border_radius=18)
        self._round_rect(card, WHITE, MONITOR_EDGE, radius=18)
        head = pygame.Rect(card.x, card.y, card.w, 56)
        pygame.draw.rect(self.screen, color, head, border_top_left_radius=18, border_top_right_radius=18)
        pygame.draw.line(self.screen, dark, (head.x, head.bottom), (head.right, head.bottom), 2)
        t = self.text(title, 34, WHITE)
        self.screen.blit(t, (card.centerx - t.get_width() / 2, head.y + 16))
        rows = [
            ("distance", f"{result.distance_travelled:.1f} m"),
            ("average speed", f"{result.average_speed:.1f} m/s"),
            ("top speed", f"{result.max_speed:.1f} m/s"),
            ("decisions", f"{result.decision_count}"),
        ]
        if self.config.sim.edge == "wall":
            rows.insert(3, ("bumps", f"{result.edge_hits}  ({result.edge_contact_time:.1f} s on the wall)"))
        y = head.bottom + 14
        for label, value in rows:
            self.screen.blit(self.text(label, 22), (card.x + 30, y))
            v = self.text(value, 22, ORANGE_PILL)
            self.screen.blit(v, (card.right - 30 - v.get_width(), y))
            y += 26
        keys = self.text("R  retry      N  new seed      Esc  quit", 22, (140, 148, 170))
        self.screen.blit(keys, (card.centerx - keys.get_width() / 2, card.bottom - 36))
