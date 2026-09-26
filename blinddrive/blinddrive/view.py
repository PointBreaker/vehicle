"""What a renderer is given each frame, and the renderer interface.

The game loop (runner.py) builds one ``Frame`` per rendered frame and hands it
to any object with ``draw(frame)`` / ``close()``. The current implementation is
the minimal top-down pygame view in renderer.py; a 3D view only needs to
implement the same two methods.

Only ``obs`` describes the road. A normal (non-debug) renderer must draw the
road from ``obs`` alone, so the player never sees more than a controller does.
``obs_state`` / ``camera`` are the car's own pose (for placing the camera), and
``debug_view`` (the full road) is set only in --debug mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .actions import Action
from .config import GameConfig
from .env import DebugView, EpisodeResult
from .observation import Observation
from .vehicle import VehicleState


@dataclass(frozen=True)
class Frame:
    config: GameConfig
    obs: Observation                 # the road as the controller sees it
    obs_state: VehicleState          # car pose at which ``obs`` was taken (latest physics tick)
    camera: VehicleState             # interpolated car pose to draw from (smooth motion)
    seed: int
    controller: str
    action: Action | None            # action currently being held
    status: str                      # "ready" | "running" | "thinking" | "over" | "error"
    edge_hits: int = 0
    message: str | None = None       # short text to show (start prompt, error, ...)
    result: EpisodeResult | None = None
    detail: str | None = None        # controller note, e.g. Jev confidence / latency
    fps: float = 0.0
    debug_view: DebugView | None = None


class View(Protocol):
    def draw(self, frame: Frame) -> None: ...

    def close(self) -> None: ...
