"""Keyboard controller.

The human gets no special API: ``act`` is called once per decision tick
(same rate as any other controller) and returns a standard Action built from
the keys held down *at that moment*. Key presses between decision ticks have
no effect until the next tick.

Keys:
    W / Up          accelerate          (+Shift: full accelerate)
    S / Down        brake
    Space           hard brake
    A / Left        steer left          (+Shift: hard left)
    D / Right       steer right         (+Shift: hard right)
    Q / E           slight left / slight right
    nothing         coast, straight
"""

from __future__ import annotations

from collections.abc import Callable, Collection

from ..actions import Action, Steering, Throttle
from ..observation import Observation

# Logical key names used by keys_to_action.
KEYS = ("up", "down", "left", "right", "slight_left", "slight_right", "shift", "space")


def keys_to_action(pressed: Collection[str]) -> Action:
    shift = "shift" in pressed

    left = "left" in pressed
    right = "right" in pressed
    if left and not right:
        steering = Steering.HARD_LEFT if shift else Steering.LEFT
    elif right and not left:
        steering = Steering.HARD_RIGHT if shift else Steering.RIGHT
    elif "slight_left" in pressed and "slight_right" not in pressed:
        steering = Steering.SLIGHT_LEFT
    elif "slight_right" in pressed and "slight_left" not in pressed:
        steering = Steering.SLIGHT_RIGHT
    else:
        steering = Steering.STRAIGHT

    if "space" in pressed:
        throttle = Throttle.HARD_BRAKE
    elif "down" in pressed:
        throttle = Throttle.BRAKE
    elif "up" in pressed:
        throttle = Throttle.FULL_ACCELERATE if shift else Throttle.ACCELERATE
    else:
        throttle = Throttle.COAST
    return Action(steering, throttle)


def pygame_key_reader() -> Callable[[], set[str]]:
    import pygame

    mapping = {
        "up": (pygame.K_w, pygame.K_UP),
        "down": (pygame.K_s, pygame.K_DOWN),
        "left": (pygame.K_a, pygame.K_LEFT),
        "right": (pygame.K_d, pygame.K_RIGHT),
        "slight_left": (pygame.K_q,),
        "slight_right": (pygame.K_e,),
        "shift": (pygame.K_LSHIFT, pygame.K_RSHIFT),
        "space": (pygame.K_SPACE,),
    }

    def read() -> set[str]:
        state = pygame.key.get_pressed()
        return {name for name, keys in mapping.items() if any(state[k] for k in keys)}

    return read


class HumanController:
    name = "human"

    def __init__(self, key_reader: Callable[[], Collection[str]] | None = None):
        self._read_keys = key_reader or pygame_key_reader()

    def reset(self) -> None:
        pass

    def act(self, observation: Observation) -> Action:
        # The human sees the observation on screen; the key state is their decision.
        return keys_to_action(self._read_keys())
