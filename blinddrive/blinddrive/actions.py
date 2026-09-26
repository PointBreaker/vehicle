"""The discrete action space shared by every controller (human or not).

7 steering levels x 5 throttle levels = 35 legal actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class InvalidActionError(ValueError):
    """Raised when something that is not a legal Action is submitted."""


class Steering(Enum):
    HARD_LEFT = "HARD_LEFT"
    LEFT = "LEFT"
    SLIGHT_LEFT = "SLIGHT_LEFT"
    STRAIGHT = "STRAIGHT"
    SLIGHT_RIGHT = "SLIGHT_RIGHT"
    RIGHT = "RIGHT"
    HARD_RIGHT = "HARD_RIGHT"


class Throttle(Enum):
    HARD_BRAKE = "HARD_BRAKE"
    BRAKE = "BRAKE"
    COAST = "COAST"
    ACCELERATE = "ACCELERATE"
    FULL_ACCELERATE = "FULL_ACCELERATE"


@dataclass(frozen=True)
class Action:
    steering: Steering
    throttle: Throttle

    def __post_init__(self) -> None:
        if type(self.steering) is not Steering:
            raise InvalidActionError(f"steering must be a Steering member, got {self.steering!r}")
        if type(self.throttle) is not Throttle:
            raise InvalidActionError(f"throttle must be a Throttle member, got {self.throttle!r}")

    def to_dict(self) -> dict[str, str]:
        return {"steering": self.steering.value, "throttle": self.throttle.value}

    @staticmethod
    def from_dict(data: Any) -> "Action":
        """Strict parser for serialized actions (e.g. from logs or a future API)."""
        if not isinstance(data, dict) or set(data) != {"steering", "throttle"}:
            raise InvalidActionError(f"action must be {{'steering', 'throttle'}}, got {data!r}")
        try:
            return Action(Steering(data["steering"]), Throttle(data["throttle"]))
        except ValueError as exc:
            raise InvalidActionError(str(exc)) from None

    def label(self) -> str:
        return f"{self.steering.value} + {self.throttle.value}"


ALL_ACTIONS: tuple[Action, ...] = tuple(Action(s, t) for s in Steering for t in Throttle)
DEFAULT_ACTION = Action(Steering.STRAIGHT, Throttle.COAST)


def validate_action(action: Any) -> Action:
    """Return ``action`` if it is exactly one of the 35 legal actions, else raise."""
    if type(action) is not Action:
        raise InvalidActionError(f"expected Action, got {type(action).__name__}: {action!r}")
    # Re-check fields: a frozen dataclass can still be tampered with via object.__setattr__.
    if type(action.steering) is not Steering or type(action.throttle) is not Throttle:
        raise InvalidActionError(f"malformed Action: {action!r}")
    return action
