"""The one interface every controller implements.

The environment does not know (or care) whether a controller is a human,
a rule, Pure Pursuit, MPC or an AI. It only calls ``act(observation)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..actions import Action
    from ..observation import Observation


@runtime_checkable
class Controller(Protocol):
    def reset(self) -> None:
        """Called once at the start of every episode."""
        ...

    def act(self, observation: "Observation") -> "Action":
        """Return one of the 35 legal Actions. Called once per decision tick."""
        ...


def controller_name(controller: object) -> str:
    return getattr(controller, "name", type(controller).__name__)
