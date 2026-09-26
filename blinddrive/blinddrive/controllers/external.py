"""Placeholder for future external / AI controllers (e.g. JEV).

Deliberately NOT implemented. There is no fake AI and no heuristic stand-in.
A future implementation must only use the Observation it is given and must
return a legal Action; it gets no other access to the environment.
"""

from __future__ import annotations

from ..actions import Action
from ..observation import Observation


class ExternalController:
    name = "external"

    def reset(self) -> None:
        raise NotImplementedError("ExternalController is a stub; no AI is wired up yet")

    def act(self, observation: Observation) -> Action:
        raise NotImplementedError("ExternalController is a stub; no AI is wired up yet")
