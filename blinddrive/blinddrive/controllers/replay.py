"""Replays a recorded action sequence. Used to re-run and audit logged episodes."""

from __future__ import annotations

from collections.abc import Sequence

from ..actions import Action
from ..observation import Observation


class ReplayExhausted(RuntimeError):
    pass


class ReplayController:
    name = "replay"

    def __init__(self, actions: Sequence[Action]):
        self._actions = tuple(actions)
        self._i = 0

    def reset(self) -> None:
        self._i = 0

    def act(self, observation: Observation) -> Action:
        if self._i >= len(self._actions):
            raise ReplayExhausted("replay ran out of recorded actions")
        action = self._actions[self._i]
        self._i += 1
        return action
