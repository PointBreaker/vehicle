import pytest

from blinddrive.actions import ALL_ACTIONS, Action, InvalidActionError, Steering, Throttle
from blinddrive.config import normal
from blinddrive.env import BlindDriveEnv, ProtocolError


def test_action_space_has_35_actions():
    assert len(ALL_ACTIONS) == 35
    assert len(set(ALL_ACTIONS)) == 35


@pytest.mark.parametrize("steering, throttle", [
    ("LEFT", Throttle.BRAKE),
    (Steering.LEFT, "BRAKE"),
    (0, Throttle.COAST),
    (Steering.LEFT, None),
    (Throttle.BRAKE, Steering.LEFT),
])
def test_malformed_action_rejected(steering, throttle):
    with pytest.raises(InvalidActionError):
        Action(steering, throttle)


@pytest.mark.parametrize("bad", [
    None, "LEFT", 3, ("LEFT", "BRAKE"), {"steering": "LEFT", "throttle": "BRAKE"},
])
def test_env_rejects_non_actions(bad):
    env = BlindDriveEnv(normal(), 1)
    with pytest.raises(InvalidActionError):
        env.apply_action(bad)


def test_env_rejects_tampered_action():
    a = Action(Steering.LEFT, Throttle.BRAKE)
    object.__setattr__(a, "steering", "TURBO")
    env = BlindDriveEnv(normal(), 1)
    with pytest.raises(InvalidActionError):
        env.apply_action(a)


@pytest.mark.parametrize("data", [
    {"steering": "TURBO_LEFT", "throttle": "BRAKE"},
    {"steering": "LEFT"},
    {"steering": "LEFT", "throttle": "BRAKE", "boost": True},
    "LEFT+BRAKE",
])
def test_from_dict_is_strict(data):
    with pytest.raises(InvalidActionError):
        Action.from_dict(data)


def test_roundtrip():
    for a in ALL_ACTIONS:
        assert Action.from_dict(a.to_dict()) == a


def test_decision_schedule_is_enforced():
    env = BlindDriveEnv(normal(), 1)
    with pytest.raises(ProtocolError):
        env.tick()  # no action yet
    a = Action(Steering.STRAIGHT, Throttle.ACCELERATE)
    env.apply_action(a)
    with pytest.raises(ProtocolError):
        env.apply_action(a)  # second decision in the same interval
    for _ in range(env.ticks_per_decision - 1):
        env.tick()
        assert not env.decision_due
    env.tick()
    assert env.decision_due
    assert env.ticks_per_decision == 60 // 4
