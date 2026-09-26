import json
import math

import pytest

from blinddrive import bench
from blinddrive.actions import ALL_ACTIONS, Action, Steering, Throttle
from blinddrive.config import normal
from blinddrive.controllers import Controller
from blinddrive.controllers.jev import (
    JevController, JevDecisionError, action_label, observation_to_state, parse_action_label, public_rules,
)
from blinddrive.env import BlindDriveEnv, run_episode
from blinddrive.recorder import load_log, verify_log
from blinddrive.typesafe import (
    TypeSafeAPIError, TypeSafeClient, TypeSafeConfig, TypeSafeError, load_dotenv,
)

from .fake_typesafe import FakeTypeSafe


def client_for(fake, **kw):
    return TypeSafeClient(TypeSafeConfig(api_key=kw.pop("key", fake.key), base_url=fake.url, **kw))


def crash_cfg():
    return normal().with_edge("crash")


def test_wire_format_matches_typesafe_api():
    with FakeTypeSafe() as fake:
        cfg = crash_cfg()
        ctl = JevController(cfg, client_for(fake, model="jev-latest"))
        env = BlindDriveEnv(cfg, 3)
        action = ctl.act(env.observe())
    body, headers = fake.requests[0], fake.headers[0]
    assert headers["Authorization"] == "Bearer test-key"
    assert headers["Content-Type"] == "application/json"
    assert set(body) == {"state", "model", "questions"}
    assert body["model"] == "jev-latest"
    assert list(body["questions"]) == ["action"]
    question = body["questions"]["action"]
    assert question["type"] == "choice"
    assert list(question["criteria"]) == [action_label(a) for a in ALL_ACTIONS]
    assert len(question["criteria"]) == 35
    assert (action.steering, action.throttle) == (Steering.STRAIGHT, Throttle.ACCELERATE)
    info = ctl.last_info
    assert info["request_id"] == "req-1"
    assert info["answer"]["choice"] == "STRAIGHT+ACCELERATE"
    assert info["answer"]["confidence"] == 0.7
    assert math.isclose(sum(info["answer"]["probabilities"].values()), 1.0)


def test_action_labels_roundtrip_and_are_strict():
    for a in ALL_ACTIONS:
        assert parse_action_label(action_label(a)) == a
    for bad in ["LEFT", "LEFT+BRAKE+BRAKE", "left+brake", "TURBO+BRAKE", "LEFT+", None, 3]:
        with pytest.raises(JevDecisionError):
            parse_action_label(bad)


def test_jev_controller_satisfies_interface():
    with FakeTypeSafe() as fake:
        assert isinstance(JevController(normal(), client_for(fake)), Controller)


def test_state_is_only_observation_and_rules():
    cfg = normal()
    env = BlindDriveEnv(cfg, 3)
    obs = env.observe()
    state = observation_to_state(obs, public_rules(cfg))
    text = json.dumps(state)
    assert set(state) == {"car", "road_ahead", "road_beyond_visible", "progress", "previous_action",
                          "coordinates", "rules"}
    assert "seed" not in text
    assert max(p["along_road_m"] for p in state["road_ahead"]) <= cfg.sim.lookahead
    assert state["road_ahead"][0]["centerline_forward_m"] == 0.0
    # Deterministic: same observation, same state.
    assert observation_to_state(obs, public_rules(cfg)) == state


def test_retries_transient_errors():
    with FakeTypeSafe() as fake:
        fake.script = [(503, {"message": "busy"}), (429, {"message": "slow down"})]
        ctl = JevController(crash_cfg(), client_for(fake))
        ctl.act(BlindDriveEnv(crash_cfg(), 1).observe())
    assert ctl.last_info["attempts"] == 3
    assert len(fake.requests) == 3


def test_auth_error_is_not_retried():
    with FakeTypeSafe() as fake:
        ctl = JevController(crash_cfg(), client_for(fake, key="wrong"))
        with pytest.raises(TypeSafeAPIError) as err:
            ctl.act(BlindDriveEnv(crash_cfg(), 1).observe())
    assert err.value.status == 401
    assert "Invalid API key" in str(err.value)
    assert len(fake.requests) == 1


def test_illegal_label_aborts_instead_of_guessing():
    with FakeTypeSafe(steering="TURBO") as fake:
        ctl = JevController(crash_cfg(), client_for(fake))
        with pytest.raises(JevDecisionError):
            ctl.act(BlindDriveEnv(crash_cfg(), 1).observe())


def test_full_episode_logs_jev_answers(tmp_path):
    with FakeTypeSafe(steering="STRAIGHT", throttle="FULL_ACCELERATE") as fake:
        cfg = crash_cfg()
        env = BlindDriveEnv(cfg, 5)
        result = run_episode(env, JevController(cfg, client_for(fake)))
    assert result.decision_count == len(fake.requests) > 3
    for rec in env.decision_log:
        chosen = parse_action_label(rec["controller_info"]["answer"]["choice"])
        assert chosen == Action.from_dict(rec["action"])


def test_missing_key_gives_clear_error():
    with pytest.raises(TypeSafeError, match="TYPESAFE_API_KEY"):
        TypeSafeConfig.from_env(env={})
    with pytest.raises(TypeSafeError):
        TypeSafeConfig.from_env(env={"TYPESAFE_API_KEY": "<your key here>"})
    cfg = TypeSafeConfig.from_env(env={"TYPESAFE_API_KEY": "abc", "TYPESAFE_DEFAULT_MODEL": "jev-x",
                                       "TYPESAFE_BASE_URL": "http://h/"})
    assert (cfg.model, cfg.base_url) == ("jev-x", "http://h")
    assert "abc" not in repr(cfg)


def test_dotenv_does_not_override_real_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text('# comment\nTYPESAFE_API_KEY="from-file"\nexport OTHER_VAR=1\n')
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("OTHER_VAR", raising=False)
    load_dotenv(env_file)
    import os
    assert os.environ["TYPESAFE_API_KEY"] == "from-file"
    assert os.environ["OTHER_VAR"] == "1"
    monkeypatch.setenv("TYPESAFE_API_KEY", "real")
    load_dotenv(env_file)
    assert os.environ["TYPESAFE_API_KEY"] == "real"


def test_bench_cli_end_to_end(tmp_path, monkeypatch, capsys):
    with FakeTypeSafe(steering="SLIGHT_LEFT", throttle="FULL_ACCELERATE") as fake:
        monkeypatch.setenv("TYPESAFE_API_KEY", fake.key)
        monkeypatch.setenv("TYPESAFE_BASE_URL", fake.url)
        assert bench.main(["--check"]) == 0
        code = bench.main(["--seeds", "1-2", "--edge", "crash", "--log-dir", str(tmp_path), "--workers", "2"])
    out = capsys.readouterr().out
    assert code == 0
    assert "jev-latest" in out and "finished 0/2" in out
    logs = sorted(tmp_path.glob("*_jev.jsonl"))
    assert len(logs) == 2
    for p in logs:
        log = load_log(p)
        assert log.header["controller_meta"]["questions"]["action"]["type"] == "choice"
        assert verify_log(log) == []
    summary = json.loads(next(tmp_path.glob("bench-*.json")).read_text())
    assert summary["summary"]["episodes"] == 2


def test_bench_records_api_failure(tmp_path, monkeypatch, capsys):
    with FakeTypeSafe() as fake:
        fake.script = [(200, {"model": "jev-latest", "usage": {}, "answers": {}})] * 1
        monkeypatch.setenv("TYPESAFE_API_KEY", fake.key)
        monkeypatch.setenv("TYPESAFE_BASE_URL", fake.url)
        code = bench.main(["--seed", "4", "--log-dir", str(tmp_path)])
    assert code == 1
    assert "ERROR" in capsys.readouterr().out
    log = load_log(next(tmp_path.glob("*_jev.jsonl")))
    assert log.error is not None and log.result is None
    assert verify_log(log) == []


def test_dry_run_sends_nothing(capsys, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert bench.main(["--dry-run", "--seed", "7"]) == 0
    body = json.loads(capsys.readouterr().out.split("\n", 1)[1])
    assert set(body) == {"state", "model", "questions"}


def test_parse_seeds():
    assert bench.parse_seeds("1-3,7") == [1, 2, 3, 7]
