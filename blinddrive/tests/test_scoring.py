import json

import pytest

from blinddrive import bench, report
from blinddrive.baselines import CrudeController, ReferenceController, oracle_config, oracle_controller
from blinddrive.config import normal
from blinddrive.env import BlindDriveEnv, run_episode
from blinddrive.scoring import PENALTY_SPEED, effective_time, mean_ci, normalized_score, percentile
from blinddrive.suite import SuiteError, build, certify, fingerprint, load_suite

SUITES = __import__("pathlib").Path(__file__).resolve().parent.parent / "suites"


def test_effective_time_and_score():
    assert effective_time(30.0, 500.0, 500.0, True) == 30.0
    assert effective_time(10.0, 300.0, 500.0, False) == 10.0 + 200.0 / PENALTY_SPEED
    assert normalized_score(48.0, 48.0, 28.0) == 0.0
    assert normalized_score(28.0, 48.0, 28.0) == 100.0
    assert normalized_score(38.0, 48.0, 28.0) == 50.0
    assert normalized_score(30.0, 30.0, 30.0) is None


def test_stats_helpers():
    assert mean_ci([]) == {"mean": None, "ci95": None, "n": 0}
    m = mean_ci([10.0, 20.0, 30.0])
    assert m["mean"] == 20.0 and m["ci95"][0] < 20.0 < m["ci95"][1]
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([], 50) is None


def test_anchor_order_on_a_seed():
    cert = certify(normal(), 1000)
    assert cert["solvable"]
    assert cert["floor"]["time"] > cert["reference"]["time"] > cert["oracle"]["time"]


def test_references_only_use_observations():
    """Reference controllers must work through the normal Controller interface."""
    cfg = normal()
    for ctl in (ReferenceController(cfg), CrudeController(cfg)):
        assert run_episode(BlindDriveEnv(cfg, 3), ctl).success
    assert run_episode(BlindDriveEnv(oracle_config(cfg), 3), oracle_controller(cfg)).success


def test_suite_build_and_load(tmp_path):
    suite = build(normal(), count=2, start=1000, name="t")
    assert [s["seed"] for s in suite["seeds"]] == [1000, 1001]
    path = tmp_path / "s.json"
    path.write_text(json.dumps(suite))
    loaded = load_suite(path)
    assert loaded["game_config"] == normal()
    # Tampering with the rules is detected.
    suite["config"]["vehicle"]["max_speed"] = 99.0
    path.write_text(json.dumps(suite))
    with pytest.raises(SuiteError, match="fingerprint"):
        load_suite(path)


def test_stale_suite_detected(tmp_path):
    suite = build(normal(), count=1, start=1000, name="t")
    suite["seeds"][0]["reference"]["time"] += 1.0
    path = tmp_path / "s.json"
    path.write_text(json.dumps(suite))
    with pytest.raises(SuiteError, match="rebuild"):
        load_suite(path)


@pytest.mark.parametrize("name", ["easy-dev", "normal-dev", "hard-dev"])
def test_committed_suites_are_valid(name):
    suite = load_suite(SUITES / f"{name}.json")
    assert len(suite["seeds"]) == 20
    assert all(s["solvable"] for s in suite["seeds"])


def test_delay_is_excluded_from_fingerprint():
    cfg = normal()
    delayed = cfg.__class__(name=cfg.name, vehicle=cfg.vehicle, road=cfg.road,
                            sim=cfg.sim.__class__(action_delay_ms=250))
    assert fingerprint(delayed) == fingerprint(cfg)


def test_bench_with_reference_on_suite(tmp_path, capsys):
    code = bench.main(["--suite", str(SUITES / "normal-dev.json"), "--controller", "reference",
                       "--log-dir", str(tmp_path), "--workers", "4"])
    assert code == 0
    run = json.loads(next(tmp_path.glob("bench-*.json")).read_text())
    card = run["scorecard"]
    assert card["finish_rate"] == 1.0
    assert card["score"]["mean"] == card["reference_score"]["mean"]   # it *is* the reference
    assert 0 < card["score"]["mean"] < 100
    html = report.render([run])
    assert "normal-dev" in html and "reference" in html


def test_bench_crude_scores_zero(tmp_path, capsys):
    bench.main(["--seeds", "5", "--controller", "crude", "--log-dir", str(tmp_path)])
    run = json.loads(next(tmp_path.glob("bench-*.json")).read_text())
    assert run["scorecard"]["score"]["mean"] == 0.0
    assert run["suite"] is None
