from __future__ import annotations

import inspect

from sts2_bench.runner import RunResult
from sts2_bench.seeds import DEFAULT_TEST_SEED
from sts2_bench.env import Sts2Env
from sts2_bench.process import Sts2Process
from tests.conftest import Game


def test_pytest_game_start_defaults_to_preconfigured_fixed_seed():
    signature = inspect.signature(Game.start)

    assert signature.parameters["seed"].default == DEFAULT_TEST_SEED
    assert DEFAULT_TEST_SEED == "ironclad_a0_3407"


def test_process_start_run_defaults_to_preconfigured_fixed_seed():
    signature = inspect.signature(Sts2Process.start_run)

    assert signature.parameters["seed"].default == DEFAULT_TEST_SEED


def test_env_reset_defaults_to_preconfigured_fixed_seed():
    signature = inspect.signature(Sts2Env.reset)

    assert signature.parameters["seed"].default == DEFAULT_TEST_SEED


def test_route_smoke_defaults_to_preconfigured_fixed_seed(tmp_path, monkeypatch):
    from sts2_bench import route_plan_smoke

    seen = {}

    class FakeLogger:
        def __init__(self, path):
            self.path = path

        def close(self):
            pass

    class FakeProcess:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

    def fake_run_one(*, seed, character, ascension, **kwargs):
        seen["seed"] = seed
        return RunResult(seed=seed, character=character, ascension=ascension)

    monkeypatch.setattr(route_plan_smoke, "JsonlLogger", FakeLogger)
    monkeypatch.setattr(route_plan_smoke, "Sts2Process", FakeProcess)
    monkeypatch.setattr(route_plan_smoke, "run_one", fake_run_one)
    monkeypatch.setattr(route_plan_smoke, "analyze_trace", lambda *_args, **_kwargs: {})

    route_plan_smoke.main(["--out-dir", str(tmp_path)])
    assert seen["seed"] == DEFAULT_TEST_SEED
