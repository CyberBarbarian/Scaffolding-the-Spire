from __future__ import annotations

import inspect

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
