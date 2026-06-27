"""Preconfigured fixed seeds used by tests and default benchmark runs."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS_FILE = ROOT / "sts2_bench" / "seeds_ironclad_a0.json"
DEFAULT_TEST_SEED = "ironclad_a0_3407"


def read_seed_file(path: str | Path) -> list[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(data, list):
        seeds = []
        for item in data:
            if isinstance(item, str):
                seeds.append(item)
            elif isinstance(item, dict) and item.get("seed"):
                seeds.append(str(item["seed"]))
            else:
                raise ValueError(f"Unsupported seed entry: {item!r}")
        return seeds
    raise ValueError("Seed file must be a JSON list of strings or objects with a seed field")
