from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

from atc_benchmark.audio.core import SAMPLE_RATE, write_canonical_wav


@pytest.fixture
def tone_audio() -> np.ndarray:
    time = np.arange(SAMPLE_RATE * 2) / SAMPLE_RATE
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 2.3 * time) ** 2
    return (
        0.15 * envelope * (np.sin(2 * np.pi * 220 * time) + 0.45 * np.sin(2 * np.pi * 880 * time))
    ).astype(np.float64)


@pytest.fixture
def reference_pool(tmp_path: Path) -> Path:
    spec = importlib.util.spec_from_file_location(
        "fixture_generator", Path("scripts/generate_fixture_audio.py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    destination = tmp_path / "reference"
    module.generate(destination)
    return destination


@pytest.fixture
def test_config(tmp_path: Path, reference_pool: Path) -> Path:
    payload = yaml.safe_load(Path("configs/showcase_v1.yaml").read_text())
    payload["dataset"]["output_name"] = "fixture_showcase"
    payload["paths"] = {
        "raw_dir": str(tmp_path / "raw"),
        "reference_dir": str(reference_pool),
        "output_dir": str(tmp_path / "output"),
    }
    path = tmp_path / "showcase_test.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


@pytest.fixture
def canonical_wav(tmp_path: Path, tone_audio: np.ndarray) -> Path:
    path = tmp_path / "tone.wav"
    write_canonical_wav(path, tone_audio)
    return path
