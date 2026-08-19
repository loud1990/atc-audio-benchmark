from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest
import soundfile as sf
from typer.testing import CliRunner

from atc_benchmark.cli import app
from atc_benchmark.config import load_config
from atc_benchmark.metadata import read_jsonl
from atc_benchmark.pipeline import build_showcase, validate_dataset

pytestmark = pytest.mark.e2e


def audio_hashes(directory: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.glob("*.wav"))
    }


def test_full_25_scenario_build_is_complete_and_deterministic(test_config: Path) -> None:
    config = load_config(test_config)
    first = build_showcase(test_config)
    output = first.output_root
    assert len(first.records) == len(config.scenarios) == 25
    assert len(list((output / "degraded").glob("*.wav"))) == 25
    assert len(list((output / "plots").glob("*.png"))) == 25
    assert len(list((output / "reference").glob("*.wav"))) == 4
    assert (output / "report" / "index.html").exists()
    assert (output / "DATASET_CARD.md").exists()
    assert (output / "configs" / "resolved_config.yaml").exists()
    for extension in ("csv", "jsonl", "parquet"):
        assert (output / "metadata" / f"manifest.{extension}").exists()
    assert len(pd.read_csv(output / "metadata" / "manifest.csv")) == 25
    assert len(pd.read_parquet(output / "metadata" / "manifest.parquet")) == 25
    assert len(read_jsonl(output / "metadata" / "manifest.jsonl")) == 25
    assert validate_dataset(output) == []
    for wav in (output / "degraded").glob("*.wav"):
        info = sf.info(wav)
        assert (info.samplerate, info.channels, info.subtype) == (16000, 1, "PCM_16")
    original_hashes = audio_hashes(output / "degraded")
    second = build_showcase(test_config, force=True)
    assert audio_hashes(second.output_root / "degraded") == original_hashes


def test_cli_build_and_validate(test_config: Path) -> None:
    runner = CliRunner()
    build = runner.invoke(app, ["build", "--config", str(test_config)])
    assert build.exit_code == 0, build.output
    assert "Scenarios: 25" in build.output
    output = test_config.parent / "output" / "fixture_showcase"
    validate = runner.invoke(app, ["validate", "--output", str(output)])
    assert validate.exit_code == 0, validate.output
