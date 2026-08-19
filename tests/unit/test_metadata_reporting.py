from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from atc_benchmark.config import load_config
from atc_benchmark.metadata import read_jsonl, write_manifests
from atc_benchmark.models import ManifestRecord
from atc_benchmark.reporting import generate_dataset_card, generate_report


def record() -> ManifestRecord:
    return ManifestRecord(
        sample_id="001",
        scenario_name="test_noise",
        scenario_description="Controlled test noise.",
        severity="moderate",
        reference_file="reference/ref_001.wav",
        degraded_file="degraded/001_test_noise.wav",
        source_parent_file="source.wav",
        source_start_sec=0,
        source_end_sec=2,
        sample_rate=16000,
        channels=1,
        duration_sec=2,
        seed=123,
        effect_chain=["noise"],
        effect_parameters=[
            {
                "type": "noise",
                "parameters": {"snr_db": 10},
                "derived": {"achieved_snr_db": 10.0, "seed": 123},
            }
        ],
        reference_peak_dbfs=-5,
        degraded_peak_dbfs=-3,
        reference_rms_dbfs=-20,
        degraded_rms_dbfs=-18,
        clipping_percentage=0,
        rms_delta_db=2,
        peak_delta_db=2,
        duration_delta_sec=0,
        spectral_centroid_hz=1200,
        bandwidth_hz=800,
        achieved_snr_db=10,
        creation_timestamp="2026-01-01T00:00:00+00:00",
        pipeline_version="1.0.0",
        git_commit="unknown",
        provenance={"license": "fixture"},
    )


def test_manifests_round_trip_nested_effect_metadata(tmp_path: Path) -> None:
    expected = record()
    write_manifests([expected], tmp_path)
    assert read_jsonl(tmp_path / "manifest.jsonl") == [expected]
    csv_row = pd.read_csv(tmp_path / "manifest.csv").iloc[0]
    assert json.loads(csv_row.effect_parameters)[0]["derived"]["seed"] == 123
    parquet_row = pd.read_parquet(tmp_path / "manifest.parquet").iloc[0]
    assert parquet_row.effect_parameters[0]["derived"]["achieved_snr_db"] == 10


def test_static_report_and_dataset_card_include_scientific_distinction(tmp_path: Path) -> None:
    item = record()
    report = generate_report(tmp_path, [item], "Fixture")
    html = report.read_text()
    assert "Reference" in html and "Degraded" in html
    assert "reference/ref_001.wav" in html
    assert "degraded/001_test_noise.wav" in html
    assert html.count('type="checkbox"') == 6
    config = load_config(Path("configs/showcase_v1.yaml"))
    card = generate_dataset_card(tmp_path, config, [item]).read_text()
    assert "Reference audio" in card
    assert "not pristine" in card
    assert "not a perfect simulation of real RF propagation" in card
