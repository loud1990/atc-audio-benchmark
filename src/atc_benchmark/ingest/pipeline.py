"""Source discovery and one-time canonical reference creation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from atc_benchmark.audio.core import decode_to_canonical
from atc_benchmark.models import NormalizationConfig

from .acquisition import LocalDirectoryAdapter


def probe_source(path: Path) -> dict[str, Any]:
    """Read useful source metadata using FFprobe."""
    command = ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return dict(json.loads(completed.stdout))


def ingest_directory(
    raw_dir: Path,
    reference_dir: Path,
    normalization: NormalizationConfig,
    *,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Decode recognized raw inputs to sequential immutable canonical references."""
    sources = LocalDirectoryAdapter().acquire(raw_dir)
    reference_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, source in enumerate(sources, start=1):
        destination = reference_dir / f"ref_{index:03d}.wav"
        if destination.exists() and not force:
            from atc_benchmark.audio.core import load_wav, measure_audio

            audio, rate = load_wav(destination)
            metrics = measure_audio(audio, rate)
        else:
            metrics = decode_to_canonical(source, destination, normalization)
        records.append(
            {
                "reference_id": destination.stem,
                "reference_file": str(destination),
                "source_parent_file": str(source),
                "source_start_sec": 0.0,
                "source_end_sec": metrics.duration_sec,
                "metrics": metrics.model_dump(),
                "original_metadata": probe_source(source),
            }
        )
    return records
