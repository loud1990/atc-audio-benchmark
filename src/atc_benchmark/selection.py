"""Reference clip selection and eligibility ranking."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from atc_benchmark.audio.core import load_wav, measure_audio
from atc_benchmark.exceptions import ConfigurationError
from atc_benchmark.models import SelectionConfig


def select_reference(reference_dir: Path, config: SelectionConfig, index: int = 0) -> Path:
    """Choose a suitable canonical reference deterministically."""
    candidates: list[tuple[float, Path]] = []
    for path in sorted(reference_dir.glob("*.wav")):
        audio, rate = load_wav(path)
        metrics = measure_audio(audio, rate)
        if not (config.minimum_duration_sec <= metrics.duration_sec <= config.maximum_duration_sec):
            continue
        if metrics.clipping_percentage > config.maximum_clipping_percentage:
            continue
        if metrics.rms_dbfs < config.minimum_rms_dbfs:
            continue
        frame = max(1, round(rate * 0.02))
        frame_rms = [
            float(np.sqrt(np.mean(audio[start : start + frame] ** 2)))
            for start in range(0, len(audio), frame)
        ]
        activity_ratio = float(
            np.mean(np.asarray(frame_rms) >= 10 ** (config.activity_threshold_dbfs / 20))
        )
        if activity_ratio < config.minimum_speech_activity_ratio:
            continue
        # Prefer strong, unclipped speech-band energy while preserving deterministic ordering.
        score = metrics.rms_dbfs + activity_ratio * 3 - metrics.clipping_percentage * 10
        candidates.append((score, path))
    if not candidates:
        raise ConfigurationError(f"No eligible references found in {reference_dir}")
    candidates.sort(key=lambda item: (-item[0], item[1].name))
    return candidates[index % len(candidates)][1]
