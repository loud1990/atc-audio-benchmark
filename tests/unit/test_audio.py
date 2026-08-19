from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from atc_benchmark.audio.core import (
    SAMPLE_RATE,
    load_wav,
    measure_audio,
    normalize_audio,
    write_canonical_wav,
)
from atc_benchmark.models import NormalizationConfig
from atc_benchmark.validation import validate_difference, validate_wav


def test_canonical_round_trip(tmp_path: Path, tone_audio: np.ndarray) -> None:
    path = tmp_path / "canonical.wav"
    write_canonical_wav(path, tone_audio)
    info = sf.info(path)
    restored, rate = load_wav(path)
    assert rate == SAMPLE_RATE
    assert info.channels == 1
    assert info.subtype == "PCM_16"
    assert np.max(np.abs(restored - tone_audio)) <= 1 / 32768
    assert validate_wav(path, 0.1, 10) == []


def test_metrics_are_mathematically_consistent(tone_audio: np.ndarray) -> None:
    metrics = measure_audio(tone_audio)
    expected_rms = 20 * np.log10(np.sqrt(np.mean(tone_audio**2)))
    assert metrics.rms_dbfs == pytest.approx(expected_rms, abs=1e-9)
    assert metrics.duration_sec == 2
    assert metrics.dc_offset == pytest.approx(float(np.mean(tone_audio)), abs=1e-12)
    assert metrics.bandwidth_hz > 0


def test_peak_limiter_does_not_raise_quiet_audio(tone_audio: np.ndarray) -> None:
    config = NormalizationConfig(mode="peak_limit", peak_dbfs=-1)
    assert np.array_equal(normalize_audio(tone_audio, config), tone_audio)
    loud = tone_audio * 10
    normalized = normalize_audio(loud, config)
    assert np.max(np.abs(normalized)) == pytest.approx(10 ** (-1 / 20))


def test_difference_validation(tone_audio: np.ndarray) -> None:
    assert validate_difference(tone_audio, tone_audio)
    assert not validate_difference(tone_audio, tone_audio * 0.9)
