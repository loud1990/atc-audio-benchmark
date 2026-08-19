"""Canonical WAV, metadata, and meaningful-difference validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from atc_benchmark.audio.core import SAMPLE_RATE, FloatArray
from atc_benchmark.models import ManifestRecord


def validate_wav(path: Path, minimum_sec: float, maximum_sec: float) -> list[str]:
    """Return validation errors for a canonical output WAV."""
    errors: list[str] = []
    try:
        info = sf.info(path)
        audio, rate = sf.read(path, dtype="float64", always_2d=False)
    except (OSError, RuntimeError) as exc:
        return [f"cannot decode {path}: {exc}"]
    if rate != SAMPLE_RATE:
        errors.append(f"sample rate is {rate}, expected {SAMPLE_RATE}")
    if info.channels != 1:
        errors.append(f"channel count is {info.channels}, expected 1")
    if info.subtype != "PCM_16":
        errors.append(f"sample subtype is {info.subtype}, expected PCM_16")
    duration = len(audio) / rate
    if duration <= minimum_sec:
        errors.append(f"duration {duration:.3f}s is not above {minimum_sec:.3f}s")
    if duration >= maximum_sec:
        errors.append(f"duration {duration:.3f}s is not below {maximum_sec:.3f}s")
    if not np.all(np.isfinite(audio)):
        errors.append("contains NaN or infinite samples")
    if not np.any(np.asarray(audio) != 0):
        errors.append("contains only silence")
    rms = float(np.sqrt(np.mean(np.asarray(audio) ** 2)))
    if rms < 10 ** (-90 / 20):
        errors.append("signal is unexpectedly near-silent (below -90 dBFS RMS)")
    if float(np.max(np.abs(audio))) > 1:
        errors.append("peak exceeds valid PCM range")
    clipping_ratio = float(np.mean(np.abs(audio) >= (32767 / 32768)))
    if clipping_ratio > 0.25:
        errors.append(f"excessive full-scale clipping ({clipping_ratio * 100:.2f}%)")
    dc_offset = float(np.mean(audio))
    if abs(dc_offset) > 0.1:
        errors.append(f"excessive DC offset ({dc_offset:.4f})")
    return errors


def validate_difference(reference: FloatArray, degraded: FloatArray) -> list[str]:
    """Detect an ineffective effect chain without requiring equal lengths."""
    overlap = min(len(reference), len(degraded))
    if overlap == 0:
        return ["reference/degraded overlap is empty"]
    sample_difference = float(np.sqrt(np.mean((reference[:overlap] - degraded[:overlap]) ** 2)))
    ref_rms = float(np.sqrt(np.mean(reference[:overlap] ** 2)))
    if sample_difference <= max(1e-5, ref_rms * 1e-4) and len(reference) == len(degraded):
        return [
            f"effect chain produced no measurable change (difference RMS={sample_difference:.3g})"
        ]
    return []


def validate_record(record: ManifestRecord) -> list[str]:
    """Check core manifest invariants and obvious quality failures."""
    errors: list[str] = []
    if record.sample_rate != SAMPLE_RATE or record.channels != 1:
        errors.append("manifest canonical format fields are invalid")
    if not record.effect_chain or len(record.effect_chain) != len(record.effect_parameters):
        errors.append("effect chain metadata is missing or inconsistent")
    if record.clipping_percentage >= 100:
        errors.append("output is 100% clipped")
    if record.degraded_rms_dbfs - record.reference_rms_dbfs > 30:
        errors.append("unexpected gain increase above 30 dB")
    return errors
