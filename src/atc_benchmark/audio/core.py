"""Canonical audio I/O, metrics, and normalization."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf
from scipy import signal

from atc_benchmark.exceptions import AudioError
from atc_benchmark.models import AudioMetrics, NormalizationConfig

FloatArray = npt.NDArray[np.float64]
SAMPLE_RATE = 16_000


def dbfs(value: float, floor: float = -120.0) -> float:
    """Convert a non-negative full-scale ratio to dBFS."""
    return max(floor, 20.0 * math.log10(max(value, 10 ** (floor / 20))))


def load_wav(path: Path) -> tuple[FloatArray, int]:
    """Load a mono audio file as float64 without changing its samples."""
    try:
        audio, sample_rate = sf.read(path, dtype="float64", always_2d=False)
    except (RuntimeError, OSError) as exc:
        raise AudioError(f"Cannot decode audio file {path}: {exc}") from exc
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    return np.asarray(audio, dtype=np.float64), int(sample_rate)


def write_canonical_wav(path: Path, audio: FloatArray) -> None:
    """Write mono, 16 kHz, signed PCM16 WAV with saturating conversion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    sf.write(path, np.clip(safe, -1.0, 1.0), SAMPLE_RATE, subtype="PCM_16", format="WAV")


def measure_audio(audio: FloatArray, sample_rate: int = SAMPLE_RATE) -> AudioMetrics:
    """Measure deterministic level, clipping, DC, and spectral sanity metrics."""
    if len(audio) == 0:
        raise AudioError("Cannot measure empty audio")
    absolute = np.abs(audio)
    peak = float(np.max(absolute))
    rms = float(np.sqrt(np.mean(np.square(audio))))
    clipping = float(np.mean(absolute >= (32767 / 32768)) * 100)
    centered = audio - np.mean(audio)
    frequencies, psd = signal.welch(centered, fs=sample_rate, nperseg=min(2048, len(audio)))
    power_sum = float(np.sum(psd))
    if power_sum > 0:
        centroid = float(np.sum(frequencies * psd) / power_sum)
        bandwidth = float(np.sqrt(np.sum(((frequencies - centroid) ** 2) * psd) / power_sum))
    else:
        centroid = bandwidth = 0.0
    return AudioMetrics(
        duration_sec=len(audio) / sample_rate,
        peak_dbfs=dbfs(peak),
        rms_dbfs=dbfs(rms),
        approximate_loudness_dbfs=dbfs(rms),
        clipping_percentage=clipping,
        dc_offset=float(np.mean(audio)),
        spectral_centroid_hz=centroid,
        bandwidth_hz=bandwidth,
    )


def normalize_audio(audio: FloatArray, config: NormalizationConfig) -> FloatArray:
    """Normalize conservatively according to explicit policy."""
    result = np.asarray(audio, dtype=np.float64).copy()
    if config.mode == "target_rms":
        current = measure_audio(result).rms_dbfs
        result *= 10 ** ((config.target_rms_dbfs - current) / 20)
    if config.mode == "peak_limit" or config.prevent_clipping:
        limit = 10 ** (config.peak_dbfs / 20)
        peak = float(np.max(np.abs(result))) if len(result) else 0.0
        if peak > limit:
            result *= limit / peak
    return result


def decode_to_canonical(
    source: Path, destination: Path, config: NormalizationConfig
) -> AudioMetrics:
    """Decode any FFmpeg-supported source once, normalize, and write canonical WAV."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map_metadata",
        "0",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise AudioError("FFmpeg is required but was not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise AudioError(f"FFmpeg could not decode {source}: {exc.stderr.strip()}") from exc
    audio, rate = load_wav(destination)
    normalized = normalize_audio(audio, config)
    write_canonical_wav(destination, normalized)
    return measure_audio(normalized, rate)
