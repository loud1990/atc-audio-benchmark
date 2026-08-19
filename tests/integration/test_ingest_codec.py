from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from atc_benchmark.audio.core import SAMPLE_RATE, load_wav
from atc_benchmark.degradation.effects import EffectContext, create_effect
from atc_benchmark.ingest import ingest_directory
from atc_benchmark.models import CodecParams, NormalizationConfig
from atc_benchmark.validation import validate_wav

pytestmark = pytest.mark.integration


def has_encoder(name: str) -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    completed = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True
    )
    return name in completed.stdout


@pytest.mark.skipif(not has_encoder("libmp3lame"), reason="FFmpeg libmp3lame encoder unavailable")
def test_ingest_mp3_to_canonical(tmp_path: Path, canonical_wav: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    mp3 = raw / "source.mp3"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(canonical_wav),
            "-c:a",
            "libmp3lame",
            "-b:a",
            "64k",
            str(mp3),
        ],
        check=True,
    )
    records = ingest_directory(raw, tmp_path / "reference", NormalizationConfig())
    output = tmp_path / "reference" / "ref_001.wav"
    assert len(records) == 1
    assert validate_wav(output, 0.1, 10) == []
    _, rate = load_wav(output)
    assert rate == SAMPLE_RATE


@pytest.mark.parametrize(
    ("codec", "encoder", "bitrate", "generations"),
    [("opus", "libopus", 10, 1), ("mp3", "libmp3lame", 32, 3)],
)
def test_codec_round_trip(
    codec: str, encoder: str, bitrate: int, generations: int, tone_audio: np.ndarray
) -> None:
    if not has_encoder(encoder):
        pytest.skip(f"FFmpeg {encoder} encoder unavailable")
    params = CodecParams(type="codec", codec=codec, bitrate_kbps=bitrate, generations=generations)
    output, result = create_effect(params).apply(
        tone_audio, np.random.default_rng(0), EffectContext()
    )
    assert len(output) >= len(tone_audio) * 0.95
    assert np.all(np.isfinite(output))
    assert not np.array_equal(output[: len(tone_audio)], tone_audio[: len(output)])
    assert result.derived["generations"] == generations
    assert result.derived["resulting_format"] == "WAV mono PCM16 16000 Hz"


def test_decode_supported_flac(tmp_path: Path, tone_audio: np.ndarray) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    sf.write(raw / "source.flac", tone_audio, SAMPLE_RATE)
    records = ingest_directory(raw, tmp_path / "reference", NormalizationConfig())
    assert len(records) == 1
    assert (tmp_path / "reference" / "ref_001.wav").exists()
