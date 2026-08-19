from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from atc_benchmark.audio.core import SAMPLE_RATE
from atc_benchmark.degradation.effects import EffectContext, create_effect
from atc_benchmark.models import (
    AdjacentSpeechParams,
    AGCParams,
    BandpassParams,
    DropoutParams,
    FadingParams,
    HardClipParams,
    HumParams,
    ImpulseNoiseParams,
    NoiseParams,
    PacketLossParams,
    SoftClipParams,
    ToneParams,
    TruncateParams,
)


def apply(params: object, audio: np.ndarray, seed: int = 7, reference_dir: Path = Path(".")):
    return create_effect(params).apply(
        audio, np.random.default_rng(seed), EffectContext(reference_dir=reference_dir)
    )  # type: ignore[arg-type]


def amplitude_at(audio: np.ndarray, frequency: float) -> float:
    frequencies = np.fft.rfftfreq(len(audio), 1 / SAMPLE_RATE)
    spectrum = np.abs(np.fft.rfft(audio))
    return float(spectrum[np.argmin(np.abs(frequencies - frequency))])


def test_bandpass_preserves_passband_and_rejects_stopbands() -> None:
    time = np.arange(SAMPLE_RATE * 3) / SAMPLE_RATE
    source = sum(np.sin(2 * np.pi * f * time) for f in (100, 1000, 6000)) / 3
    output, _ = apply(BandpassParams(type="bandpass", low_hz=300, high_hz=3400, order=8), source)
    assert amplitude_at(output, 1000) > amplitude_at(output, 100) * 100
    assert amplitude_at(output, 1000) > amplitude_at(output, 6000) * 100
    assert amplitude_at(output, 1000) > amplitude_at(source, 1000) * 0.8


@pytest.mark.parametrize("noise_type", ["white", "pink", "static", "vhf"])
def test_noise_achieves_requested_snr(noise_type: str, tone_audio: np.ndarray) -> None:
    params = NoiseParams(type="noise", noise_type=noise_type, snr_db=10)
    output, result = apply(params, tone_audio)
    noise = output - tone_audio
    achieved = 20 * np.log10(np.sqrt(np.mean(tone_audio**2)) / np.sqrt(np.mean(noise**2)))
    assert achieved == pytest.approx(10, abs=0.01)
    assert result.derived["achieved_snr_db"] == pytest.approx(10, abs=0.01)


def test_noise_seed_controls_randomness(tone_audio: np.ndarray) -> None:
    params = NoiseParams(type="noise", noise_type="white", snr_db=10)
    first, _ = apply(params, tone_audio, 9)
    repeat, _ = apply(params, tone_audio, 9)
    different, _ = apply(params, tone_audio, 10)
    assert np.array_equal(first, repeat)
    assert not np.array_equal(first, different)


@pytest.mark.parametrize("waveform", ["sine", "triangle", "random"])
def test_fading_depth_and_determinism(waveform: str, tone_audio: np.ndarray) -> None:
    params = FadingParams(
        type="fading", depth_db=12, rate_hz=1.1, waveform=waveform, randomness=0.1
    )
    first, result = apply(params, tone_audio, 4)
    repeat, _ = apply(params, tone_audio, 4)
    assert np.array_equal(first, repeat)
    assert float(result.derived["minimum_gain_db"]) <= -9
    assert not np.array_equal(first, tone_audio)


def test_dropout_metadata_matches_samples(tone_audio: np.ndarray) -> None:
    params = DropoutParams(
        type="dropout", count=4, min_duration_ms=80, max_duration_ms=80, attenuation_db=120
    )
    output, result = apply(params, tone_audio)
    intervals = result.derived["intervals"]
    assert len(intervals) == 4
    for interval in intervals:
        start, end = interval["start_sample"], interval["end_sample"]
        assert end - start == 1280
        assert np.max(np.abs(output[start:end])) < np.max(np.abs(tone_audio[start:end])) * 1e-5


@pytest.mark.parametrize("edge", ["beginning", "ending"])
def test_truncate_removes_requested_duration_from_activity(
    edge: str, tone_audio: np.ndarray
) -> None:
    params = TruncateParams(type="truncate", edge=edge, duration_ms=200, active_threshold_dbfs=-60)
    output, result = apply(params, tone_audio)
    assert len(tone_audio) - len(output) == 3200
    assert result.derived["removed_end_sample"] > result.derived["removed_start_sample"]


def test_hard_clipping_increases_clipped_samples(tone_audio: np.ndarray) -> None:
    output, result = apply(
        HardClipParams(type="hard_clip", threshold=0.15, pre_gain_db=8), tone_audio
    )
    clipped = np.mean(np.abs(output) == 0.15) * 100
    assert clipped > 10
    assert result.derived["clipped_sample_percentage"] == pytest.approx(clipped)


def test_soft_clip_is_smooth_and_bounded(tone_audio: np.ndarray) -> None:
    output, _ = apply(SoftClipParams(type="soft_clip", drive=3), tone_audio)
    assert np.all(np.isfinite(output))
    assert np.max(np.abs(output)) <= 1
    assert not np.array_equal(output, tone_audio)


def test_agc_is_envelope_driven(tone_audio: np.ndarray) -> None:
    params = AGCParams(
        type="agc_pumping",
        attack_ms=5,
        release_ms=200,
        modulation_depth_db=10,
        min_gain_db=-12,
        max_gain_db=12,
        target_dbfs=-18,
    )
    output, result = apply(params, tone_audio)
    assert not np.array_equal(output, tone_audio)
    assert result.derived["maximum_applied_gain_db"] > result.derived["minimum_applied_gain_db"]


def test_impulses_have_exact_recorded_timestamps(tone_audio: np.ndarray) -> None:
    params = ImpulseNoiseParams(
        type="impulse_noise",
        events_per_second=2,
        amplitude=0.4,
        min_duration_ms=1,
        max_duration_ms=3,
        polarity="random",
    )
    output, result = apply(params, tone_audio, 12)
    assert len(result.derived["events"]) == 4
    for event in result.derived["events"]:
        assert event["timestamp_sec"] == event["sample"] / SAMPLE_RATE
        assert output[event["sample"]] != tone_audio[event["sample"]]


def test_hum_has_configured_spectral_peaks(tone_audio: np.ndarray) -> None:
    output, _ = apply(
        HumParams(type="hum", fundamental_hz=60, harmonics=[1, 2, 3], level_db_below_rms=3),
        tone_audio,
    )
    added = output - tone_audio
    noise_floor = np.median(np.abs(np.fft.rfft(added)))
    for frequency in (60, 120, 180):
        assert amplitude_at(added, frequency) > noise_floor * 1000


def test_interference_tone_and_drift(tone_audio: np.ndarray) -> None:
    output, _ = apply(
        ToneParams(type="tone", frequencies_hz=[1250], level_db_below_rms=6, drift_hz=10),
        tone_audio,
    )
    added = output - tone_audio
    frequencies, psd = signal.welch(added, fs=SAMPLE_RATE, nperseg=4096)
    peak = frequencies[np.argmax(psd)]
    assert 1240 <= peak <= 1270


def test_adjacent_speech_uses_secondary_atc_reference(
    tone_audio: np.ndarray, reference_pool: Path
) -> None:
    params = AdjacentSpeechParams(
        type="adjacent_speech",
        secondary_reference="ref_004.wav",
        relative_level_db=-10,
        start_offset_sec=0.1,
        low_hz=300,
        high_hz=3000,
    )
    output, result = apply(params, tone_audio, reference_dir=reference_pool)
    assert np.array_equal(output[:1600], tone_audio[:1600])
    assert not np.array_equal(output[1600:], tone_audio[1600:])
    assert result.derived["secondary_file"].endswith("ref_004.wav")


def test_packet_loss_records_and_replaces_frames(tone_audio: np.ndarray) -> None:
    params = PacketLossParams(
        type="packet_loss",
        frame_ms=20,
        loss_probability=0.4,
        burst_probability=0.8,
        concealment="zeros",
    )
    output, result = apply(params, tone_audio, 3)
    lost = result.derived["lost_frame_indices"]
    assert lost
    frame = result.derived["frame_samples"]
    for index in lost:
        assert np.all(output[index * frame : min(len(output), (index + 1) * frame)] == 0)
