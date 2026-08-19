"""Reusable deterministic audio degradation transforms."""

from __future__ import annotations

import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from scipy import signal

from atc_benchmark.audio.core import (
    SAMPLE_RATE,
    FloatArray,
    load_wav,
    write_canonical_wav,
)
from atc_benchmark.exceptions import AudioError, ConfigurationError
from atc_benchmark.models import (
    AdjacentSpeechParams,
    AGCParams,
    BandpassParams,
    CodecParams,
    DropoutParams,
    EffectParams,
    EffectResult,
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


@dataclass(frozen=True)
class EffectContext:
    """External resources available to a transform."""

    sample_rate: int = SAMPLE_RATE
    reference_dir: Path = Path("data/reference")


class Effect(Protocol):
    """Protocol implemented by every transform."""

    def apply(
        self, audio: FloatArray, rng: np.random.Generator, context: EffectContext
    ) -> tuple[FloatArray, EffectResult]: ...


def _sos_bandpass(audio: FloatArray, low: float, high: float, order: int, rate: int) -> FloatArray:
    sos = signal.butter(order, [low, high], btype="bandpass", fs=rate, output="sos")
    # SOS filtering is stable for radio-style steep filters and short utterances.
    return cast(FloatArray, signal.sosfilt(sos, audio))


def _scale_to_snr(
    reference: FloatArray, noise: FloatArray, snr_db: float
) -> tuple[FloatArray, float]:
    signal_rms = float(np.sqrt(np.mean(np.square(reference))))
    noise_rms = float(np.sqrt(np.mean(np.square(noise))))
    if signal_rms <= 1e-12:
        raise AudioError("Cannot target SNR for a silent reference")
    if noise_rms <= 1e-12:
        raise AudioError("Generated noise has zero energy")
    scaled = noise * (signal_rms / (10 ** (snr_db / 20) * noise_rms))
    achieved = 20 * math.log10(signal_rms / float(np.sqrt(np.mean(np.square(scaled)))))
    return scaled, achieved


class BandpassEffect:
    def __init__(self, params: BandpassParams) -> None:
        self.params = params

    def apply(
        self, audio: FloatArray, rng: np.random.Generator, context: EffectContext
    ) -> tuple[FloatArray, EffectResult]:
        del rng
        output = _sos_bandpass(
            audio, self.params.low_hz, self.params.high_hz, self.params.order, context.sample_rate
        )
        return output, EffectResult(type=self.params.type, parameters=self.params.model_dump())


class NoiseEffect:
    def __init__(self, params: NoiseParams) -> None:
        self.params = params

    def apply(
        self, audio: FloatArray, rng: np.random.Generator, context: EffectContext
    ) -> tuple[FloatArray, EffectResult]:
        noise = rng.normal(0, 1, len(audio))
        if self.params.noise_type == "pink":
            spectrum = np.fft.rfft(noise)
            frequencies = np.fft.rfftfreq(len(noise), 1 / context.sample_rate)
            spectrum *= 1 / np.sqrt(np.maximum(frequencies, 1.0))
            noise = np.fft.irfft(spectrum, n=len(noise))
            noise -= np.mean(noise)
        elif self.params.noise_type == "vhf":
            noise = _sos_bandpass(
                noise, self.params.low_hz, self.params.high_hz, 6, context.sample_rate
            )
        elif self.params.noise_type == "static":
            # A slight high-frequency emphasis distinguishes receiver static from flat white noise.
            noise = cast(FloatArray, signal.lfilter([1.0, -0.7], [1.0], noise))
        scaled, achieved = _scale_to_snr(audio, np.asarray(noise), self.params.snr_db)
        output = audio + scaled
        return output, EffectResult(
            type=self.params.type,
            parameters=self.params.model_dump(),
            derived={"achieved_snr_db": achieved, "noise_rms": float(np.sqrt(np.mean(scaled**2)))},
        )


class FadingEffect:
    def __init__(self, params: FadingParams) -> None:
        self.params = params

    def apply(
        self, audio: FloatArray, rng: np.random.Generator, context: EffectContext
    ) -> tuple[FloatArray, EffectResult]:
        time = np.arange(len(audio)) / context.sample_rate
        phase = 2 * np.pi * self.params.rate_hz * time
        if self.params.waveform == "triangle":
            modulation = (signal.sawtooth(phase, width=0.5) + 1) / 2
        elif self.params.waveform == "random":
            knots = max(2, int(len(audio) / context.sample_rate * self.params.rate_hz) + 2)
            modulation = np.interp(
                np.arange(len(audio)), np.linspace(0, len(audio) - 1, knots), rng.random(knots)
            )
        else:
            modulation = (1 + np.sin(phase)) / 2
        if self.params.randomness:
            knot_count = max(2, round(len(audio) / context.sample_rate * 10) + 1)
            random_curve = np.interp(
                np.arange(len(audio)),
                np.linspace(0, max(0, len(audio) - 1), knot_count),
                rng.random(knot_count),
            )
            modulation = (
                1 - self.params.randomness
            ) * modulation + self.params.randomness * random_curve
        gain_db = -self.params.depth_db * modulation
        gain = 10 ** (gain_db / 20)
        return audio * gain, EffectResult(
            type=self.params.type,
            parameters=self.params.model_dump(),
            derived={
                "minimum_gain_db": float(np.min(gain_db)),
                "maximum_gain_db": float(np.max(gain_db)),
            },
        )


class DropoutEffect:
    def __init__(self, params: DropoutParams) -> None:
        self.params = params

    def apply(
        self, audio: FloatArray, rng: np.random.Generator, context: EffectContext
    ) -> tuple[FloatArray, EffectResult]:
        output = audio.copy()
        intervals: list[dict[str, float | int]] = []
        factor = 10 ** (-self.params.attenuation_db / 20)
        for _ in range(self.params.count):
            samples = round(
                rng.uniform(self.params.min_duration_ms, self.params.max_duration_ms)
                * context.sample_rate
                / 1000
            )
            samples = min(max(1, samples), len(audio))
            start = int(rng.integers(0, max(1, len(audio) - samples + 1)))
            end = start + samples
            output[start:end] *= factor
            intervals.append(
                {
                    "start_sample": start,
                    "end_sample": end,
                    "start_sec": start / context.sample_rate,
                    "end_sec": end / context.sample_rate,
                }
            )
        return output, EffectResult(
            type=self.params.type,
            parameters=self.params.model_dump(),
            derived={"intervals": intervals},
        )


class TruncateEffect:
    def __init__(self, params: TruncateParams) -> None:
        self.params = params

    def apply(
        self, audio: FloatArray, rng: np.random.Generator, context: EffectContext
    ) -> tuple[FloatArray, EffectResult]:
        del rng
        frame = max(1, int(context.sample_rate * 0.01))
        energies = np.array(
            [
                np.sqrt(np.mean(chunk**2))
                for chunk in np.array_split(
                    audio[: len(audio) // frame * frame], max(1, len(audio) // frame)
                )
            ]
        )
        active = np.flatnonzero(energies >= 10 ** (self.params.active_threshold_dbfs / 20))
        remove = min(
            round(self.params.duration_ms * context.sample_rate / 1000), max(1, len(audio) - 1)
        )
        if len(active):
            first = int(active[0] * frame)
            last = min(len(audio), int((active[-1] + 1) * frame))
        else:
            first, last = 0, len(audio)
        if self.params.edge == "beginning":
            boundary = min(len(audio) - 1, first + remove)
            output = audio[boundary:]
            removed = (0, boundary)
        else:
            boundary = max(1, last - remove)
            output = audio[:boundary]
            removed = (boundary, len(audio))
        return output.copy(), EffectResult(
            type=self.params.type,
            parameters=self.params.model_dump(),
            derived={
                "removed_start_sample": removed[0],
                "removed_end_sample": removed[1],
                "active_start_sample": first,
                "active_end_sample": last,
            },
        )


class HardClipEffect:
    def __init__(self, params: HardClipParams) -> None:
        self.params = params

    def apply(
        self, audio: FloatArray, rng: np.random.Generator, context: EffectContext
    ) -> tuple[FloatArray, EffectResult]:
        del rng, context
        driven = audio * 10 ** (self.params.pre_gain_db / 20)
        mask = np.abs(driven) >= self.params.threshold
        output = np.clip(driven, -self.params.threshold, self.params.threshold)
        return output, EffectResult(
            type=self.params.type,
            parameters=self.params.model_dump(),
            derived={"clipped_sample_percentage": float(np.mean(mask) * 100)},
        )


class SoftClipEffect:
    def __init__(self, params: SoftClipParams) -> None:
        self.params = params

    def apply(
        self, audio: FloatArray, rng: np.random.Generator, context: EffectContext
    ) -> tuple[FloatArray, EffectResult]:
        del rng, context
        output = np.tanh(audio * self.params.drive) / np.tanh(self.params.drive)
        return output, EffectResult(type=self.params.type, parameters=self.params.model_dump())


class AGCEffect:
    def __init__(self, params: AGCParams) -> None:
        self.params = params

    def apply(
        self, audio: FloatArray, rng: np.random.Generator, context: EffectContext
    ) -> tuple[FloatArray, EffectResult]:
        del rng
        attack = math.exp(-1 / (context.sample_rate * self.params.attack_ms / 1000))
        release = math.exp(-1 / (context.sample_rate * self.params.release_ms / 1000))
        envelope = np.empty(len(audio))
        current = 1e-6
        for index, sample in enumerate(np.abs(audio)):
            coefficient = attack if sample > current else release
            current = coefficient * current + (1 - coefficient) * sample
            envelope[index] = current
        desired_db = self.params.target_dbfs - 20 * np.log10(np.maximum(envelope, 1e-6))
        # Pumping is linked to the detected envelope, not an unrelated periodic modulation.
        desired_db *= self.params.modulation_depth_db / max(self.params.modulation_depth_db, 12)
        gain_db = np.clip(desired_db, self.params.min_gain_db, self.params.max_gain_db)
        output = audio * 10 ** (gain_db / 20)
        return output, EffectResult(
            type=self.params.type,
            parameters=self.params.model_dump(),
            derived={
                "minimum_applied_gain_db": float(np.min(gain_db)),
                "maximum_applied_gain_db": float(np.max(gain_db)),
            },
        )


class ImpulseNoiseEffect:
    def __init__(self, params: ImpulseNoiseParams) -> None:
        self.params = params

    def apply(
        self, audio: FloatArray, rng: np.random.Generator, context: EffectContext
    ) -> tuple[FloatArray, EffectResult]:
        output = audio.copy()
        count = max(1, round(len(audio) / context.sample_rate * self.params.events_per_second))
        events: list[dict[str, float | int]] = []
        for _ in range(count):
            duration = max(
                1,
                round(
                    rng.uniform(self.params.min_duration_ms, self.params.max_duration_ms)
                    * context.sample_rate
                    / 1000
                ),
            )
            start = int(rng.integers(0, max(1, len(audio) - duration + 1)))
            polarity = (
                1
                if self.params.polarity == "positive"
                else -1
                if self.params.polarity == "negative"
                else int(rng.choice([-1, 1]))
            )
            window = signal.windows.exponential(
                duration, center=0, tau=max(1, duration / 4), sym=False
            )
            output[start : start + duration] += polarity * self.params.amplitude * window
            events.append(
                {
                    "sample": start,
                    "timestamp_sec": start / context.sample_rate,
                    "duration_samples": duration,
                    "polarity": polarity,
                }
            )
        return output, EffectResult(
            type=self.params.type, parameters=self.params.model_dump(), derived={"events": events}
        )


class HumEffect:
    def __init__(self, params: HumParams) -> None:
        self.params = params

    def apply(
        self, audio: FloatArray, rng: np.random.Generator, context: EffectContext
    ) -> tuple[FloatArray, EffectResult]:
        del rng
        time = np.arange(len(audio)) / context.sample_rate
        hum = np.zeros(len(audio), dtype=np.float64)
        for harmonic in self.params.harmonics:
            hum += np.sin(2 * np.pi * self.params.fundamental_hz * harmonic * time) / harmonic
        rms = float(np.sqrt(np.mean(audio**2)))
        hum_rms = float(np.sqrt(np.mean(hum**2)))
        hum *= rms * 10 ** (-self.params.level_db_below_rms / 20) / max(hum_rms, 1e-12)
        return audio + hum, EffectResult(
            type=self.params.type,
            parameters=self.params.model_dump(),
            derived={
                "frequencies_hz": [
                    self.params.fundamental_hz * value for value in self.params.harmonics
                ]
            },
        )


class ToneEffect:
    def __init__(self, params: ToneParams) -> None:
        self.params = params

    def apply(
        self, audio: FloatArray, rng: np.random.Generator, context: EffectContext
    ) -> tuple[FloatArray, EffectResult]:
        del rng
        time = np.arange(len(audio)) / context.sample_rate
        tones = np.zeros(len(audio))
        for frequency in self.params.frequencies_hz:
            phase = (
                2
                * np.pi
                * (
                    frequency * time
                    + (self.params.drift_hz / max(1, 2 * len(time) / context.sample_rate)) * time**2
                )
            )
            tones += np.sin(phase)
        source_rms = float(np.sqrt(np.mean(audio**2)))
        tones *= (
            source_rms
            * 10 ** (-self.params.level_db_below_rms / 20)
            / max(float(np.sqrt(np.mean(tones**2))), 1e-12)
        )
        return audio + tones, EffectResult(
            type=self.params.type, parameters=self.params.model_dump()
        )


class AdjacentSpeechEffect:
    def __init__(self, params: AdjacentSpeechParams) -> None:
        self.params = params

    def apply(
        self, audio: FloatArray, rng: np.random.Generator, context: EffectContext
    ) -> tuple[FloatArray, EffectResult]:
        del rng
        path = context.reference_dir / self.params.secondary_reference
        if path.suffix.lower() != ".wav":
            path = path.with_suffix(".wav")
        if not path.exists():
            raise ConfigurationError(f"Adjacent-channel secondary reference does not exist: {path}")
        secondary, rate = load_wav(path)
        if rate != context.sample_rate:
            raise AudioError(
                f"Secondary reference {path} is {rate} Hz; expected {context.sample_rate} Hz"
            )
        if self.params.low_hz is not None and self.params.high_hz is not None:
            secondary = _sos_bandpass(secondary, self.params.low_hz, self.params.high_hz, 4, rate)
        offset = round(self.params.start_offset_sec * rate)
        available = max(0, len(audio) - offset)
        tiled = np.resize(secondary, available)
        primary_rms = float(np.sqrt(np.mean(audio**2)))
        secondary_rms = float(np.sqrt(np.mean(tiled**2))) if len(tiled) else 0
        scaled = tiled * (
            primary_rms * 10 ** (self.params.relative_level_db / 20) / max(secondary_rms, 1e-12)
        )
        output = audio.copy()
        output[offset : offset + available] += scaled
        return output, EffectResult(
            type=self.params.type,
            parameters=self.params.model_dump(),
            derived={"secondary_file": str(path), "start_sample": offset},
        )


class CodecEffect:
    def __init__(self, params: CodecParams) -> None:
        self.params = params

    def apply(
        self, audio: FloatArray, rng: np.random.Generator, context: EffectContext
    ) -> tuple[FloatArray, EffectResult]:
        del rng
        extension = ".opus" if self.params.codec == "opus" else ".mp3"
        encoder = "libopus" if self.params.codec == "opus" else "libmp3lame"
        commands: list[list[str]] = []
        with tempfile.TemporaryDirectory(prefix="atc-codec-") as temp_string:
            temp = Path(temp_string)
            current = temp / "input.wav"
            write_canonical_wav(current, audio)
            for generation in range(self.params.generations):
                encoded = temp / f"generation-{generation}{extension}"
                decoded = temp / f"generation-{generation}.wav"
                encode = [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(current),
                    "-c:a",
                    encoder,
                    "-b:a",
                    f"{self.params.bitrate_kbps}k",
                    str(encoded),
                ]
                decode = [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(encoded),
                    "-ac",
                    "1",
                    "-ar",
                    str(context.sample_rate),
                    "-c:a",
                    "pcm_s16le",
                    str(decoded),
                ]
                try:
                    subprocess.run(encode, check=True, capture_output=True)
                    subprocess.run(decode, check=True, capture_output=True)
                except FileNotFoundError as exc:
                    raise AudioError("FFmpeg is required for codec degradation") from exc
                except subprocess.CalledProcessError as exc:
                    raise AudioError(
                        f"FFmpeg {self.params.codec} round-trip failed: {exc.stderr.decode(errors='replace')}"
                    ) from exc
                commands.extend((encode[5:], decode[5:]))
                current = decoded
            output, _ = load_wav(current)
        return output, EffectResult(
            type=self.params.type,
            parameters=self.params.model_dump(),
            derived={
                "codec": self.params.codec,
                "bitrate_kbps": self.params.bitrate_kbps,
                "generations": self.params.generations,
                "ffmpeg_arguments": commands,
                "resulting_format": "WAV mono PCM16 16000 Hz",
            },
        )


class PacketLossEffect:
    def __init__(self, params: PacketLossParams) -> None:
        self.params = params

    def apply(
        self, audio: FloatArray, rng: np.random.Generator, context: EffectContext
    ) -> tuple[FloatArray, EffectResult]:
        output = audio.copy()
        frame = max(1, round(self.params.frame_ms * context.sample_rate / 1000))
        frames = math.ceil(len(audio) / frame)
        lost: list[int] = []
        previous_lost = False
        previous_good = np.zeros(frame)
        for index in range(frames):
            start, end = index * frame, min((index + 1) * frame, len(audio))
            probability = max(
                self.params.loss_probability, self.params.burst_probability if previous_lost else 0
            )
            if rng.random() < probability:
                lost.append(index)
                if self.params.concealment == "zeros":
                    output[start:end] = 0
                else:
                    output[start:end] = previous_good[: end - start]
                previous_lost = True
            else:
                previous_good = np.pad(audio[start:end], (0, frame - (end - start)))
                previous_lost = False
        return output, EffectResult(
            type=self.params.type,
            parameters=self.params.model_dump(),
            derived={
                "frame_samples": frame,
                "lost_frame_indices": lost,
                "approximation": "deterministic time-domain frame erasure with configured concealment",
            },
        )


def create_effect(params: EffectParams) -> Effect:
    """Create the transform associated with a validated parameter model."""
    mapping: dict[type[Any], type[Any]] = {
        BandpassParams: BandpassEffect,
        NoiseParams: NoiseEffect,
        FadingParams: FadingEffect,
        DropoutParams: DropoutEffect,
        TruncateParams: TruncateEffect,
        HardClipParams: HardClipEffect,
        SoftClipParams: SoftClipEffect,
        AGCParams: AGCEffect,
        ImpulseNoiseParams: ImpulseNoiseEffect,
        HumParams: HumEffect,
        ToneParams: ToneEffect,
        AdjacentSpeechParams: AdjacentSpeechEffect,
        CodecParams: CodecEffect,
        PacketLossParams: PacketLossEffect,
    }
    effect_class = mapping.get(type(params))
    if effect_class is None:
        raise ConfigurationError(f"No transform is registered for {type(params).__name__}")
    return cast(Effect, effect_class(params))


def apply_effect_chain(
    audio: FloatArray,
    effects: list[EffectParams],
    seeds: list[int],
    context: EffectContext,
) -> tuple[FloatArray, list[EffectResult]]:
    """Apply a validated effect chain without mutating the reference array."""
    output = audio.copy()
    results: list[EffectResult] = []
    for params, seed in zip(effects, seeds, strict=True):
        explicit_seed = getattr(params, "seed", None)
        rng = np.random.default_rng(explicit_seed if explicit_seed is not None else seed)
        output, result = create_effect(params).apply(output, rng, context)
        result.derived["seed"] = explicit_seed if explicit_seed is not None else seed
        results.append(result)
    return output, results
