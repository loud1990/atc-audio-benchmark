from __future__ import annotations

import hashlib

import numpy as np
import pytest

from atc_benchmark.audio.core import write_canonical_wav
from atc_benchmark.degradation.effects import EffectContext, create_effect
from atc_benchmark.models import BandpassParams, DropoutParams, NoiseParams

pytestmark = pytest.mark.regression


def fingerprint(audio: np.ndarray) -> tuple[float, float, float]:
    return (
        float(np.sqrt(np.mean(audio**2))),
        float(np.max(np.abs(audio))),
        float(np.mean(np.abs(np.diff(audio)))),
    )


@pytest.mark.parametrize(
    ("params", "seed", "expected"),
    [
        (
            BandpassParams(type="bandpass", low_hz=300, high_hz=3400, order=6),
            1,
            (0.0400481282215567, 0.08504077112703154, 0.01178457415370987),
        ),
        (
            NoiseParams(type="noise", noise_type="vhf", snr_db=9),
            4821,
            (0.09976560393885203, 0.31184958181433087, 0.024141493048735363),
        ),
        (
            DropoutParams(type="dropout", count=3, min_duration_ms=50, max_duration_ms=150),
            918,
            (0.08508115914750627, 0.20742559509725064, 0.010585578743739457),
        ),
    ],
)
def test_numerical_fingerprints(
    params: object, seed: int, expected: tuple[float, float, float], tone_audio: np.ndarray
) -> None:
    output, _ = create_effect(params).apply(
        tone_audio, np.random.default_rng(seed), EffectContext()
    )  # type: ignore[arg-type]
    assert fingerprint(output) == pytest.approx(expected, rel=2e-7, abs=2e-9)


def test_pcm_hash_is_deterministic(tmp_path, tone_audio: np.ndarray) -> None:
    params = NoiseParams(type="noise", noise_type="white", snr_db=10)
    output, _ = create_effect(params).apply(tone_audio, np.random.default_rng(123), EffectContext())
    path = tmp_path / "result.wav"
    write_canonical_wav(path, output)
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == "13cc793062348048c0db3d2bce14588335b585f3269072edbdc3183437e63b00"
    )
