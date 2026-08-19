"""Generate deterministic pseudo-transmission fixtures; no third-party audio is used."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

RATE = 16_000


def pseudo_transmission(seconds: float, fundamental: float, seed: int) -> np.ndarray:
    """Synthesize modulated voiced/noisy bursts separated by radio-like pauses."""
    rng = np.random.default_rng(seed)
    time = np.arange(int(seconds * RATE)) / RATE
    excitation = sum(
        (1 / harmonic) * np.sin(2 * np.pi * fundamental * harmonic * time + harmonic * 0.31)
        for harmonic in range(1, 13)
    )
    formants = signal.sosfilt(
        signal.butter(4, [280, 3300], btype="bandpass", fs=RATE, output="sos"),
        excitation + 0.08 * rng.normal(size=len(time)),
    )
    syllables = np.maximum(0, np.sin(2 * np.pi * (2.7 + seed * 0.05) * time)) ** 0.6
    phrase = ((time > 0.15) & (time < seconds - 0.15)).astype(float)
    gaps = np.ones_like(time)
    gaps[(time > seconds * 0.43) & (time < seconds * 0.52)] = 0
    envelope = signal.sosfilt(
        signal.butter(2, 12, fs=RATE, output="sos"), syllables * phrase * gaps
    )
    output = formants * envelope
    output += 0.005 * rng.normal(size=len(output))
    output -= np.mean(output)
    output *= 0.32 / max(np.max(np.abs(output)), 1e-12)
    return output.astype(np.float64)


def generate(destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, fundamental in enumerate((105.0, 132.0, 167.0, 205.0), start=1):
        path = destination / f"ref_{index:03d}.wav"
        sf.write(
            path,
            pseudo_transmission(2.4 + index * 0.15, fundamental, index),
            RATE,
            subtype="PCM_16",
        )
        paths.append(path)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    for generated in generate(args.destination):
        print(generated)
