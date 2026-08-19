"""Deterministic energy-based transmission segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from atc_benchmark.audio.core import FloatArray
from atc_benchmark.models import SegmentationConfig


@dataclass(frozen=True)
class Segment:
    """Sample-exact region in a parent recording."""

    start_sample: int
    end_sample: int
    sample_rate: int

    @property
    def start_sec(self) -> float:
        return self.start_sample / self.sample_rate

    @property
    def end_sec(self) -> float:
        return self.end_sample / self.sample_rate


def energy_segments(
    audio: FloatArray, sample_rate: int, config: SegmentationConfig
) -> list[Segment]:
    """Find active regions, merge nearby frames, add context, and cap length."""
    frame = max(1, round(config.frame_ms * sample_rate / 1000))
    count = int(np.ceil(len(audio) / frame))
    rms = np.zeros(count)
    for index in range(count):
        chunk = audio[index * frame : min(len(audio), (index + 1) * frame)]
        rms[index] = np.sqrt(np.mean(chunk**2)) if len(chunk) else 0
    active = rms >= 10 ** (config.threshold_dbfs / 20)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(active):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(active) - 1):
            end = index + 1 if value else index
            runs.append((start * frame, min(len(audio), end * frame)))
            start = None
    merge_gap = round(config.merge_gap_sec * sample_rate)
    merged: list[list[int]] = []
    for run_start, run_end in runs:
        if merged and run_start - merged[-1][1] <= merge_gap:
            merged[-1][1] = run_end
        else:
            merged.append([run_start, run_end])
    pre = round(config.pre_roll_sec * sample_rate)
    post = round(config.post_roll_sec * sample_rate)
    minimum = round(config.min_speech_sec * sample_rate)
    maximum = round(config.max_clip_sec * sample_rate)
    segments: list[Segment] = []
    for run_start, run_end in merged:
        if run_end - run_start < minimum:
            continue
        expanded_start = max(0, run_start - pre)
        expanded_end = min(len(audio), run_end + post)
        cursor = expanded_start
        while cursor < expanded_end:
            end = min(expanded_end, cursor + maximum)
            segments.append(Segment(cursor, end, sample_rate))
            cursor = end
    return segments
