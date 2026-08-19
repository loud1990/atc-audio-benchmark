import numpy as np

from atc_benchmark.models import SegmentationConfig
from atc_benchmark.segmentation import energy_segments


def test_energy_segments_merge_and_add_roll() -> None:
    rate = 1000
    audio = np.zeros(3000)
    audio[500:900] = 0.2
    audio[1000:1500] = 0.2
    config = SegmentationConfig(
        enabled=True,
        threshold_dbfs=-30,
        frame_ms=20,
        min_speech_sec=0.2,
        max_clip_sec=5,
        pre_roll_sec=0.1,
        post_roll_sec=0.2,
        merge_gap_sec=0.15,
    )
    segments = energy_segments(audio, rate, config)
    assert len(segments) == 1
    assert segments[0].start_sample == 400
    assert segments[0].end_sample == 1700


def test_energy_segments_cap_long_regions() -> None:
    audio = np.full(5000, 0.2)
    config = SegmentationConfig(
        enabled=True,
        threshold_dbfs=-30,
        frame_ms=20,
        min_speech_sec=0.1,
        max_clip_sec=2,
        pre_roll_sec=0,
        post_roll_sec=0,
        merge_gap_sec=0,
    )
    segments = energy_segments(audio, 1000, config)
    assert [segment.end_sample - segment.start_sample for segment in segments] == [2000, 2000, 1000]
