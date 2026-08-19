from pathlib import Path

import numpy as np

from atc_benchmark.audio.core import write_canonical_wav
from atc_benchmark.models import SelectionConfig
from atc_benchmark.selection import select_reference
from atc_benchmark.validation import validate_wav


def test_automatic_selection_rejects_silent_and_clipped(
    tmp_path: Path, tone_audio: np.ndarray
) -> None:
    write_canonical_wav(tmp_path / "a_silent.wav", np.zeros_like(tone_audio))
    write_canonical_wav(tmp_path / "b_good.wav", tone_audio)
    write_canonical_wav(tmp_path / "c_clipped.wav", np.ones_like(tone_audio))
    config = SelectionConfig(
        mode="automatic",
        minimum_duration_sec=1,
        maximum_duration_sec=3,
        maximum_clipping_percentage=1,
        minimum_rms_dbfs=-40,
    )
    assert select_reference(tmp_path, config).name == "b_good.wav"


def test_validation_reports_wrong_format(tmp_path: Path) -> None:
    import soundfile as sf

    path = tmp_path / "wrong.wav"
    sf.write(path, np.ones(8000), 8000, subtype="FLOAT")
    errors = validate_wav(path, 0.1, 2)
    assert any("sample rate" in error for error in errors)
    assert any("PCM_16" in error for error in errors)
