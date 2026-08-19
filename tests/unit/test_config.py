from pathlib import Path

import pytest
import yaml

from atc_benchmark.config import derived_seed, load_config
from atc_benchmark.exceptions import ConfigurationError
from atc_benchmark.models import BandpassParams, DropoutParams


def test_showcase_has_exactly_25_scenarios() -> None:
    config = load_config(Path("configs/showcase_v1.yaml"))
    assert len(config.scenarios) == 25
    assert len({scenario.id for scenario in config.scenarios}) == 25


def test_strict_config_rejects_unknown_key(tmp_path: Path) -> None:
    payload = yaml.safe_load(Path("configs/showcase_v1.yaml").read_text())
    payload["dataset"]["typo"] = True
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ConfigurationError, match="typo"):
        load_config(path)


def test_parameter_ranges_fail_fast() -> None:
    with pytest.raises(ValueError, match="low_hz"):
        BandpassParams(type="bandpass", low_hz=4000, high_hz=3000)
    with pytest.raises(ValueError, match="min_duration_ms"):
        DropoutParams(type="dropout", count=2, min_duration_ms=200, max_duration_ms=100)


def test_derived_seed_is_stable_and_namespaced() -> None:
    assert derived_seed(123, "a", "b") == derived_seed(123, "a", "b")
    assert derived_seed(123, "a", "b") != derived_seed(123, "a", "c")
