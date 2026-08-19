"""Strict configuration and metadata models."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Strict base model that rejects misspelled configuration keys."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class BandpassParams(StrictModel):
    type: Literal["bandpass"]
    low_hz: float = Field(gt=0)
    high_hz: float = Field(gt=0)
    order: int = Field(default=6, ge=2, le=12)

    @model_validator(mode="after")
    def valid_band(self) -> BandpassParams:
        if self.low_hz >= self.high_hz:
            raise ValueError("low_hz must be below high_hz")
        if self.high_hz >= 8000:
            raise ValueError("high_hz must be below the 8 kHz Nyquist frequency")
        return self


class NoiseParams(StrictModel):
    type: Literal["noise"]
    noise_type: Literal["white", "pink", "static", "vhf"] = "white"
    snr_db: float = Field(ge=-10, le=60)
    low_hz: float = Field(default=300, gt=0)
    high_hz: float = Field(default=3400, gt=0, lt=8000)
    seed: int | None = None

    @model_validator(mode="after")
    def valid_noise_band(self) -> NoiseParams:
        if self.low_hz >= self.high_hz:
            raise ValueError("low_hz must be below high_hz")
        return self


class FadingParams(StrictModel):
    type: Literal["fading"]
    depth_db: float = Field(gt=0, le=60)
    rate_hz: float = Field(gt=0, le=20)
    waveform: Literal["sine", "triangle", "random"] = "sine"
    randomness: float = Field(default=0, ge=0, le=1)
    seed: int | None = None


class DropoutParams(StrictModel):
    type: Literal["dropout"]
    count: int = Field(ge=1, le=100)
    min_duration_ms: float = Field(gt=0)
    max_duration_ms: float = Field(gt=0)
    attenuation_db: float = Field(default=80, ge=0, le=120)
    seed: int | None = None

    @model_validator(mode="after")
    def valid_duration(self) -> DropoutParams:
        if self.min_duration_ms > self.max_duration_ms:
            raise ValueError("min_duration_ms must not exceed max_duration_ms")
        return self


class TruncateParams(StrictModel):
    type: Literal["truncate"]
    edge: Literal["beginning", "ending"]
    duration_ms: float = Field(gt=0)
    active_threshold_dbfs: float = Field(default=-45, ge=-100, le=0)


class HardClipParams(StrictModel):
    type: Literal["hard_clip"]
    threshold: float = Field(gt=0, le=1)
    pre_gain_db: float = Field(default=0, ge=-20, le=40)


class SoftClipParams(StrictModel):
    type: Literal["soft_clip"]
    drive: float = Field(gt=0, le=20)


class AGCParams(StrictModel):
    type: Literal["agc_pumping"]
    attack_ms: float = Field(default=10, gt=0)
    release_ms: float = Field(default=250, gt=0)
    modulation_depth_db: float = Field(default=8, ge=0, le=30)
    min_gain_db: float = Field(default=-15, ge=-60, le=20)
    max_gain_db: float = Field(default=12, ge=-20, le=40)
    target_dbfs: float = Field(default=-18, ge=-60, le=-1)

    @model_validator(mode="after")
    def valid_gain_range(self) -> AGCParams:
        if self.min_gain_db >= self.max_gain_db:
            raise ValueError("min_gain_db must be below max_gain_db")
        return self


class ImpulseNoiseParams(StrictModel):
    type: Literal["impulse_noise"]
    events_per_second: float = Field(gt=0, le=100)
    amplitude: float = Field(default=0.35, gt=0, le=2)
    min_duration_ms: float = Field(default=0.5, gt=0)
    max_duration_ms: float = Field(default=4, gt=0)
    polarity: Literal["positive", "negative", "random"] = "random"
    seed: int | None = None

    @model_validator(mode="after")
    def valid_impulse_duration(self) -> ImpulseNoiseParams:
        if self.min_duration_ms > self.max_duration_ms:
            raise ValueError("min_duration_ms must not exceed max_duration_ms")
        return self


class HumParams(StrictModel):
    type: Literal["hum"]
    fundamental_hz: float = Field(default=60, gt=0, lt=1000)
    harmonics: list[int] = Field(default_factory=lambda: [1, 2, 3, 4], min_length=1)
    level_db_below_rms: float = Field(default=18, ge=0, le=80)

    @model_validator(mode="after")
    def valid_harmonics(self) -> HumParams:
        if any(value < 1 for value in self.harmonics):
            raise ValueError("harmonics must contain positive integers")
        if any(self.fundamental_hz * value >= 8000 for value in self.harmonics):
            raise ValueError("hum harmonics must remain below Nyquist")
        return self


class ToneParams(StrictModel):
    type: Literal["tone"]
    frequencies_hz: list[float] = Field(min_length=1)
    level_db_below_rms: float = Field(default=12, ge=-10, le=80)
    drift_hz: float = Field(default=0, ge=0, le=1000)

    @model_validator(mode="after")
    def valid_frequencies(self) -> ToneParams:
        if any(value <= 0 or value + self.drift_hz >= 8000 for value in self.frequencies_hz):
            raise ValueError("tone frequencies including drift must remain between 0 and Nyquist")
        return self


class AdjacentSpeechParams(StrictModel):
    type: Literal["adjacent_speech"]
    secondary_reference: str
    relative_level_db: float = Field(default=-10, ge=-60, le=20)
    start_offset_sec: float = Field(default=0, ge=0)
    low_hz: float | None = Field(default=300, gt=0)
    high_hz: float | None = Field(default=3000, gt=0, lt=8000)

    @model_validator(mode="after")
    def valid_optional_band(self) -> AdjacentSpeechParams:
        if (self.low_hz is None) != (self.high_hz is None):
            raise ValueError("low_hz and high_hz must either both be set or both be null")
        if self.low_hz is not None and self.high_hz is not None and self.low_hz >= self.high_hz:
            raise ValueError("low_hz must be below high_hz")
        return self


class CodecParams(StrictModel):
    type: Literal["codec"]
    codec: Literal["opus", "mp3"]
    bitrate_kbps: int = Field(ge=6, le=320)
    generations: int = Field(default=1, ge=1, le=10)


class PacketLossParams(StrictModel):
    type: Literal["packet_loss"]
    frame_ms: float = Field(default=20, gt=0, le=200)
    loss_probability: float = Field(gt=0, le=1)
    burst_probability: float = Field(default=0, ge=0, le=1)
    concealment: Literal["zeros", "repeat"] = "repeat"
    seed: int | None = None


EffectParams = Annotated[
    BandpassParams
    | NoiseParams
    | FadingParams
    | DropoutParams
    | TruncateParams
    | HardClipParams
    | SoftClipParams
    | AGCParams
    | ImpulseNoiseParams
    | HumParams
    | ToneParams
    | AdjacentSpeechParams
    | CodecParams
    | PacketLossParams,
    Field(discriminator="type"),
]


class DatasetConfig(StrictModel):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    output_name: str = Field(default="showcase_v1", pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    sample_rate: Literal[16000] = 16000
    channels: Literal[1] = 1
    sample_width_bits: Literal[16] = 16
    seed: int = 12345
    min_duration_sec: float = Field(default=0.25, gt=0)
    max_duration_sec: float = Field(default=60, gt=0)
    pipeline_version: str = "1.0.0"

    @model_validator(mode="after")
    def valid_duration_range(self) -> DatasetConfig:
        if self.min_duration_sec >= self.max_duration_sec:
            raise ValueError("min_duration_sec must be below max_duration_sec")
        return self


class PathsConfig(StrictModel):
    raw_dir: str = "data/raw"
    reference_dir: str = "data/reference"
    output_dir: str = "data/output"


class NormalizationConfig(StrictModel):
    mode: Literal["peak_limit", "none", "target_rms"] = "peak_limit"
    peak_dbfs: float = Field(default=-1, ge=-20, le=0)
    target_rms_dbfs: float = Field(default=-22, ge=-60, le=-1)
    prevent_clipping: bool = True


class SegmentationConfig(StrictModel):
    enabled: bool = False
    mode: Literal["energy", "silero"] = "energy"
    threshold_dbfs: float = Field(default=-40, ge=-100, le=0)
    frame_ms: float = Field(default=20, gt=0)
    min_speech_sec: float = Field(default=0.4, gt=0)
    max_clip_sec: float = Field(default=15, gt=0)
    pre_roll_sec: float = Field(default=0.15, ge=0)
    post_roll_sec: float = Field(default=0.25, ge=0)
    merge_gap_sec: float = Field(default=0.35, ge=0)

    @model_validator(mode="after")
    def valid_clip_duration(self) -> SegmentationConfig:
        if self.min_speech_sec > self.max_clip_sec:
            raise ValueError("min_speech_sec must not exceed max_clip_sec")
        return self


class SelectionConfig(StrictModel):
    mode: Literal["explicit", "automatic"] = "explicit"
    minimum_duration_sec: float = Field(default=1, gt=0)
    maximum_duration_sec: float = Field(default=15, gt=0)
    maximum_clipping_percentage: float = Field(default=1, ge=0, le=100)
    minimum_rms_dbfs: float = Field(default=-45, ge=-120, le=0)
    minimum_speech_activity_ratio: float = Field(default=0.05, ge=0, le=1)
    activity_threshold_dbfs: float = Field(default=-45, ge=-100, le=0)

    @model_validator(mode="after")
    def valid_selection_duration(self) -> SelectionConfig:
        if self.minimum_duration_sec > self.maximum_duration_sec:
            raise ValueError("minimum_duration_sec must not exceed maximum_duration_sec")
        return self


class ScenarioConfig(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    name: str = Field(pattern=r"^[a-z0-9_]+$")
    description: str
    severity: Literal["mild", "moderate", "severe", "compound"]
    reference: str
    effects: list[EffectParams] = Field(min_length=1)
    seed: int | None = None


class ShowcaseConfig(StrictModel):
    dataset: DatasetConfig
    paths: PathsConfig = Field(default_factory=PathsConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    scenarios: list[ScenarioConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_scenarios(self) -> ShowcaseConfig:
        ids = [scenario.id for scenario in self.scenarios]
        names = [scenario.name for scenario in self.scenarios]
        if len(ids) != len(set(ids)):
            raise ValueError("scenario ids must be unique")
        if len(names) != len(set(names)):
            raise ValueError("scenario names must be unique")
        return self


class AudioMetrics(StrictModel):
    duration_sec: float
    peak_dbfs: float
    rms_dbfs: float
    approximate_loudness_dbfs: float
    clipping_percentage: float
    dc_offset: float
    spectral_centroid_hz: float
    bandwidth_hz: float


class EffectResult(StrictModel):
    type: str
    parameters: dict[str, Any]
    derived: dict[str, Any] = Field(default_factory=dict)


class ManifestRecord(StrictModel):
    sample_id: str
    scenario_name: str
    scenario_description: str
    severity: str
    reference_file: str
    degraded_file: str
    source_parent_file: str
    source_start_sec: float
    source_end_sec: float
    sample_rate: int
    channels: int
    duration_sec: float
    seed: int
    effect_chain: list[str]
    effect_parameters: list[dict[str, Any]]
    reference_peak_dbfs: float
    degraded_peak_dbfs: float
    reference_rms_dbfs: float
    degraded_rms_dbfs: float
    clipping_percentage: float
    rms_delta_db: float
    peak_delta_db: float
    duration_delta_sec: float
    spectral_centroid_hz: float
    bandwidth_hz: float
    achieved_snr_db: float | None = None
    creation_timestamp: str
    pipeline_version: str
    git_commit: str
    provenance: dict[str, Any] = Field(default_factory=dict)
