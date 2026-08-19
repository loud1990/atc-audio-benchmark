"""End-to-end showcase build orchestration."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from atc_benchmark.audio.core import SAMPLE_RATE, load_wav, measure_audio, write_canonical_wav
from atc_benchmark.config import derived_seed, load_config, resolved_config_dict
from atc_benchmark.degradation import EffectContext, apply_effect_chain
from atc_benchmark.exceptions import BenchmarkError, ConfigurationError, ValidationFailure
from atc_benchmark.ingest import ingest_directory
from atc_benchmark.metadata import write_manifests
from atc_benchmark.models import ManifestRecord, ScenarioConfig, ShowcaseConfig
from atc_benchmark.reporting import generate_dataset_card, generate_report, regenerate_plots
from atc_benchmark.selection import select_reference
from atc_benchmark.validation import validate_difference, validate_record, validate_wav

LOGGER = logging.getLogger("atc_benchmark")


@dataclass(frozen=True)
class BuildResult:
    """Paths and counts produced by a successful build."""

    output_root: Path
    records: list[ManifestRecord]
    warnings: list[str]

    @property
    def manifest_path(self) -> Path:
        return self.output_root / "metadata" / "manifest.parquet"

    @property
    def report_path(self) -> Path:
        return self.output_root / "report" / "index.html"


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def _creation_timestamp() -> str:
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    instant = datetime.fromtimestamp(int(epoch), UTC) if epoch is not None else datetime.now(UTC)
    return instant.isoformat()


def _resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _load_provenance(reference_dir: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for name in ("reference_manifest.json", "segmentation_manifest.json"):
        path = reference_dir / name
        if path.exists():
            rows = json.loads(path.read_text(encoding="utf-8"))
            records.update({str(row["reference_id"]): dict(row) for row in rows})
    return records


def _resolve_reference(
    scenario: ScenarioConfig,
    reference_dir: Path,
    config: ShowcaseConfig,
    scenario_index: int,
) -> Path:
    if config.selection.mode == "automatic" or scenario.reference == "auto":
        return select_reference(reference_dir, config.selection, scenario_index)
    name = (
        scenario.reference if scenario.reference.endswith(".wav") else f"{scenario.reference}.wav"
    )
    path = reference_dir / name
    if not path.exists():
        raise ConfigurationError(
            f"Scenario '{scenario.name}' requested reference '{name}', but {path} does not exist. "
            "Place authorized audio in data/raw and run ingest, or correct the scenario reference."
        )
    return path


def _ensure_references(
    config: ShowcaseConfig, project_root: Path, force_ingest: bool = False
) -> list[dict[str, Any]]:
    raw_dir = _resolve_path(config.paths.raw_dir, project_root)
    reference_dir = _resolve_path(config.paths.reference_dir, project_root)
    reference_dir.mkdir(parents=True, exist_ok=True)
    if list(reference_dir.glob("*.wav")) and not force_ingest:
        return []
    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        raise ConfigurationError(
            f"No canonical references exist in {reference_dir}, and no authorized source audio was found in {raw_dir}."
        )
    records = ingest_directory(raw_dir, reference_dir, config.normalization, force=force_ingest)
    (reference_dir / "reference_manifest.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    for row in records:
        LOGGER.info(
            "[INGEST] %s → %s",
            Path(row["source_parent_file"]).name,
            Path(row["reference_file"]).name,
        )
    return records


def build_showcase(
    config_path: Path,
    *,
    force: bool = False,
    seed_override: int | None = None,
    strict: bool = True,
    project_root: Path | None = None,
) -> BuildResult:
    """Run ingest-if-needed, DSP, metadata, validation, plots, report, and card."""
    config = load_config(config_path)
    if seed_override is not None:
        config.dataset.seed = seed_override
    root = (project_root or Path.cwd()).resolve()
    reference_dir = _resolve_path(config.paths.reference_dir, root)
    _ensure_references(config, root)
    output_root = _resolve_path(config.paths.output_dir, root) / config.dataset.output_name
    if output_root.exists():
        if not force:
            raise ConfigurationError(
                f"Output already exists: {output_root}. Re-run with --force to replace it."
            )
        # Exact configured dataset directory only; raw/reference inputs are never removed.
        shutil.rmtree(output_root)
    for directory in ("reference", "degraded", "metadata", "configs", "plots", "report"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    (output_root / "configs" / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_config_dict(config), sort_keys=False), encoding="utf-8"
    )

    provenance = _load_provenance(reference_dir)
    commit = _git_commit()
    created = _creation_timestamp()
    records: list[ManifestRecord] = []
    warnings: list[str] = []
    failures: list[str] = []
    copied_references: set[str] = set()

    for scenario_index, scenario in enumerate(config.scenarios):
        try:
            reference_path = _resolve_reference(scenario, reference_dir, config, scenario_index)
            reference, rate = load_wav(reference_path)
            if rate != SAMPLE_RATE:
                raise ValidationFailure(
                    f"Reference {reference_path} is {rate} Hz; run ingest to canonicalize it"
                )
            reference_metrics = measure_audio(reference, rate)
            scenario_seed = (
                scenario.seed
                if scenario.seed is not None
                else derived_seed(config.dataset.seed, scenario.id, scenario.name)
            )
            effect_seeds = [
                derived_seed(scenario_seed, str(index), effect.type)
                for index, effect in enumerate(scenario.effects)
            ]
            degraded, effect_results = apply_effect_chain(
                reference,
                scenario.effects,
                effect_seeds,
                EffectContext(sample_rate=rate, reference_dir=reference_dir),
            )
            peak = float(np.max(np.abs(degraded)))
            safety_scale = 1.0
            if peak > 0.999:
                safety_scale = 0.999 / peak
                degraded *= safety_scale
            filename = f"{scenario.id}_{scenario.name}.wav"
            degraded_path = output_root / "degraded" / filename
            write_canonical_wav(degraded_path, degraded)
            # Measure the actual quantized deliverable, not the in-memory precursor.
            final_audio, final_rate = load_wav(degraded_path)
            degraded_metrics = measure_audio(final_audio, final_rate)
            reference_output = output_root / "reference" / reference_path.name
            if reference_path.name not in copied_references:
                shutil.copy2(reference_path, reference_output)
                copied_references.add(reference_path.name)
            errors = validate_wav(
                degraded_path, config.dataset.min_duration_sec, config.dataset.max_duration_sec
            )
            errors.extend(validate_difference(reference, final_audio))
            duration_changing = any(
                effect.type in {"truncate", "codec"} for effect in scenario.effects
            )
            if not duration_changing and abs(len(final_audio) - len(reference)) > 1:
                errors.append(
                    f"unexpected duration change of {(len(final_audio) - len(reference)) / rate:.6f}s"
                )
            metadata: list[dict[str, Any]] = []
            achieved_snr: float | None = None
            for effect_result in effect_results:
                details = {
                    "type": effect_result.type,
                    "parameters": effect_result.parameters,
                    "derived": effect_result.derived,
                }
                metadata.append(details)
                if "achieved_snr_db" in effect_result.derived:
                    achieved_snr = float(effect_result.derived["achieved_snr_db"])
                    LOGGER.info(
                        "[VERIFY] requested SNR=%s dB, achieved=%.2f dB",
                        effect_result.parameters["snr_db"],
                        achieved_snr,
                    )
            if safety_scale != 1:
                metadata[-1]["derived"]["output_safety_linear_scale"] = safety_scale
            source = provenance.get(reference_path.stem, {})
            record = ManifestRecord(
                sample_id=scenario.id,
                scenario_name=scenario.name,
                scenario_description=scenario.description,
                severity=scenario.severity,
                reference_file=f"reference/{reference_path.name}",
                degraded_file=f"degraded/{filename}",
                source_parent_file=str(source.get("source_parent_file", "PROVENANCE_REQUIRED")),
                source_start_sec=float(source.get("source_start_sec", 0)),
                source_end_sec=float(source.get("source_end_sec", reference_metrics.duration_sec)),
                sample_rate=SAMPLE_RATE,
                channels=1,
                duration_sec=degraded_metrics.duration_sec,
                seed=scenario_seed,
                effect_chain=[effect.type for effect in scenario.effects],
                effect_parameters=metadata,
                reference_peak_dbfs=reference_metrics.peak_dbfs,
                degraded_peak_dbfs=degraded_metrics.peak_dbfs,
                reference_rms_dbfs=reference_metrics.rms_dbfs,
                degraded_rms_dbfs=degraded_metrics.rms_dbfs,
                clipping_percentage=degraded_metrics.clipping_percentage,
                rms_delta_db=degraded_metrics.rms_dbfs - reference_metrics.rms_dbfs,
                peak_delta_db=degraded_metrics.peak_dbfs - reference_metrics.peak_dbfs,
                duration_delta_sec=degraded_metrics.duration_sec - reference_metrics.duration_sec,
                spectral_centroid_hz=degraded_metrics.spectral_centroid_hz,
                bandwidth_hz=degraded_metrics.bandwidth_hz,
                achieved_snr_db=achieved_snr,
                creation_timestamp=created,
                pipeline_version=config.dataset.pipeline_version,
                git_commit=commit,
                provenance={
                    "reference_id": reference_path.stem,
                    "source_metadata_available": bool(source),
                },
            )
            errors.extend(validate_record(record))
            if errors:
                raise ValidationFailure(
                    f"Scenario '{scenario.name}' failed validation: {'; '.join(errors)}"
                )
            records.append(record)
            LOGGER.info("[BUILD] %s %s", scenario.id, scenario.name)
            LOGGER.info("[PASS] %s", filename)
        except BenchmarkError as exc:
            failures.append(str(exc))
            LOGGER.error("[FAIL] %s: %s", scenario.name, exc)
            if strict:
                raise

    if failures:
        raise ValidationFailure("Build had scenario failures:\n- " + "\n- ".join(failures))
    if len(records) != len(config.scenarios):
        raise ValidationFailure(
            f"Expected {len(config.scenarios)} outputs but generated {len(records)}"
        )
    write_manifests(records, output_root / "metadata")
    regenerate_plots(output_root, records)
    generate_report(output_root, records, config.dataset.name)
    generate_dataset_card(output_root, config, records)
    return BuildResult(output_root=output_root, records=records, warnings=warnings)


def validate_dataset(
    output_root: Path, minimum_sec: float = 0.1, maximum_sec: float = 60
) -> list[str]:
    """Validate every manifest row and referenced artifact in an existing dataset."""
    from atc_benchmark.metadata import read_jsonl

    records = read_jsonl(output_root / "metadata" / "manifest.jsonl")
    errors: list[str] = []
    for record in records:
        reference_path = output_root / record.reference_file
        degraded_path = output_root / record.degraded_file
        if not reference_path.exists():
            errors.append(f"missing reference file: {reference_path}")
        if not degraded_path.exists():
            errors.append(f"missing degraded file: {degraded_path}")
            continue
        errors.extend(
            f"{record.sample_id}: {error}"
            for error in validate_wav(degraded_path, minimum_sec, maximum_sec)
        )
        errors.extend(f"{record.sample_id}: {error}" for error in validate_record(record))
    return errors
