"""Command-line interface for all benchmark stages."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Annotated

import typer

from atc_benchmark.audio.core import load_wav, measure_audio, write_canonical_wav
from atc_benchmark.config import load_config
from atc_benchmark.exceptions import BenchmarkError, ConfigurationError
from atc_benchmark.ingest import ingest_directory
from atc_benchmark.metadata import read_jsonl
from atc_benchmark.pipeline import build_showcase, validate_dataset
from atc_benchmark.reporting import generate_report, regenerate_plots
from atc_benchmark.segmentation import energy_segments

app = typer.Typer(
    no_args_is_help=True, help="Build rigorously controlled ATC degradation showcases."
)


def _logging(verbose: bool = False) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def _abort(exc: Exception) -> None:
    typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
    raise typer.Exit(1) from exc


@app.command()
def ingest(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/showcase_v1.yaml"),
    force: Annotated[bool, typer.Option(help="Re-decode existing canonical references.")] = False,
) -> None:
    """Decode authorized raw inputs into canonical reference WAV files."""
    _logging()
    try:
        settings = load_config(config)
        records = ingest_directory(
            Path(settings.paths.raw_dir),
            Path(settings.paths.reference_dir),
            settings.normalization,
            force=force,
        )
        manifest = Path(settings.paths.reference_dir) / "reference_manifest.json"
        manifest.write_text(json.dumps(records, indent=2), encoding="utf-8")
        for row in records:
            typer.echo(
                f"[INGEST] {Path(row['source_parent_file']).name} → {Path(row['reference_file']).name}"
            )
        typer.echo(f"Canonical references: {len(records)}\nManifest: {manifest}")
    except (BenchmarkError, OSError) as exc:
        _abort(exc)


@app.command()
def segment(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/showcase_v1.yaml"),
) -> None:
    """Extract energy-detected transmission regions from canonical references."""
    try:
        settings = load_config(config)
        if settings.segmentation.mode == "silero":
            raise ConfigurationError(
                "Silero VAD is an optional integration and is not installed by the core package; "
                "set segmentation.mode to 'energy'."
            )
        reference_dir = Path(settings.paths.reference_dir)
        rows = []
        for parent in sorted(reference_dir.glob("ref_[0-9][0-9][0-9].wav")):
            audio, rate = load_wav(parent)
            for index, item in enumerate(
                energy_segments(audio, rate, settings.segmentation), start=1
            ):
                target = reference_dir / f"{parent.stem}_seg_{index:03d}.wav"
                write_canonical_wav(target, audio[item.start_sample : item.end_sample])
                rows.append(
                    {
                        "reference_id": target.stem,
                        "reference_file": str(target),
                        "source_parent_file": str(parent),
                        "source_start_sec": item.start_sec,
                        "source_end_sec": item.end_sec,
                    }
                )
                typer.echo(
                    f"[SEGMENT] {parent.name} {item.start_sec:.3f}-{item.end_sec:.3f}s -> {target.name}"
                )
        (reference_dir / "segmentation_manifest.json").write_text(
            json.dumps(rows, indent=2), encoding="utf-8"
        )
        typer.echo(f"Segments: {len(rows)}")
    except (BenchmarkError, OSError) as exc:
        _abort(exc)


@app.command()
def inspect(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/showcase_v1.yaml"),
) -> None:
    """List reference suitability metrics for manual selection."""
    try:
        settings = load_config(config)
        typer.echo("file\tduration_s\trms_dbfs\tpeak_dbfs\tclipping_%\tcentroid_hz")
        for path in sorted(Path(settings.paths.reference_dir).glob("*.wav")):
            audio, rate = load_wav(path)
            metrics = measure_audio(audio, rate)
            typer.echo(
                f"{path.name}\t{metrics.duration_sec:.3f}\t{metrics.rms_dbfs:.2f}\t{metrics.peak_dbfs:.2f}\t{metrics.clipping_percentage:.4f}\t{metrics.spectral_centroid_hz:.1f}"
            )
    except (BenchmarkError, OSError) as exc:
        _abort(exc)


@app.command()
def build(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/showcase_v1.yaml"),
    force: Annotated[
        bool, typer.Option(help="Replace only this configured output dataset.")
    ] = False,
    seed: Annotated[int | None, typer.Option(help="Override the root deterministic seed.")] = None,
    strict: Annotated[
        bool, typer.Option("--strict/--no-strict", help="Stop on the first scenario failure.")
    ] = True,
) -> None:
    """Execute the complete showcase pipeline."""
    _logging()
    try:
        result = build_showcase(config, force=force, seed_override=seed, strict=strict)
    except (BenchmarkError, OSError) as exc:
        _abort(exc)
    typer.echo(
        "\nShowcase build complete\n\n"
        f"References: {len({record.reference_file for record in result.records})}\n"
        f"Scenarios: {len(result.records)}\nGenerated: {len(result.records)}\n"
        f"Passed validation: {len(result.records)}\nWarnings: {len(result.warnings)}\nFailed: 0\n\n"
        f"Manifest:\n{result.manifest_path}\n\nReport:\n{result.report_path}"
    )


@app.command(name="validate")
def validate_command(
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("data/output/showcase_v1"),
) -> None:
    """Validate an existing built dataset."""
    try:
        errors = validate_dataset(output)
    except (BenchmarkError, OSError) as exc:
        _abort(exc)
    if errors:
        for error in errors:
            typer.secho(f"[FAIL] {error}", fg=typer.colors.RED)
        raise typer.Exit(1)
    typer.echo("[PASS] All audio and metadata validation checks passed")


@app.command()
def report(
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("data/output/showcase_v1"),
) -> None:
    """Regenerate plots and the listening report for an existing build."""
    try:
        records = read_jsonl(output / "metadata" / "manifest.jsonl")
        regenerate_plots(output, records)
        path = generate_report(output, records, output.name)
        typer.echo(f"Report: {path}")
    except (BenchmarkError, OSError) as exc:
        _abort(exc)


@app.command()
def clean(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/showcase_v1.yaml"),
    yes: Annotated[bool, typer.Option("--yes", help="Confirm deletion without a prompt.")] = False,
) -> None:
    """Remove only the configured generated output; preserve raw and references."""
    try:
        settings = load_config(config)
        target = (Path(settings.paths.output_dir) / settings.dataset.output_name).resolve()
        if not target.exists():
            typer.echo(f"Nothing to clean: {target}")
            return
        if not yes and not typer.confirm(f"Remove generated dataset {target}?"):
            raise typer.Abort()
        shutil.rmtree(target)
        typer.echo(f"Removed generated dataset: {target} (not recoverable by this tool)")
    except (BenchmarkError, OSError) as exc:
        _abort(exc)


if __name__ == "__main__":
    app()
