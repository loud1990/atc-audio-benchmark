"""Static plots, listening report, and dataset-card generation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Environment, select_autoescape
from scipy import signal

from atc_benchmark.audio.core import FloatArray, load_wav
from atc_benchmark.models import ManifestRecord, ShowcaseConfig


def create_comparison_plot(
    reference: FloatArray, degraded: FloatArray, sample_rate: int, destination: Path, title: str
) -> None:
    """Create aligned waveform and spectrogram panels with shared scales."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    maximum = max(float(np.max(np.abs(reference))), float(np.max(np.abs(degraded))), 0.05)
    figure, axes = plt.subplots(2, 2, figsize=(12, 6), constrained_layout=True)
    for column, (audio, label) in enumerate(((reference, "Reference"), (degraded, "Degraded"))):
        time = np.arange(len(audio)) / sample_rate
        axes[0, column].plot(
            time, audio, linewidth=0.35, color="#2563eb" if column == 0 else "#dc2626"
        )
        axes[0, column].set(
            title=f"{label} waveform",
            xlabel="Time (s)",
            ylabel="Amplitude",
            ylim=(-maximum, maximum),
        )
        frequencies, times, spectrum = signal.spectrogram(
            audio, fs=sample_rate, nperseg=512, noverlap=384
        )
        db = 10 * np.log10(np.maximum(spectrum, 1e-12))
        axes[1, column].pcolormesh(
            times, frequencies, db, shading="auto", cmap="magma", vmin=-100, vmax=-20
        )
        axes[1, column].set(
            title=f"{label} spectrogram", xlabel="Time (s)", ylabel="Frequency (Hz)", ylim=(0, 8000)
        )
    figure.suptitle(title)
    figure.savefig(destination, dpi=120)
    plt.close(figure)


_REPORT_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{ dataset_name }} — degradation showcase</title>
<style>
body{font:15px/1.5 system-ui,sans-serif;margin:0;background:#f3f4f6;color:#111827}main{max-width:1120px;margin:auto;padding:2rem}header,.card{background:white;border-radius:12px;padding:1.3rem;margin-bottom:1.2rem;box-shadow:0 1px 4px #0002}.card h2{margin-top:0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:1rem}audio{width:100%}img{width:100%;height:auto}.metrics{font-family:ui-monospace,monospace;font-size:.9rem}details pre{white-space:pre-wrap;overflow:auto;background:#f8fafc;padding:.8rem}.checklist{columns:2}small{color:#4b5563}
</style></head><body><main><header><h1>{{ dataset_name }}</h1><p>{{ count }} controlled additional corruptions applied to received/source ATC reference audio. “Reference” does not mean pristine speech.</p></header>
{% for item in items %}<section class="card" id="{{ item.record.sample_id }}"><h2>{{ item.record.sample_id }} — {{ item.record.scenario_name }}</h2><p>{{ item.record.scenario_description }} <small>Severity: {{ item.record.severity }}</small></p>
<div class="grid"><div><strong>Reference</strong><audio controls preload="metadata" src="{{ item.reference_audio }}"></audio></div><div><strong>Degraded</strong><audio controls preload="metadata" src="{{ item.degraded_audio }}"></audio></div></div>
<p class="metrics">Duration {{ '%.3f'|format(item.record.duration_sec) }} s · RMS {{ '%.2f'|format(item.record.reference_rms_dbfs) }} → {{ '%.2f'|format(item.record.degraded_rms_dbfs) }} dBFS · Peak {{ '%.2f'|format(item.record.reference_peak_dbfs) }} → {{ '%.2f'|format(item.record.degraded_peak_dbfs) }} dBFS · Clipping {{ '%.4f'|format(item.record.clipping_percentage) }}%</p>
<img src="{{ item.plot }}" alt="Waveform and spectrogram comparison for {{ item.record.scenario_name }}">
<details><summary>Effect parameters and derived events</summary><pre>{{ item.parameters }}</pre></details>
<div class="checklist"><label><input type="checkbox"> reference is intelligible</label><br><label><input type="checkbox"> degraded output is audible</label><br><label><input type="checkbox"> degradation resembles intended effect</label><br><label><input type="checkbox"> output does not sound unintentionally corrupted</label><br><label><input type="checkbox"> scenario is materially different from reference</label><br><label><input type="checkbox"> degradation severity appears reasonable</label></div></section>{% endfor %}
</main></body></html>"""


def generate_report(output_root: Path, records: list[ManifestRecord], dataset_name: str) -> Path:
    """Generate a self-contained static index that references local WAV and PNG assets."""
    report_dir = output_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for record in records:
        items.append(
            {
                "record": record,
                "reference_audio": f"../{record.reference_file}",
                "degraded_audio": f"../{record.degraded_file}",
                "plot": f"../plots/{record.sample_id}.png",
                "parameters": json.dumps(record.effect_parameters, indent=2, sort_keys=True),
            }
        )
    environment = Environment(autoescape=select_autoescape(["html"]))
    content = environment.from_string(_REPORT_TEMPLATE).render(
        dataset_name=dataset_name, count=len(records), items=items
    )
    path = report_dir / "index.html"
    path.write_text(content, encoding="utf-8")
    return path


def generate_dataset_card(
    output_root: Path, config: ShowcaseConfig, records: list[ManifestRecord]
) -> Path:
    """Write a transparent limitations and provenance card for a built dataset."""
    scenarios = "\n".join(
        f"- `{record.sample_id}` — {record.scenario_name}: {record.scenario_description}"
        for record in records
    )
    content = f"""# {config.dataset.name}

## Purpose and intended use

This small showcase evaluates systems that attempt to improve degraded ATC audio. It contains controlled additional corruption applied to existing received/source ATC radio recordings.

## Terminology

**Reference audio** means the unmodified received/source ATC recording used as the baseline for a synthetic degradation. It is not pristine, studio-clean, or ground-truth clean speech. **Degraded audio** adds a known synthetic corruption layer to that reference.

## Format and reproducibility

All audio is mono, 16 kHz, signed PCM16 WAV. Dataset version: `{config.dataset.version}`. Root random seed: `{config.dataset.seed}`. Full parameters, derived random events, software version, and commit are recorded in each manifest row; the resolved configuration is included under `configs/`.

## Source provenance and licensing

Source provenance must be completed by the dataset publisher: **[SOURCE, DATES, AIRPORT/FREQUENCY, LICENSE/PERMISSION]**. Users are responsible for ensuring that they have permission to use and redistribute source audio. This repository does not acquire restricted recordings or grant rights to any input.

## Scenarios

{scenarios}

## Metadata

CSV, JSONL, and Parquet manifests identify reference and degraded paths, source parent/timestamps, canonical format, seed/effect chain, signal metrics, creation timestamp, pipeline version, commit, and optional provenance fields.

## Limitations

- Synthetic degradation is not a perfect simulation of real RF propagation, receiver electronics, codec networks, operator behavior, or acoustic environments.
- Existing radio artifacts in the reference can interact with added effects.
- Automatic activity detection is energy-based and may mistake noise for speech.
- Packet loss uses documented time-domain frame erasure and concealment rather than transport-level packet manipulation.
- Approximate loudness is RMS-derived rather than gated broadcast LUFS.

Do not use this dataset alone to claim performance on all live ATC channels, speakers, airports, radios, or safety-critical conditions.
"""
    path = output_root / "DATASET_CARD.md"
    path.write_text(content, encoding="utf-8")
    return path


def regenerate_plots(output_root: Path, records: list[ManifestRecord]) -> None:
    """Create all comparison plots from manifest paths."""
    for record in records:
        reference, rate = load_wav(output_root / record.reference_file)
        degraded, _ = load_wav(output_root / record.degraded_file)
        create_comparison_plot(
            reference,
            degraded,
            rate,
            output_root / "plots" / f"{record.sample_id}.png",
            record.scenario_name,
        )
