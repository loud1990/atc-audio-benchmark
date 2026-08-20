# ATC Audio Benchmark

A reproducible Python 3.12+ pipeline for building a small, rigorously controlled ATC audio degradation showcase. It preserves authorized source recordings, converts them once to a canonical reference format, applies deterministic additional corruption, validates every deliverable, and generates machine-readable manifests plus a local listening report.

> **Reference audio is not pristine audio.** In this project, reference audio means the unmodified received/source ATC recording used as the baseline for a synthetic degradation. Real radio artifacts may already be present. A degraded sample is controlled additional corruption applied to that existing ATC radio recording.

## Legal and acquisition boundary

Only use recordings you are authorized to access, process, and redistribute. Put manually obtained or explicitly permitted files in `data/raw/`, or implement the `AcquisitionAdapter` protocol for a lawful source. The project contains no scraper, authentication bypass, rate-limit bypass, or third-party ATC recordings. You are responsible for provenance, licensing, privacy, and redistribution rights.

## Prerequisites and installation

- Python 3.12 or newer
- FFmpeg and FFprobe (codec support must include `libopus` and `libmp3lame` for all scenarios)
- [uv](https://docs.astral.sh/uv/)

```bash
git clone <your-repository-url>
cd atc-audio-benchmark
uv sync --extra dev
```

Confirm FFmpeg codec support with `ffmpeg -encoders | grep -E 'libopus|libmp3lame'`.

## Add and prepare source audio

Supported input extensions are MP3, WAV, M4A, FLAC, OGG, and AAC. Original files remain untouched.

```text
data/raw/
├── authorized_recording_a.mp3
└── authorized_recording_b.m4a
```

```bash
uv run atc-benchmark ingest --config configs/showcase_v1.yaml
uv run atc-benchmark inspect --config configs/showcase_v1.yaml
```

Ingestion uses FFmpeg to produce mono, 16 kHz, signed PCM16 WAV references. The default normalization only limits peaks that would clip; it does not force every recording to the same loudness. Configure `normalization.mode` as `none`, `peak_limit`, or `target_rms`.

For long archive files, enable/tune `segmentation` and run:

```bash
uv run atc-benchmark segment --config configs/showcase_v1.yaml
```

Energy segmentation supports minimum active duration, maximum clip length, pre/post roll, and nearby-region merging. The `silero` enum is reserved as an optional integration mode; the core installation intentionally does not pull a heavyweight ML runtime.

## Configure references and build

Edit each scenario's `reference` in `configs/showcase_v1.yaml` after inspecting canonical references. Related severity scenarios intentionally share a baseline. `selection.mode: automatic` provides deterministic suitability ranking by duration, clipping, and RMS.

```bash
uv run atc-benchmark build --config configs/showcase_v1.yaml
```

Replace an existing build or override the root seed:

```bash
uv run atc-benchmark build --config configs/showcase_v1.yaml --force
uv run atc-benchmark build --config configs/showcase_v1.yaml --force --seed 12345
```

The initial config contains exactly 25 scenarios, but the runner accepts any positive scenario count. Each configured scenario creates exactly one degraded WAV.

## Outputs

A checked-in set of 25 geographically distributed reference/degraded pairs is available in
[`examples/regional_atc/`](examples/regional_atc/). Its README documents the area, facility,
pairing, provenance, and reuse notes for every recording.

```text
data/output/showcase_v1/
├── reference/             # only references used by this build
├── degraded/              # exactly one WAV per scenario
├── metadata/
│   ├── manifest.csv
│   ├── manifest.jsonl
│   └── manifest.parquet
├── configs/resolved_config.yaml
├── plots/                 # aligned waveform/spectrogram comparisons
├── report/index.html
└── DATASET_CARD.md
```

```bash
xdg-open data/output/showcase_v1/report/index.html
uv run atc-benchmark validate --output data/output/showcase_v1
```

The static report has paired audio players, level metrics, parameters and derived random events, comparison plots, and a manual listening checklist. It needs no web server.

## Effects and reproducibility

The transform registry includes stable Butterworth bandpass filtering; exact-SNR white, pink, receiver-static, and VHF-band noise; fading; sample-exact dropouts; active-aware edge truncation; hard and soft clipping; envelope-driven AGC pumping; impulse crackle; harmonic hum; drifting tones; secondary ATC mixing; low-rate Opus and repeated MP3 round trips; documented frame erasure; and composable weak-VHF corruption.

Every randomized effect receives an explicit seed or a stable SHA-256-derived seed. Derived dropout intervals, impulse timestamps, packet indices, achieved SNR, codec arguments, and safety scaling are stored. PCM16 WAVs from non-codec transforms are bit-reproducible for identical inputs/configuration/library versions. FFmpeg transforms are verified numerically rather than promised bit-identical across different FFmpeg builds.

## Commands

```text
atc-benchmark ingest    decode authorized raw inputs
atc-benchmark segment   find activity regions in canonical references
atc-benchmark inspect   show suitability and signal metrics
atc-benchmark build     execute the full pipeline
atc-benchmark validate  validate an existing output
atc-benchmark report    regenerate plots and HTML
atc-benchmark clean     remove only the configured generated dataset
```

## Development

Synthetic waveforms are generated in tests; no external ATC audio is required.

```bash
make test
make lint
make typecheck
make e2e
```

CI installs FFmpeg and runs linting, strict type checking, all tests, and the fixture build. Codec tests skip locally when FFmpeg or the relevant encoder is unavailable.

## Extending the project

Add a strict Pydantic parameter model to `models.py`, implement the transform protocol in `degradation/effects.py`, register it in `create_effect`, and add unit/signal-level tests. Dataset scale is configuration-driven: add scenarios or generate YAML with more reference assignments; the build engine does not assume a count of 25.

Optional transcription, intelligibility, and learned quality metrics should remain isolated extras. Synthetic corruption is a controlled test instrument, not a perfect model of RF propagation or proof of live operational performance.
