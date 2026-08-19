"""Multi-format manifest serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from atc_benchmark.models import ManifestRecord


def _flat_record(record: ManifestRecord) -> dict[str, Any]:
    data = record.model_dump(mode="json")
    data["effect_chain"] = json.dumps(data["effect_chain"], separators=(",", ":"))
    data["effect_parameters"] = json.dumps(
        data["effect_parameters"], separators=(",", ":"), sort_keys=True
    )
    data["provenance"] = json.dumps(data["provenance"], separators=(",", ":"), sort_keys=True)
    return data


def write_manifests(records: list[ManifestRecord], directory: Path) -> None:
    """Write CSV, JSONL, and Parquet representations of the same records."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "manifest.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(record.model_dump_json() + "\n")
    pd.DataFrame([_flat_record(record) for record in records]).to_csv(
        directory / "manifest.csv", index=False
    )
    # Arrow preserves lists and nested objects when constructed from native records.
    pd.DataFrame([record.model_dump(mode="json") for record in records]).to_parquet(
        directory / "manifest.parquet", index=False, engine="pyarrow"
    )


def read_jsonl(path: Path) -> list[ManifestRecord]:
    """Read and validate a JSONL manifest."""
    return [
        ManifestRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
