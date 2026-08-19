"""Extension point for explicitly permitted source acquisition."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Protocol


class AcquisitionAdapter(Protocol):
    """Adapter implemented by user-supplied, legally authorized acquisition code."""

    def acquire(self, destination: Path) -> list[Path]:
        """Place authorized source files in destination and return their paths."""
        ...


class LocalDirectoryAdapter:
    """Treat already-present local audio files as the acquisition source."""

    extensions: ClassVar[set[str]] = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}

    def acquire(self, destination: Path) -> list[Path]:
        destination.mkdir(parents=True, exist_ok=True)
        return sorted(
            path for path in destination.iterdir() if path.suffix.lower() in self.extensions
        )
