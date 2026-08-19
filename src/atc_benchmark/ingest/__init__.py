"""Authorized local ingestion and extension interfaces."""

from .acquisition import AcquisitionAdapter, LocalDirectoryAdapter
from .pipeline import ingest_directory

__all__ = ["AcquisitionAdapter", "LocalDirectoryAdapter", "ingest_directory"]
