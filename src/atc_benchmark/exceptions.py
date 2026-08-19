"""Domain exceptions with actionable user-facing messages."""


class BenchmarkError(Exception):
    """Base class for expected pipeline failures."""


class ConfigurationError(BenchmarkError):
    """A configuration is invalid or internally inconsistent."""


class AudioError(BenchmarkError):
    """Audio decoding, encoding, or signal processing failed."""


class ValidationFailure(BenchmarkError):
    """A generated artifact failed mandatory validation."""
