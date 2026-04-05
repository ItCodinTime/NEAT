"""Custom exception hierarchy for NEAT."""

class NEATError(Exception):
    """Base exception for NEAT."""


class ConfigurationError(NEATError):
    """Raised when configuration values are invalid."""


class NativeCoreUnavailableError(NEATError):
    """Raised when the native core is requested but not available."""


class ShapeError(NEATError):
    """Raised when tensor or gradient shapes do not match the API contract."""
