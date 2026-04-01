class NEATError(Exception):
    """Base exception for NEAT."""


class ConfigurationError(NEATError):
    """Raised when configuration values are invalid."""


class NativeCoreUnavailableError(NEATError):
    """Raised when the native core is requested but not available."""
