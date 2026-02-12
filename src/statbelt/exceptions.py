class StatbeltError(Exception):
    """Base exception for statbelt errors."""


class ConfigurationError(StatbeltError):
    """Raised when a harness configuration is incomplete or inconsistent."""


class StateError(StatbeltError):
    """Raised when an operation is not valid for the current harness state."""


class ValidationError(StatbeltError):
    """Raised when user-provided values fail validation."""
