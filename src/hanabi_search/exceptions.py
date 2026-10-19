"""Exception hierarchy for hanabi_search."""


class HanabiSearchError(Exception):
    """Base class for all errors raised by this package."""


class EngineLoadError(HanabiSearchError):
    """The engine's shared library could not be found or loaded."""


class EngineCallError(HanabiSearchError):
    """An engine call reported failure (e.g. a C API function returned false)."""


class IllegalMoveError(HanabiSearchError):
    """A move was requested that is not legal in the current state."""


class SerializationError(HanabiSearchError):
    """An engine object could not be serialized or deserialized."""


class ConfigError(HanabiSearchError):
    """An experiment or agent configuration is invalid."""


class NonFiniteLossError(HanabiSearchError):
    """A training loss became NaN or infinite."""
