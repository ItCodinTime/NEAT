from __future__ import annotations

from importlib import import_module
from typing import Any

from neat_optim.exceptions import NativeCoreUnavailableError


def load_native_core() -> Any:
    try:
        return import_module("neat_optim._neat_core")
    except ImportError as exc:
        raise NativeCoreUnavailableError(
            "The native NEAT core is not available in this environment."
        ) from exc
