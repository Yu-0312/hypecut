"""Signal registry.

Importing this package registers every built-in detector. Third-party
detectors register themselves through the ``hypecut.signals`` entry-point
group, which is loaded lazily on first use.
"""

from __future__ import annotations

from . import audio as _audio  # noqa: F401
from . import visual as _visual  # noqa: F401
from .base import (  # noqa: F401
    Signal,
    available_signals,
    build_signals,
    get_signal,
    register,
)

_PLUGINS_LOADED = False


def load_plugins() -> list[str]:
    """Import third-party signals declared under the ``hypecut.signals`` group."""
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return []
    _PLUGINS_LOADED = True

    from importlib.metadata import entry_points

    loaded: list[str] = []
    try:
        eps = entry_points(group="hypecut.signals")
    except TypeError:  # pragma: no cover - Python < 3.10 shim
        eps = entry_points().get("hypecut.signals", [])  # type: ignore[assignment]
    for ep in eps:
        try:
            ep.load()
            loaded.append(ep.name)
        except Exception as exc:  # pragma: no cover - plugin errors are non-fatal
            import warnings

            warnings.warn(f"Failed to load signal plugin {ep.name!r}: {exc}", stacklevel=2)
    return loaded


__all__ = ["Signal", "register", "get_signal", "build_signals", "available_signals", "load_plugins"]
