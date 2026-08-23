"""Refiner registry (stage 2 of the hybrid pipeline)."""

from __future__ import annotations

from . import builtin as _builtin  # noqa: F401
from . import similarity as _similarity  # noqa: F401
from .base import Refiner, available_refiners, build_refiners, get_refiner, register  # noqa: F401

_PLUGINS_LOADED = False


def load_plugins() -> list[str]:
    """Import third-party refiners from the ``hypecut.refiners`` entry-point group."""
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return []
    _PLUGINS_LOADED = True

    from importlib.metadata import entry_points

    loaded: list[str] = []
    try:
        eps = entry_points(group="hypecut.refiners")
    except TypeError:  # pragma: no cover
        eps = entry_points().get("hypecut.refiners", [])  # type: ignore[assignment]
    for ep in eps:
        try:
            ep.load()
            loaded.append(ep.name)
        except Exception as exc:  # pragma: no cover
            import warnings

            warnings.warn(f"Failed to load refiner plugin {ep.name!r}: {exc}", stacklevel=2)
    return loaded


__all__ = [
    "Refiner",
    "register",
    "get_refiner",
    "build_refiners",
    "available_refiners",
    "load_plugins",
]
