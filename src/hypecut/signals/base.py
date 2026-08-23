"""The signal plugin contract.

A signal answers one question about every moment of the video: *how
interesting is right now, by my particular measure?* It returns a raw
float per grid step. Normalisation, weighting and fusion are somebody
else's job — write the honest measurement and stop.

Adding a detector is three steps:

1. Subclass :class:`Signal`, implement ``compute``.
2. Decorate with ``@register("my_signal")``.
3. Add its name to ``signals.enabled`` in a profile.

Signals must be cheap and must not decode the video themselves: everything
available is already on the :class:`~hypecut.types.AnalysisContext`.
"""

from __future__ import annotations

import abc
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from ..types import AnalysisContext, SignalTrack

__all__ = ["Signal", "register", "get_signal", "build_signals", "available_signals"]

_REGISTRY: dict[str, type[Signal]] = {}


class Signal(abc.ABC):
    """Base class for all detectors."""

    #: Registry key; set by :func:`register`.
    name: str = "signal"
    #: Human-readable one-liner shown by ``hypecut signals``.
    description: str = ""
    #: If True the signal is skipped when the input has no audio track.
    requires_audio: bool = False
    #: If True the signal is skipped when frames could not be decoded.
    requires_video: bool = False
    #: Smallest rise above this signal's own median that means anything, in
    #: whatever units ``compute`` returns. Only :func:`hypecut.fusion.prominence`
    #: reads it, and only to answer "does this video contain anything at all".
    #:
    #: It exists because a ratio has no opinion about scale. In footage that
    #: never changes, ``scene_change`` has a median-absolute-deviation near
    #: zero, so compression flicker of five hundredths of a luma level divides
    #: out to "nine times the usual" — indistinguishable, as a ratio, from a
    #: real cut. A signal that can say "below 1.5 luma levels is nothing"
    #: closes that hole, and only the signal itself knows the number.
    #:
    #: Leave it 0.0 when the signal's output has no physically meaningful
    #: unit; that abstains from the check rather than guessing.
    noise_floor: float = 0.0

    def __init__(self, **params: Any) -> None:
        self.params = params

    @abc.abstractmethod
    def compute(self, ctx: AnalysisContext) -> np.ndarray:
        """Return a ``(T,)`` float array aligned to ``ctx.times``."""

    def applicable(self, ctx: AnalysisContext) -> bool:
        if self.requires_audio and (ctx.audio is None or ctx.audio.size == 0):
            return False
        return not (self.requires_video and (ctx.gray is None or ctx.gray.shape[0] == 0))

    def track(self, ctx: AnalysisContext, weight: float = 1.0) -> SignalTrack:
        values = np.asarray(self.compute(ctx), dtype=np.float64)
        values = _fit(values, ctx.n)
        return SignalTrack(
            name=self.name, values=values, weight=float(weight), noise_floor=float(self.noise_floor)
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging nicety
        return f"<Signal {self.name} {self.params}>"


def register(name: str) -> Callable[[type[Signal]], type[Signal]]:
    """Class decorator that adds a signal to the global registry."""

    def wrap(cls: type[Signal]) -> type[Signal]:
        if name in _REGISTRY:
            raise ValueError(f"Signal {name!r} is already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return wrap


def get_signal(name: str) -> type[Signal]:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown signal {name!r}. Available: {', '.join(sorted(_REGISTRY))}"
        ) from None


def available_signals() -> dict[str, str]:
    """Map of registered signal name -> description."""
    return {k: (v.description or "").strip() for k, v in sorted(_REGISTRY.items())}


def build_signals(
    names: Iterable[str], params: dict[str, dict[str, Any]] | None = None
) -> list[Signal]:
    params = params or {}
    return [get_signal(n)(**params.get(n, {})) for n in names]


def _fit(values: np.ndarray, n: int) -> np.ndarray:
    """Coerce a signal to exactly ``n`` samples (pad with edge, or trim)."""
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if values.shape[0] == n:
        return values
    if values.shape[0] == 0:
        return np.zeros(n, dtype=np.float64)
    if values.shape[0] > n:
        return values[:n]
    pad = np.full(n - values.shape[0], values[-1], dtype=np.float64)
    return np.concatenate([values, pad])
