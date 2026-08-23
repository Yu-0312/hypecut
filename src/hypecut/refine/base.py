"""The refiner plugin contract — stage 2 of the hybrid pipeline.

Stage 1 (signals) is cheap, runs over the whole video, and is deliberately
generous: it proposes far more candidates than the reel needs. A refiner
sees only those candidates — typically a few dozen windows — and may
rescore, reorder, retime or reject them. That is what makes an expensive
model affordable: it never touches the 99% of footage that was obviously
boring.

Refiners must degrade gracefully. If a model is not installed, say so and
return the candidates untouched rather than failing the run.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any

from ..types import AnalysisContext, Candidate, VideoInfo

__all__ = ["Refiner", "register", "get_refiner", "build_refiners", "available_refiners"]

_REGISTRY: dict[str, type[Refiner]] = {}


class Refiner(abc.ABC):
    """Rescore or filter stage-1 candidates."""

    name: str = "refiner"
    description: str = ""

    #: The decoded frames and audio, set by the pipeline before ``refine``
    #: runs. Optional on purpose: it is an attribute rather than an argument
    #: so that every refiner written against the older two-argument signature
    #: keeps working untouched. It is ``None`` when a refiner is driven
    #: directly rather than through :func:`~hypecut.pipeline.analyze`, so a
    #: refiner that uses it must handle that and degrade to doing nothing.
    ctx: AnalysisContext | None = None

    def __init__(self, **params: Any) -> None:
        self.params = params

    def available(self) -> tuple[bool, str]:
        """Return ``(usable, reason)``; used to skip missing optional deps."""
        return True, ""

    @abc.abstractmethod
    def refine(self, info: VideoInfo, candidates: list[Candidate]) -> list[Candidate]:
        """Return a (possibly reordered, reweighted, filtered) candidate list."""

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Refiner {self.name} {self.params}>"


def register(name: str) -> Callable[[type[Refiner]], type[Refiner]]:
    def wrap(cls: type[Refiner]) -> type[Refiner]:
        if name in _REGISTRY:
            raise ValueError(f"Refiner {name!r} is already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return wrap


def get_refiner(name: str) -> type[Refiner]:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown refiner {name!r}. Available: {', '.join(sorted(_REGISTRY))}"
        ) from None


def available_refiners() -> dict[str, str]:
    return {k: (v.description or "").strip() for k, v in sorted(_REGISTRY.items())}


def build_refiners(
    names: list[str], params: dict[str, dict[str, Any]] | None = None
) -> list[Refiner]:
    params = params or {}
    return [get_refiner(n)(**params.get(n, {})) for n in names]
