"""Registry for built-in GWAS engines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .base import GwasEngine
from .engines import Plink2Engine, RegenieEngine


@dataclass(frozen=True)
class EngineRegistration:
    name: str
    config_attr: str
    factory: Callable[[Any], GwasEngine]


_ENGINE_REGISTRY: dict[str, EngineRegistration] = {
    "plink2": EngineRegistration(
        name="plink2",
        config_attr="plink2",
        factory=lambda cfg: Plink2Engine(cfg.gwas.plink2),
    ),
    "regenie": EngineRegistration(
        name="regenie",
        config_attr="regenie",
        factory=lambda cfg: RegenieEngine(cfg.gwas.regenie),
    ),
}


def available_engine_names() -> list[str]:
    """Return stable built-in GWAS engine names."""

    return sorted(_ENGINE_REGISTRY)


def validate_engine_names(engine_names: list[str]) -> None:
    """Raise if any requested engine name is unsupported."""

    valid = set(_ENGINE_REGISTRY)
    invalid = [name for name in engine_names if name not in valid]
    if invalid:
        raise ValueError(f"unsupported GWAS engine(s): {invalid}")


def build_engines(config: Any, selected: list[str] | None = None) -> dict[str, GwasEngine]:
    """Construct configured engine instances in request order."""

    requested = list(config.gwas.engines if selected is None else selected)
    validate_engine_names(requested)
    return {name: _ENGINE_REGISTRY[name].factory(config) for name in requested}


__all__ = [
    "EngineRegistration",
    "available_engine_names",
    "build_engines",
    "validate_engine_names",
]
