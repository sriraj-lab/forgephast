"""Polygenic phenotype simulation subpackage for forgephast."""

from .config import RunConfig
from .phenotypes.simulate import run_simulation

__all__ = [
    "RunConfig",
    "run_simulation",
]
