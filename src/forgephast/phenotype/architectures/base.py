"""Architecture plugin interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ArchitectureInput:
    """Inputs required for one architecture simulation call."""

    sample_ids: List[str]
    sample_populations: List[str]
    dosage_df: pd.DataFrame
    membership_df: pd.DataFrame
    beta_df: pd.DataFrame
    variant_id_column: str
    haplotypes: Optional[np.ndarray] = None
    local_ancestry_codes: Optional[np.ndarray] = None
    ancestry_order: Optional[List[str]] = None
    la_chunk_n_samples: Optional[int] = None
    la_chunk_n_variants: Optional[int] = None
    use_numba: bool = False


class ArchitecturePlugin(ABC):
    """Base plugin API for phenotype genetic architecture models."""

    @abstractmethod
    def name(self) -> str:
        """Return plugin name."""

    @abstractmethod
    def compute_genetic_component(self, data: ArchitectureInput) -> Tuple[np.ndarray, pd.DataFrame]:
        """Compute per-sample genetic component and return detailed per-variant effect table."""


PLUGIN_REGISTRY: Dict[str, type[ArchitecturePlugin]] = {}


def register_plugin(key: str, plugin_cls: type[ArchitecturePlugin]) -> None:
    """Register architecture plugin class."""
    PLUGIN_REGISTRY[key] = plugin_cls


def create_plugin(key: str) -> ArchitecturePlugin:
    """Instantiate plugin by key."""
    if key not in PLUGIN_REGISTRY:
        raise ValueError(f"Unknown architecture plugin: {key}")
    return PLUGIN_REGISTRY[key]()
