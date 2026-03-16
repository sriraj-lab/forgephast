"""Genetic map loaders and interpolation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd



def load_genetic_map(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load map as (positions_bp, centimorgans).

    Supported formats:
    - 2 columns: bp, cM
    - 4+ columns (PLINK-style): chr, id, cM, bp
    """

    p = Path(path)
    # Try tab first; fallback to whitespace if tab parsing collapses to one column.
    try:
        df = pd.read_csv(p, sep="\t", header=None, comment="#")
    except Exception:
        df = pd.DataFrame()

    if df.shape[1] < 2:
        df = pd.read_csv(p, sep=r"\s+", header=None, comment="#")

    if df.shape[1] < 2:
        raise ValueError(f"genetic map file must have at least 2 columns: {p}")

    if df.shape[1] >= 4:
        cm = pd.to_numeric(df.iloc[:, 2], errors="coerce")
        bp = pd.to_numeric(df.iloc[:, 3], errors="coerce")
    else:
        bp = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        cm = pd.to_numeric(df.iloc[:, 1], errors="coerce")

    valid = (~bp.isna()) & (~cm.isna())
    bp = bp[valid].astype(int).to_numpy()
    cm = cm[valid].astype(float).to_numpy()

    order = np.argsort(bp)
    bp = bp[order]
    cm = cm[order]

    if bp.size < 2:
        raise ValueError(f"genetic map must have >=2 valid points: {p}")

    return bp, cm



def interpolate_cm(marker_positions: np.ndarray, map_bp: np.ndarray, map_cm: np.ndarray) -> np.ndarray:
    if marker_positions.size == 0:
        return np.array([], dtype=float)
    return np.interp(marker_positions.astype(float), map_bp.astype(float), map_cm.astype(float))



def switch_probs_from_positions(marker_positions: np.ndarray, map_bp: np.ndarray, map_cm: np.ndarray, mean_tract_cm: float) -> np.ndarray:
    if marker_positions.size < 2:
        return np.array([], dtype=np.float32)
    cm = interpolate_cm(marker_positions, map_bp, map_cm)
    deltas = np.diff(cm)
    deltas = np.clip(deltas, 0.0, None)
    probs = 1.0 - np.exp(-deltas / float(mean_tract_cm))
    return probs.astype(np.float32)
