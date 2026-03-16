"""Sample-ID and table alignment helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd


def read_sample_ids(path: str | Path) -> List[str]:
    """Read sample IDs from a one- or two-column file.

    Accepted formats:
    - one-column IID list (optionally header #IID/IID)
    - two-column FID IID table
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"sample ID file not found: {path}")

    rows: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 1:
                token = parts[0]
                if token.upper() in {"IID", "#IID"}:
                    continue
                rows.append(token)
            else:
                fid, iid = parts[0], parts[1]
                if fid.upper() == "FID" and iid.upper() == "IID":
                    continue
                rows.append(iid)

    if not rows:
        raise ValueError(f"no sample IDs parsed from {path}")
    return rows


def ensure_id_column(df: pd.DataFrame, preferred: Iterable[str] = ("IID", "#IID", "ID")) -> str:
    """Return the first present ID column name from preferred candidates."""
    for name in preferred:
        if name in df.columns:
            return name
    raise ValueError(f"No ID column found. Tried {list(preferred)} in columns {list(df.columns)}")


def align_by_ids(df: pd.DataFrame, id_column: str, ids: List[str], context: str) -> pd.DataFrame:
    """Align dataframe rows to explicit ID ordering; fail on missing IDs."""
    if id_column not in df.columns:
        raise ValueError(f"{context}: missing ID column {id_column}")

    indexed = df.set_index(id_column)
    missing = [sample_id for sample_id in ids if sample_id not in indexed.index]
    if missing:
        raise ValueError(
            f"{context}: missing {len(missing)} required IDs; first few: {missing[:5]}"
        )
    out = indexed.loc[ids].reset_index()
    return out

