"""ID alignment guard tests."""

from __future__ import annotations

import pandas as pd
import pytest

from forgephast.phenotype.data.genotypes import check_id_alignment


def test_id_alignment_detects_order_mismatch() -> None:
    sample_ids = ["s1", "s2", "s3"]
    df = pd.DataFrame(
        {
            "FID": ["s1", "s3", "s2"],
            "IID": ["s1", "s3", "s2"],
            "v1": [0, 1, 2],
        }
    )
    with pytest.raises(ValueError):
        check_id_alignment(sample_ids, df)

