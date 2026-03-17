from __future__ import annotations

from hapstore.ids import align_by_ids, read_sample_ids

from forgephast.phenotype.covariates import model as covariates_model
from forgephast.phenotype.data import global_pcs


def test_forgephast_uses_hapstore_id_helpers() -> None:
    assert covariates_model.align_by_ids is align_by_ids
    assert global_pcs.align_by_ids is align_by_ids
    assert global_pcs.read_sample_ids is read_sample_ids
