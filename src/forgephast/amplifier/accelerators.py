"""Optional accelerator discovery utilities."""

from __future__ import annotations


def has_numba() -> bool:
    try:
        import numba  # noqa: F401

        return True
    except Exception:
        return False



def has_cython_extension() -> bool:
    try:
        from . import _cykernels  # noqa: F401

        return True
    except Exception:
        return False
