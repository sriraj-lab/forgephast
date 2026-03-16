from __future__ import annotations

import numpy as np

from forgephast.amplifier.simulation import _simulate_dispatch, _simulate_kernel_numpy



def test_kernel_parity_numba_vs_numpy():
    rng = np.random.default_rng(42)
    donor_haps = rng.integers(0, 2, size=(6, 12), dtype=np.uint8)
    switch_probs = rng.random(11, dtype=np.float32)
    init_donors = rng.integers(0, donor_haps.shape[0], size=10, dtype=np.int32)
    u = rng.random((10, 11), dtype=np.float32)
    replacement = rng.integers(0, donor_haps.shape[0], size=(10, 11), dtype=np.int32)

    out_np = _simulate_kernel_numpy(donor_haps, switch_probs, init_donors, u, replacement)
    out_dispatch = _simulate_dispatch(
        donor_haps=donor_haps,
        switch_probs=switch_probs,
        init_donors=init_donors,
        u=u,
        replacement=replacement,
        use_parallel_kernel=False,
        threads=1,
    )

    assert np.array_equal(out_np, out_dispatch)
