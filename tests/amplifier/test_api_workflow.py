from __future__ import annotations

from forgephast.amplifier.api import build_cache, build_manifest

from .conftest import write_test_config



def test_api_build_manifest_and_cache(tmp_path):
    cfg_path = write_test_config(tmp_path, name="api_flow")

    manifest = build_manifest(cfg_path)
    assert manifest.manifest_path.exists()

    cache = build_cache(cfg_path, chroms=[21])
    assert cache.entries
    assert all(entry.cache_path.exists() for entry in cache.entries)
