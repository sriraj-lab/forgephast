from __future__ import annotations

import logging

from forgephast.amplifier.config import from_yaml
from forgephast.amplifier.donors import build_manifest_from_config
from forgephast.amplifier.simulation import simulate_chromosome_population_shard_pool
from forgephast.amplifier.utils import setup_logging

from .conftest import write_test_config


def test_simulate_pool_spawn_queue_logging_writes_worker_substeps(tmp_path):
    log_path = tmp_path / "popamp_run.log"

    root = logging.getLogger()
    old_level = root.level
    old_handlers = list(root.handlers)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    try:
        setup_logging(
            log_file=log_path,
            reset_handlers=True,
            stream_stdout=False,
            logger_name=None,
        )

        cfg_path = write_test_config(tmp_path, name="sim_pool_mp_queue_log")
        cfg = from_yaml(cfg_path)
        build_manifest_from_config(cfg)

        simulate_chromosome_population_shard_pool(
            cfg,
            chrom=21,
            n_shards=2,
            task_workers=2,
            threads_per_task=1,
            task_start_method="spawn",
        )

        assert log_path.exists()
        content = log_path.read_text(encoding="utf-8")
        assert "simulate-pool mp logging enabled" in content
        assert "SUBSTEP start name=simulate-pool.task" in content
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
        for handler in old_handlers:
            root.addHandler(handler)
        root.setLevel(old_level)
