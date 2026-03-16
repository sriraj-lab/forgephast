from __future__ import annotations

from forgephast.amplifier.utils import resolve_log_file_path, setup_logging

from .conftest import write_test_config


def test_setup_logging_writes_dedicated_file(tmp_path):
    log_path = tmp_path / "logs" / "runs" / "popamp_test.log"
    logger, active_log_file = setup_logging(
        log_file=log_path,
        logger_name="forgephast.amplifier.test.logging",
        reset_handlers=True,
    )
    logger.info("hello-log")

    assert active_log_file == log_path
    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8")
    assert "hello-log" in content


def test_resolve_log_file_path_uses_run_root_from_config(tmp_path):
    cfg_path = write_test_config(tmp_path)
    resolved = resolve_log_file_path(command="simulate-pool", config_path=cfg_path)

    expected_prefix = tmp_path / "outputs" / "test_run" / "logs" / "runs"
    assert resolved is not None
    assert str(resolved).startswith(str(expected_prefix))
    assert resolved.name.startswith("popamp_simulate-pool_")


def test_resolve_log_file_path_can_disable_file_logging(tmp_path):
    cfg_path = write_test_config(tmp_path)
    resolved = resolve_log_file_path(
        command="qc",
        config_path=cfg_path,
        disable_file_logging=True,
    )
    assert resolved is None
