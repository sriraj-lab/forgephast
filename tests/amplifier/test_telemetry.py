from __future__ import annotations

import logging

from forgephast.amplifier.telemetry import resource_snapshot, telemetry_span


def test_resource_snapshot_never_raises_and_has_min_fields():
    snap = resource_snapshot()
    assert isinstance(snap, dict)
    assert "ts_utc" in snap
    assert "pid" in snap
    assert isinstance(snap["pid"], int)


def test_telemetry_span_emits_start_and_end(caplog):
    caplog.set_level(logging.INFO, logger="forgephast.amplifier.test.telemetry")
    log = logging.getLogger("forgephast.amplifier.test.telemetry")
    with telemetry_span(log, "unit-test", chrom=21, pop="P1"):
        log.info("inside-span")

    messages = [rec.getMessage() for rec in caplog.records if rec.name == "forgephast.amplifier.test.telemetry"]
    assert any(msg.startswith("SUBSTEP start name=unit-test") for msg in messages)
    assert any(msg.startswith("SUBSTEP end name=unit-test") and "elapsed_sec=" in msg for msg in messages)

