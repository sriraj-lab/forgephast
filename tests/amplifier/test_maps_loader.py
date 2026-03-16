from __future__ import annotations

from pathlib import Path

from forgephast.amplifier.maps import load_genetic_map


def test_load_genetic_map_parses_whitespace_plink_format(tmp_path: Path) -> None:
    map_path = tmp_path / "plink.chrchr21.GRCh38.map"
    map_path.write_text(
        "chr21 . 0.5000 100\n"
        "chr21 . 0.7000 200\n"
        "chr21 . 0.9000 300\n",
        encoding="utf-8",
    )

    bp, cm = load_genetic_map(map_path)
    assert bp.tolist() == [100, 200, 300]
    assert cm.tolist() == [0.5, 0.7, 0.9]

