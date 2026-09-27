"""Re-chunk web_chunks.csv through clean_text + the current chunker."""
from __future__ import annotations

import csv
from pathlib import Path

from phase2_web.rechunk import rechunk_web_csv, stitch_chunk_texts


def test_stitch_drops_overlap_prefix():
    a = "The seventh house and Saturn describe marriage delay."
    b = "Saturn describe marriage delay. The star lord must agree."
    out = stitch_chunk_texts([a, b])
    assert out.count("Saturn describe marriage delay.") == 1
    assert "star lord" in out


def test_rechunk_csv_cleans_detects_lang_and_skips_sitemap(tmp_path: Path):
    csv_path = tmp_path / "web_chunks.csv"
    article = (
        "The <mark>sub sub lord</mark> of the seventh house is Saturn. "
        "KP horary judges the star lord with the significators of marriage. "
    ) * 6
    sitemap_text = "Cafe Astrology site map. " + " ".join(
        f"Link {i} to an article." for i in range(20)
    )
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "chunk_id", "text", "source_type", "source_id", "chunk_index",
            "title", "url", "lang",
        ])
        w.writeheader()
        w.writerow({
            "chunk_id": "a0", "text": article, "source_type": "web",
            "source_id": "https://cafeastrology.com/saturn.html",
            "chunk_index": "0", "title": "Saturn",
            "url": "https://cafeastrology.com/saturn.html", "lang": "",
        })
        w.writerow({
            "chunk_id": "s0", "text": sitemap_text, "source_type": "web",
            "source_id": "https://cafeastrology.com/sitemap.html",
            "chunk_index": "0", "title": "Site Map",
            "url": "https://cafeastrology.com/sitemap.html", "lang": "",
        })
    chunks = rechunk_web_csv(csv_path, quality_cfg={"enabled": True})
    assert chunks
    assert all(c.source_type == "web" for c in chunks)
    assert all("sitemap" not in c.source_id for c in chunks)
    joined = "\n".join(c.text for c in chunks)
    assert "<mark>" not in joined
    assert all(c.lang and c.lang.strip() for c in chunks)
    assert all(c.content_hash for c in chunks)
