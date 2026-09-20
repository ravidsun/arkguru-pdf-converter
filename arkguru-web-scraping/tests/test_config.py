"""Config pins schema web and the four free-resource seeds."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_datastore_yaml_is_schema_web():
    raw = yaml.safe_load((ROOT / "config" / "datastore.yaml").read_text())
    pg = raw["datastore"]["postgres"]
    assert pg["schema"] == "web"
    assert pg["chunks_table"] == "chunks"
    assert pg["dsn_env"] == "PG_DSN"


def test_config_has_four_learning_seeds():
    raw = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())
    seeds = raw["phase2"]["seeds"]
    joined = " ".join(seeds).lower()
    assert "vedicastrologer.org" in joined
    assert "astrolearn.co" in joined
    assert "cosmicinsights.net" in joined
    assert "appliedjyotish.com" in joined
    assert any("/jh/index.htm" in s for s in seeds)
    # commercial portals stay out of first pass
    assert "astrotalk.com" not in joined
    assert "astrosage.com" not in joined
    assert raw["phase2"]["max_pages_per_seed"] == 80
    assert raw["phase2"]["same_domain_only"] is True
