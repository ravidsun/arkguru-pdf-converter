"""Config pins schema web and the astrology article-seed allowlist."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

FIRST_PASS_HOSTS = (
    "vedicastrologer.org",
    "astrolearn.co",
    "cosmicinsights.net",
    "appliedjyotish.com",
    "barbarapijan.com",
    "lightonvedicastrology.com",
    "astrojyoti.com",
    "jyotishvidya.com",
    "siddhantika.com",
    "wisdomlib.org",
    "sacred-texts.com",
    "skyscript.co.uk",
    "cafeastrology.com",
    "astro.com",
    "sevenstarsastrology.com",
    "theastrologyplace.com",
    "constellationsofwords.com",
)

COMMERCIAL_PORTALS = (
    "astrotalk.com",
    "astrosage.com",
    "ganeshaspeaks.com",
    "clickastro.com",
    "astroved.com",
    "astro-seek.com",
)


def test_datastore_yaml_is_schema_web():
    raw = yaml.safe_load((ROOT / "config" / "datastore.yaml").read_text())
    pg = raw["datastore"]["postgres"]
    assert pg["schema"] == "web"
    assert pg["chunks_table"] == "chunks"
    assert pg["dsn_env"] == "PG_DSN"


def test_config_has_astrology_article_seeds():
    raw = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())
    seeds = raw["phase2"]["seeds"]
    joined = " ".join(seeds).lower()
    for host in FIRST_PASS_HOSTS:
        assert host in joined, host
    assert any("/jh/index.htm" in s for s in seeds)
    assert any("brihat-parashara-hora-shastra" in s for s in seeds)
    assert any("/astrology/" in s for s in seeds)
    for host in COMMERCIAL_PORTALS:
        assert host not in joined, host
    assert raw["domain_name"] == "vedic-astrology"
    assert raw["phase2"]["max_pages_per_seed"] == 80
    assert raw["phase2"]["max_pages"] >= len(seeds) * raw["phase2"]["max_pages_per_seed"]
    assert raw["phase2"]["same_domain_only"] is True
