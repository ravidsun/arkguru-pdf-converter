from backend.paths import discover


def test_discover_finds_umbrella_and_phase2():
    layout = discover()
    assert layout.umbrella.name == "workspace" or (layout.umbrella / "arkguru-ui").is_dir()
    assert layout.ui.name == "arkguru-ui"
    assert layout.phase2 is not None
    assert (layout.phase2 / "phase2_web" / "pipeline.py").exists()
    assert layout.common is not None


def test_pythonpath_includes_common():
    layout = discover()
    pp = layout.pythonpath(layout.phase2)
    assert layout.common is not None
    assert str(layout.common) in pp
    assert str(layout.phase2) in pp
