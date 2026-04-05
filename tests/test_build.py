"""Tests for the Tribal Village build helpers."""

from pathlib import Path

import tribal_village_env
from tribal_village_env import build


def test_collect_source_files_limits_rebuild_inputs():
    root = Path(__file__).resolve().parent.parent
    rel_paths = {path.relative_to(root).as_posix() for path in build._collect_source_files(root)}

    assert "tribal_village.nim" in rel_paths
    assert "tribal_village.nimble" in rel_paths
    assert any(path.startswith("src/") for path in rel_paths)
    assert not any(path.startswith("tests/") for path in rel_paths)
    assert not any(path.startswith("scripts/") for path in rel_paths)


def test_package_exports_are_lazy():
    assert "TribalVillageEnv" not in tribal_village_env.__dict__

    ensure = tribal_village_env.ensure_nim_library_current

    assert callable(ensure)
    assert "ensure_nim_library_current" in tribal_village_env.__dict__
