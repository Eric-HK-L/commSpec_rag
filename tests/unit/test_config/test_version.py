"""版本号一致性 — src.__version__ 与 pyproject.toml 必须同步."""

import tomllib
from pathlib import Path

from src import __version__


class TestVersionConsistency:

    def test_src_version_matches_pyproject(self):
        root = Path(__file__).resolve().parents[3]
        with open(root / "pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        assert data["project"]["version"] == __version__

    def test_version_is_semver(self):
        parts = __version__.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)
