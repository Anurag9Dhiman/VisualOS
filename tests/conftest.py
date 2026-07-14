"""Shared pytest fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src import db as db_module
from src import rate_limiter


@pytest.fixture()
def tmp_db(monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point db.DB_PATH at a fresh temp file for each test."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    monkeypatch.setattr(db_module, "DB_PATH", path)
    db_module.init_db(path)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def clear_rate_limiter():
    """Reset rate-limiter state before every test."""
    rate_limiter.reset()
    yield
    rate_limiter.reset()
