from pathlib import Path

import pytest

from healthrag import runtime


def test_available_memory_is_read_from_meminfo(monkeypatch, tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 8388608 kB\nMemAvailable: 4194304 kB\n")
    original_open = open

    def controlled_open(path, *args, **kwargs):
        return original_open(meminfo if path == "/proc/meminfo" else path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", controlled_open)
    assert runtime.available_memory_gib() == 4.0


def test_guard_fails_before_a_heavy_stage(monkeypatch):
    monkeypatch.setattr(runtime, "available_memory_gib", lambda: 0.5)
    with pytest.raises(RuntimeError, match="requires at least 2.5 GiB"):
        runtime.require_memory("Hybrid retrieval", 2.5)


def test_guard_can_be_delegated_to_an_external_scheduler(monkeypatch):
    monkeypatch.setenv("HEALTHRAG_SKIP_MEMORY_GUARD", "1")
    monkeypatch.setattr(runtime, "available_memory_gib", lambda: 0.0)
    runtime.require_memory("Hybrid retrieval", 2.5)
