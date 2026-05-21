"""Tests for the ragdrift × Arize bridge.

Note: these import ragdrift, so they need a built wheel. The CI sets up
ragdrift-py via pip install before running pytest.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from ragdrift_arize import DriftConfig, DriftSpanProcessor
except RuntimeError:
    pytest.skip("ragdrift-py not installed", allow_module_level=True)


class FakeSpan:
    """Minimal stand-in for an OTel span. We only need set_attribute."""

    def __init__(self) -> None:
        self.attrs: dict[str, Any] = {}

    def set_attribute(self, key: str, value: Any) -> None:
        self.attrs[key] = value


def _bridge_with_baseline(seed: int = 0) -> DriftSpanProcessor:
    rng = np.random.default_rng(seed)
    return DriftSpanProcessor(
        DriftConfig(
            baseline_embeddings=rng.normal(size=(200, 16)).astype(np.float32),
            embedding_threshold=0.05,
            flush_every=32,
        )
    )


def test_baseline_run_does_not_exceed_threshold() -> None:
    bridge = _bridge_with_baseline()
    rng = np.random.default_rng(1)
    for _ in range(64):
        bridge.observe(rng.normal(size=(16,)).astype(np.float32))
    bridge.force_flush()
    assert bridge._cache.embedding_exceeded is False


def test_shifted_input_does_exceed_threshold() -> None:
    bridge = _bridge_with_baseline()
    rng = np.random.default_rng(1)
    for _ in range(64):
        bridge.observe(rng.normal(loc=1.5, size=(16,)).astype(np.float32))
    bridge.force_flush()
    assert bridge._cache.embedding_exceeded is True
    assert bridge._cache.embedding > 0.05


def test_on_start_sets_span_attributes() -> None:
    bridge = _bridge_with_baseline()
    bridge.force_flush()
    span = FakeSpan()
    bridge.on_start(span)
    assert "drift.embedding" in span.attrs
    assert "drift.embedding.exceeded" in span.attrs
    assert isinstance(span.attrs["drift.embedding"], float)
    assert isinstance(span.attrs["drift.embedding.exceeded"], bool)


def test_no_observe_yields_zero_drift() -> None:
    bridge = _bridge_with_baseline()
    span = FakeSpan()
    bridge.on_start(span)
    assert span.attrs["drift.embedding"] == 0.0
    assert span.attrs["drift.embedding.exceeded"] is False


def test_flush_every_triggers_refresh() -> None:
    bridge = _bridge_with_baseline()
    rng = np.random.default_rng(1)
    # observe one less than flush_every — no refresh yet.
    for _ in range(31):
        bridge.observe(rng.normal(loc=1.5, size=(16,)).astype(np.float32))
    assert bridge._cache.embedding == 0.0
    # 32nd observation triggers the refresh.
    bridge.observe(rng.normal(loc=1.5, size=(16,)).astype(np.float32))
    assert bridge._cache.embedding > 0.0
