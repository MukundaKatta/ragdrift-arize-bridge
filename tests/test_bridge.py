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


# MMD over small windows (200 baseline vs 32-64 current, dim 16) has a
# finite-sample noise floor around 0.06-0.10 even when both samples come from
# the SAME distribution. A genuine +1.5 mean shift scores ~1.0. We pick a
# threshold in the valley between the two so "no drift" stays quiet while a
# real shift fires.
_EMBEDDING_THRESHOLD = 0.5


def _bridge_with_baseline(seed: int = 0) -> DriftSpanProcessor:
    rng = np.random.default_rng(seed)
    return DriftSpanProcessor(
        DriftConfig(
            baseline_embeddings=rng.normal(size=(200, 16)).astype(np.float32),
            embedding_threshold=_EMBEDDING_THRESHOLD,
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
    assert bridge._cache.embedding > _EMBEDDING_THRESHOLD


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


def test_response_dimension_is_scored_when_baseline_present() -> None:
    """A response_lengths baseline plus observed lengths populates that scalar."""
    rng = np.random.default_rng(0)
    bridge = DriftSpanProcessor(
        DriftConfig(
            baseline_embeddings=rng.normal(size=(200, 16)).astype(np.float32),
            baseline_response_lengths=rng.normal(loc=50, scale=10, size=(200,)).astype(np.float64),
            embedding_threshold=_EMBEDDING_THRESHOLD,
            response_threshold=0.20,
            flush_every=32,
        )
    )
    rng2 = np.random.default_rng(1)
    for _ in range(64):
        # Shift response lengths well above the baseline so the KS stat fires.
        bridge.observe(
            rng2.normal(size=(16,)).astype(np.float32),
            response_length=float(rng2.normal(loc=120, scale=10)),
        )
    bridge.force_flush()
    assert bridge._cache.response > 0.0
    assert bridge._cache.response_exceeded is True


def test_force_flush_returns_true_and_shutdown_is_noop() -> None:
    bridge = _bridge_with_baseline()
    assert bridge.force_flush() is True
    assert bridge.on_end(FakeSpan()) is None
    assert bridge.shutdown() is None
