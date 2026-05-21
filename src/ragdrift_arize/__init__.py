"""ragdrift × Arize bridge.

A thin OpenTelemetry SpanProcessor that batches predictions, runs
``ragdrift.RagDriftMonitor.evaluate()`` every N batches, and writes the
five scalars onto the next outgoing span as attributes::

    otel.attr.drift.embedding = 1.0324
    otel.attr.drift.embedding.exceeded = true
    otel.attr.drift.data = 0.012
    otel.attr.drift.response = 0.34
    otel.attr.drift.confidence = 0.08
    otel.attr.drift.query = 0.07

Phoenix collects the span. Arize routes the alert. Lambda-friendly because
the math is in Rust (PyO3 binding to ``ragdrift-py``).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Sequence

import numpy as np

try:
    from ragdrift import RagDriftMonitor  # type: ignore[import-untyped]
except ImportError as e:  # pragma: no cover
    raise RuntimeError(
        "ragdrift-py is required. Install with: pip install ragdrift-py>=0.1.4"
    ) from e


__version__ = "0.1.0"


@dataclass
class DriftConfig:
    """Configure the bridge.

    ``baseline_embeddings`` is required to enable embedding drift. The
    other arrays are optional — the monitor only evaluates dimensions
    whose baseline is present, mirroring ragdrift's own behavior.

    ``flush_every`` controls how often the bridge actually re-runs the
    drift math. With ``flush_every=64``, the SpanProcessor accumulates
    64 prediction embeddings before computing a fresh report. The
    previous report's scalars are written to every intervening span.
    """

    baseline_embeddings: np.ndarray
    baseline_features: np.ndarray | None = None
    baseline_response_lengths: np.ndarray | None = None
    baseline_confidence_scores: np.ndarray | None = None
    baseline_query_embeddings: np.ndarray | None = None

    embedding_threshold: float | None = 0.05
    data_threshold: float | None = 0.10
    response_threshold: float | None = 0.20
    confidence_threshold: float | None = 0.20
    query_threshold: float | None = 0.10

    flush_every: int = 64


@dataclass
class _Cache:
    """Holds the most recent report values for span emission."""

    embedding: float = 0.0
    data: float = 0.0
    response: float = 0.0
    confidence: float = 0.0
    query: float = 0.0
    embedding_exceeded: bool = False
    data_exceeded: bool = False
    response_exceeded: bool = False
    confidence_exceeded: bool = False
    query_exceeded: bool = False

    def to_attrs(self) -> dict[str, Any]:
        return {
            "drift.embedding": round(self.embedding, 6),
            "drift.embedding.exceeded": self.embedding_exceeded,
            "drift.data": round(self.data, 6),
            "drift.data.exceeded": self.data_exceeded,
            "drift.response": round(self.response, 6),
            "drift.response.exceeded": self.response_exceeded,
            "drift.confidence": round(self.confidence, 6),
            "drift.confidence.exceeded": self.confidence_exceeded,
            "drift.query": round(self.query, 6),
            "drift.query.exceeded": self.query_exceeded,
        }


class DriftSpanProcessor:
    """Add ragdrift attributes to spans Phoenix is already collecting.

    Conforms loosely to the OpenTelemetry SpanProcessor interface. The
    real SpanProcessor protocol from opentelemetry-sdk is implemented as
    a Protocol-style class with on_start / on_end / shutdown methods;
    we match the shape so existing Phoenix exporters pick it up.

    Usage::

        from opentelemetry.sdk.trace import TracerProvider
        provider = TracerProvider()
        provider.add_span_processor(DriftSpanProcessor(config))

    Then ingest predictions::

        bridge = provider._active_span_processor  # or capture the handle
        bridge.observe(prediction_embedding, response_length=42, confidence=0.83)
    """

    def __init__(self, config: DriftConfig) -> None:
        self._config = config
        self._monitor = RagDriftMonitor(
            embedding_threshold=config.embedding_threshold,
            data_threshold=config.data_threshold,
            response_threshold=config.response_threshold,
            confidence_threshold=config.confidence_threshold,
            query_threshold=config.query_threshold,
        )
        # Sliding window over recent predictions, capped at flush_every*4 so
        # memory stays bounded even with skew between observe() and flush.
        self._buf_emb: Deque[np.ndarray] = deque(maxlen=config.flush_every * 4)
        self._buf_resp: Deque[float] = deque(maxlen=config.flush_every * 4)
        self._buf_conf: Deque[float] = deque(maxlen=config.flush_every * 4)
        self._cache = _Cache()

    # --- ingestion API --------------------------------------------------

    def observe(
        self,
        embedding: np.ndarray | Sequence[float],
        *,
        response_length: float | None = None,
        confidence: float | None = None,
    ) -> None:
        """Record one prediction. Triggers a re-evaluation every ``flush_every``."""
        emb = np.asarray(embedding, dtype=np.float32)
        self._buf_emb.append(emb)
        if response_length is not None:
            self._buf_resp.append(float(response_length))
        if confidence is not None:
            self._buf_conf.append(float(confidence))
        if len(self._buf_emb) % self._config.flush_every == 0:
            self._refresh()

    def _refresh(self) -> None:
        if not self._buf_emb:
            return
        current_emb = np.stack(list(self._buf_emb), axis=0)
        kwargs: dict[str, Any] = {
            "baseline_embeddings": self._config.baseline_embeddings,
            "current_embeddings": current_emb,
        }
        if self._config.baseline_response_lengths is not None and self._buf_resp:
            kwargs["baseline_response_lengths"] = self._config.baseline_response_lengths
            kwargs["current_response_lengths"] = np.asarray(list(self._buf_resp), dtype=np.float64)
        if self._config.baseline_confidence_scores is not None and self._buf_conf:
            kwargs["baseline_confidence_scores"] = self._config.baseline_confidence_scores
            kwargs["current_confidence_scores"] = np.asarray(list(self._buf_conf), dtype=np.float64)

        report = self._monitor.evaluate(**kwargs)
        for score in report.scores:
            dim = score.dimension.value
            setattr(self._cache, dim, score.score)
            setattr(self._cache, f"{dim}_exceeded", score.exceeded)

    # --- OpenTelemetry SpanProcessor protocol --------------------------

    def on_start(self, span: Any, parent_context: Any = None) -> None:  # noqa: D401
        """Set the cached drift attributes on every span as it starts."""
        for k, v in self._cache.to_attrs().items():
            span.set_attribute(k, v)

    def on_end(self, span: Any) -> None:  # noqa: D401
        """No-op. The drift attributes were already set on_start."""
        return None

    def shutdown(self) -> None:  # noqa: D401
        """No-op. ragdrift has no background threads to stop."""
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:  # noqa: ARG002
        """Compute a fresh report immediately. Returns True."""
        self._refresh()
        return True


__all__ = [
    "DriftConfig",
    "DriftSpanProcessor",
    "__version__",
]
